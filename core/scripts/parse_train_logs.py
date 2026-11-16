#!/usr/bin/env python3
"""
parse_train_logs.py — 扒 MNER train.log 里的 per-class P/R/F1，输出 CSV + Markdown。

支持的日志格式（自动探测，按优先级）：
  1) seqeval classification_report
       "              precision    recall  f1-score   support"
       "         PER     0.9234    0.9101    0.9167       234"
       数值范围 [0, 1]
  2) conlleval 风格
       "accuracy: 99.04%; precision: 78.23%; recall: 79.45%; FB1: 78.83"
       "              PER: precision:  88.45%; recall:  91.23%; FB1: 89.81  234"
       数值范围 [0, 100]
  3) key-value 兜底
       "test_PER_f1: 0.9167"  /  "PER_f1=0.9167"  /  "PER f1 0.9167"

选块策略：每个 log 内取 overall F1 最高的那个 report 块（best-of-all-epochs）。
若日志里同时有 dev 和 test 块，要求脚本只取 test：用 --test_only （需要 log 里有 "test"/"Test" 标记）。

用法
----
扫所有 run：
  python parse_train_logs.py --logs_root outputs \
      --out_csv outputs/per_class_summary.csv \
      --out_md  outputs/per_class_summary.md

只看单个 log（诊断）：
  python parse_train_logs.py --single outputs/some_run/train.log
  python parse_train_logs.py --inspect outputs/some_run/train.log
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

LABELS = ["PER", "LOC", "ORG", "MISC"]

# ---------- 格式 1: seqeval ----------
SEQEVAL_HEADER_RE = re.compile(r"precision\s+recall\s+f1-score\s+support")
SEQEVAL_ROW_RE = re.compile(
    r"^\s*(?P<label>PER|LOC|ORG|MISC|micro avg|macro avg|weighted avg)"
    r"\s+(?P<p>[01]\.\d+)"
    r"\s+(?P<r>[01]\.\d+)"
    r"\s+(?P<f1>[01]\.\d+)"
    r"\s+(?P<sup>\d+)\s*$"
)

# ---------- 格式 2: conlleval ----------
CONLLEVAL_OVERALL_RE = re.compile(
    r"accuracy:\s*(?P<acc>[\d.]+)%;\s*"
    r"precision:\s*(?P<p>[\d.]+)%;\s*"
    r"recall:\s*(?P<r>[\d.]+)%;\s*"
    r"FB?1:\s*(?P<f1>[\d.]+)"
)
CONLLEVAL_ROW_RE = re.compile(
    r"^\s*(?P<label>PER|LOC|ORG|MISC):\s*"
    r"precision:\s*(?P<p>[\d.]+)%;\s*"
    r"recall:\s*(?P<r>[\d.]+)%;\s*"
    r"FB?1:\s*(?P<f1>[\d.]+)\s*(?P<sup>\d+)?"
)

# ---------- 格式 3: key-value ----------
# 抓 "PER_f1: 0.9167" / "test_PER f1=0.9167" / "PER f1 0.9167"
KV_PRF_RE = re.compile(
    r"(?<![A-Za-z])(?P<label>PER|LOC|ORG|MISC)"
    r"[ _\-]+"
    r"(?P<metric>precision|recall|f1|p|r)"
    r"\s*[:=]?\s*"
    r"(?P<v>[01]?\.\d+)",
    re.I,
)
KV_OVERALL_F1_RE = re.compile(
    r"(?:overall|micro|span|test|eval)[_\s\-]*"
    r"(?:f1|F1|fb1|FB1)\s*[:=]?\s*"
    r"(?P<v>[01]?\.\d+)"
)

EPOCH_RE = re.compile(r"[Ee]poch\s*[:=#]?\s*(?P<v>\d+)")
TEST_TAG_RE = re.compile(r"\b[Tt]est\b")


# ============================================================
# 解析
# ============================================================
def parse_seqeval_blocks(text):
    """返回 list of dict: {line_start, rows: {label: {p,r,f1,sup}}}"""
    lines = text.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        if not SEQEVAL_HEADER_RE.search(line):
            continue
        rows = OrderedDict()
        # 向后扫 30 行，遇到非空非匹配且 rows 非空就停
        for j in range(i + 1, min(i + 35, len(lines))):
            m = SEQEVAL_ROW_RE.match(lines[j])
            if m:
                rows[m.group("label")] = {
                    "p": float(m.group("p")),
                    "r": float(m.group("r")),
                    "f1": float(m.group("f1")),
                    "sup": int(m.group("sup")),
                }
            elif lines[j].strip() == "":
                continue
            else:
                if rows:
                    break
        if rows:
            blocks.append({"line_start": i, "rows": rows})
    return blocks


def parse_conlleval_blocks(text):
    """返回 list of dict: {line_start, overall_*, rows: {label: {p,r,f1,sup}}}"""
    lines = text.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        m = CONLLEVAL_OVERALL_RE.search(line)
        if not m:
            continue
        block = {
            "line_start": i,
            "overall_p": float(m.group("p")) / 100,
            "overall_r": float(m.group("r")) / 100,
            "overall_f1": float(m.group("f1")) / 100,
            "rows": OrderedDict(),
        }
        # 后 8 行内找 per-class
        for j in range(i + 1, min(i + 10, len(lines))):
            mm = CONLLEVAL_ROW_RE.match(lines[j])
            if mm:
                block["rows"][mm.group("label")] = {
                    "p": float(mm.group("p")) / 100,
                    "r": float(mm.group("r")) / 100,
                    "f1": float(mm.group("f1")) / 100,
                    "sup": int(mm.group("sup")) if mm.group("sup") else 0,
                }
        if block["rows"]:
            blocks.append(block)
    return blocks


def parse_kv_blocks(text):
    """兜底：滑窗内若能凑齐 ≥3 类的 F1，记为一个块"""
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        win_hi = min(i + 12, len(lines))
        rows = OrderedDict()
        for j in range(i, win_hi):
            for m in KV_PRF_RE.finditer(lines[j]):
                lab = m.group("label").upper()
                metric = m.group("metric").lower()
                if metric.startswith("p"):
                    norm = "p"
                elif metric.startswith("r"):
                    norm = "r"
                else:
                    norm = "f1"
                rows.setdefault(lab, {})[norm] = float(m.group("v"))
        # 至少 3 类有 F1
        complete = sum(1 for L in LABELS if L in rows and "f1" in rows[L])
        if complete >= 3:
            # 补 sup=0
            for lab, vals in rows.items():
                vals.setdefault("sup", 0)
            blocks.append({"line_start": i, "rows": rows})
            i = win_hi
        else:
            i += 1
    return blocks


def find_overall_near(text_lines, line_start, win_before=6, win_after=12):
    """块附近找 overall F1（KV 格式）"""
    lo = max(0, line_start - win_before)
    hi = min(len(text_lines), line_start + win_after)
    candidates = []
    for j in range(lo, hi):
        m = KV_OVERALL_F1_RE.search(text_lines[j])
        if m:
            v = float(m.group("v"))
            # KV 可能写成 78.83 也可能 0.7883；> 1 视为百分数
            if v > 1.0:
                v = v / 100
            candidates.append(v)
    return max(candidates) if candidates else None


def find_nearest_epoch(text_lines, line_start, look_back=200):
    """向前找最近的 epoch 号"""
    stop = max(-1, line_start - look_back - 1)
    for j in range(line_start, stop, -1):
        if 0 <= j < len(text_lines):
            m = EPOCH_RE.search(text_lines[j])
            if m:
                return int(m.group("v"))
    return None


def is_test_block(text_lines, line_start, look_back=8, look_ahead=4):
    """块附近有没有 test 标记"""
    lo = max(0, line_start - look_back)
    hi = min(len(text_lines), line_start + look_ahead)
    for j in range(lo, hi):
        if TEST_TAG_RE.search(text_lines[j]):
            return True
    return False


def enrich_block(block, text_lines, fmt):
    """补 overall_f1 / overall_p / overall_r"""
    if fmt == "conlleval":
        return block  # 已经有 overall
    rows = block["rows"]
    if "micro avg" in rows:
        block["overall_p"] = rows["micro avg"]["p"]
        block["overall_r"] = rows["micro avg"]["r"]
        block["overall_f1"] = rows["micro avg"]["f1"]
    elif "weighted avg" in rows:
        block["overall_p"] = rows["weighted avg"]["p"]
        block["overall_r"] = rows["weighted avg"]["r"]
        block["overall_f1"] = rows["weighted avg"]["f1"]
    else:
        # KV 模式：附近找 overall F1
        v = find_overall_near(text_lines, block["line_start"])
        if v is not None:
            block["overall_f1"] = v
    return block


def pick_best_block(blocks, prefer_test=False, text_lines=None):
    """选 overall F1 最大的块"""
    if prefer_test and text_lines is not None:
        test_blocks = [b for b in blocks if is_test_block(text_lines, b["line_start"])]
        if test_blocks:
            blocks = test_blocks
    blocks_with_score = [b for b in blocks if b.get("overall_f1") is not None]
    if not blocks_with_score:
        return None
    return max(blocks_with_score, key=lambda b: b["overall_f1"])


def extract_from_log(log_path, test_only=False):
    """返回 dict，包含 overall + 4 类 P/R/F1 + 诊断字段"""
    try:
        text = Path(log_path).read_text(errors="ignore")
    except Exception as e:
        return {"error": f"read failed: {e}"}

    text_lines = text.splitlines()

    # 按优先级试 3 种格式
    blocks = parse_seqeval_blocks(text)
    fmt = "seqeval" if blocks else None
    if not blocks:
        blocks = parse_conlleval_blocks(text)
        fmt = "conlleval" if blocks else None
    if not blocks:
        blocks = parse_kv_blocks(text)
        fmt = "kv" if blocks else None

    if not blocks:
        return {
            "error": "no per-class report block found",
            "format": None,
            "n_blocks": 0,
            "log_lines": len(text_lines),
        }

    for b in blocks:
        enrich_block(b, text_lines, fmt)

    best = pick_best_block(blocks, prefer_test=test_only, text_lines=text_lines)
    if best is None:
        return {
            "error": "no overall F1 found in any block",
            "format": fmt,
            "n_blocks": len(blocks),
            "log_lines": len(text_lines),
        }

    out = {
        "format": fmt,
        "n_blocks": len(blocks),
        "best_block_line": best["line_start"],
        "best_epoch": find_nearest_epoch(text_lines, best["line_start"]),
        "overall_p": best.get("overall_p"),
        "overall_r": best.get("overall_r"),
        "overall_f1": best.get("overall_f1"),
    }
    for L in LABELS:
        if L in best["rows"]:
            out[f"{L}_p"] = best["rows"][L].get("p")
            out[f"{L}_r"] = best["rows"][L].get("r")
            out[f"{L}_f1"] = best["rows"][L].get("f1")
        else:
            out[f"{L}_p"] = None
            out[f"{L}_r"] = None
            out[f"{L}_f1"] = None
    return out


# ============================================================
# Inspect 模式：把含关键字的行打印出来，给 Claude/人看格式用
# ============================================================
def inspect_log(log_path, max_lines=80):
    keywords = re.compile(
        r"\b(PER|LOC|ORG|MISC)\b|precision|recall|\bf1\b|\bF1\b|FB1|overall|micro avg|"
        r"best|Epoch|epoch",
        re.I,
    )
    try:
        text = Path(log_path).read_text(errors="ignore")
    except Exception as e:
        print(f"[ERR] cannot read {log_path}: {e}")
        return
    lines = text.splitlines()
    matched = [(i, ln.rstrip()) for i, ln in enumerate(lines) if keywords.search(ln)]
    print(f"=== INSPECT {log_path} ===")
    print(f"total_lines={len(lines)}  matched_lines={len(matched)}")
    print("-" * 60)
    if len(matched) <= max_lines:
        for i, ln in matched:
            print(f"L{i+1}: {ln}")
    else:
        head = max_lines // 2
        tail = max_lines - head
        for i, ln in matched[:head]:
            print(f"L{i+1}: {ln}")
        print(f"... [omitted {len(matched) - max_lines} lines] ...")
        for i, ln in matched[-tail:]:
            print(f"L{i+1}: {ln}")


# ============================================================
# I/O
# ============================================================
def collect_logs(logs_root):
    root = Path(logs_root)
    if not root.exists():
        return []
    # 标准位置：outputs/<run>/train.log
    candidates = list(root.glob("*/train.log"))
    # 兜底：outputs/<run>/*.log
    if not candidates:
        candidates = [
            p for p in root.glob("*/*.log") if "train" in p.name.lower() or "console" in p.name.lower()
        ]
    return sorted(candidates)


def fmt_v(v, pct=True):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if pct and isinstance(v, float) and v <= 1.5:
            return f"{v*100:.2f}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return str(v)


def write_csv(rows, path):
    fieldnames = [
        "run", "best_epoch",
        "overall_p", "overall_r", "overall_f1",
        "PER_p", "PER_r", "PER_f1",
        "LOC_p", "LOC_r", "LOC_f1",
        "ORG_p", "ORG_r", "ORG_f1",
        "MISC_p", "MISC_r", "MISC_f1",
        "format", "n_blocks", "best_block_line", "log_path", "error",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows, path):
    md = []
    md.append("# Per-class summary\n")
    md.append(f"`{len(rows)}` runs scanned. All F1/P/R reported in percent (×100).\n")

    md.append("\n## F1 only\n")
    md.append("| run | best_ep | overall | PER | LOC | ORG | MISC |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x.get("overall_f1") or 0), reverse=True):
        md.append(
            f"| `{r.get('run','?')}` "
            f"| {r.get('best_epoch') if r.get('best_epoch') is not None else '—'} "
            f"| {fmt_v(r.get('overall_f1'))} "
            f"| {fmt_v(r.get('PER_f1'))} "
            f"| {fmt_v(r.get('LOC_f1'))} "
            f"| {fmt_v(r.get('ORG_f1'))} "
            f"| {fmt_v(r.get('MISC_f1'))} |"
        )

    md.append("\n## Full P / R / F1\n")
    md.append("| run | overall (P/R/F1) | PER (P/R/F1) | LOC (P/R/F1) | ORG (P/R/F1) | MISC (P/R/F1) |")
    md.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x.get("overall_f1") or 0), reverse=True):
        def trio(prefix):
            return f"{fmt_v(r.get(f'{prefix}_p'))} / {fmt_v(r.get(f'{prefix}_r'))} / {fmt_v(r.get(f'{prefix}_f1'))}"
        md.append(
            f"| `{r.get('run','?')}` | {trio('overall')} | {trio('PER')} | {trio('LOC')} | {trio('ORG')} | {trio('MISC')} |"
        )

    md.append("\n## Diagnostics\n")
    md.append("| run | format | n_blocks | best_block_line | error |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| `{r.get('run','?')}` | {r.get('format') or '—'} | {r.get('n_blocks') or 0} "
            f"| {r.get('best_block_line') if r.get('best_block_line') is not None else '—'} "
            f"| {r.get('error') or '—'} |"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(md), encoding="utf-8")


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="outputs", help="包含若干 <run>/train.log 的根目录")
    ap.add_argument("--out_csv", default="outputs/per_class_summary.csv")
    ap.add_argument("--out_md", default="outputs/per_class_summary.md")
    ap.add_argument("--inspect", default=None, help="单 log 探查：把含关键字的行打印出来")
    ap.add_argument("--single", default=None, help="单 log 提取：打印 JSON 结果")
    ap.add_argument("--test_only", action="store_true", help="只取附近含 'test' 标记的块（避免选到 dev）")
    args = ap.parse_args()

    if args.inspect:
        inspect_log(args.inspect)
        return

    if args.single:
        r = extract_from_log(args.single, test_only=args.test_only)
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
        return

    logs = collect_logs(args.logs_root)
    if not logs:
        print(f"[WARN] no train.log found under {args.logs_root}/*/train.log", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] found {len(logs)} train.log files under {args.logs_root}/")
    rows = []
    for log in logs:
        run_name = log.parent.name
        r = extract_from_log(log, test_only=args.test_only)
        r["run"] = run_name
        r["log_path"] = str(log)
        rows.append(r)

    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    print(f"[INFO] wrote {args.out_csv}")
    print(f"[INFO] wrote {args.out_md}")

    # 终端摘要
    fmt_counts = {}
    err_runs = []
    for r in rows:
        fmt_counts[r.get("format")] = fmt_counts.get(r.get("format"), 0) + 1
        if r.get("error"):
            err_runs.append(r)
    print(f"\n[SUMMARY] format detection: {fmt_counts}")
    if err_runs:
        print(f"[SUMMARY] {len(err_runs)} run(s) failed to parse:")
        for r in err_runs[:10]:
            print(f"  - {r['run']}: {r.get('error')} (fmt={r.get('format')}, n_blocks={r.get('n_blocks')})")
    ok_rows = [r for r in rows if not r.get("error")]
    if ok_rows:
        ok_rows.sort(key=lambda x: (x.get("overall_f1") or 0), reverse=True)
        print(f"\n[TOP 5 by overall F1]")
        for r in ok_rows[:5]:
            print(
                f"  {r['run']:50s}  ep={r.get('best_epoch')}  "
                f"overall={fmt_v(r.get('overall_f1'))}  MISC={fmt_v(r.get('MISC_f1'))}"
            )


if __name__ == "__main__":
    main()
