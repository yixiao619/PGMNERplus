#!/usr/bin/env python3
"""汇总 outputs/sens_beta_* 与 outputs/sens_gw_* 的 test/dev best ckpt span_f1 → outputs/sensitivity_summary.txt"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "sensitivity_summary.txt")

PAT = re.compile(
    r"\*\*\*\*\* Eval results (test|dev) best ckpt \*\*\*\*\*\nspan_f1 = ([\d.]+)"
)


def extract_pair(text):
    test_f1 = dev_f1 = ""
    for m in PAT.finditer(text):
        if m.group(1) == "test":
            test_f1 = m.group(2)
        else:
            dev_f1 = m.group(2)
    # last occurrence wins if multiple; scan order: prefer explicit blocks
    blocks = text.split("***** Eval results test best ckpt *****")
    if len(blocks) > 1:
        tail = blocks[-1]
        mm = re.search(r"span_f1 = ([\d.]+)", tail)
        if mm:
            test_f1 = mm.group(1)
    blocks_d = text.split("***** Eval results dev best ckpt *****")
    if len(blocks_d) > 1:
        tail = blocks_d[-1]
        mm = re.search(r"span_f1 = ([\d.]+)", tail)
        if mm:
            dev_f1 = mm.group(1)
    return test_f1, dev_f1


def main():
    lines = []
    betas = []
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "sens_beta_*", "eval_results.txt"))):
        tag = os.path.basename(os.path.dirname(path))
        beta = tag.replace("sens_beta_", "")
        with open(path, encoding="utf-8", errors="replace") as f:
            t, d = extract_pair(f.read())
        betas.append((beta, t, d))
    lines.append("# beta_polar 扫描（目录名 sens_beta_*）| test_f1 | dev_f1")
    for beta, t, d in betas:
        lines.append(f"{beta}\t{t}\t{d}")

    lines.append("")
    lines.append("# lambda_gw 扫描 | test_f1 | dev_f1")
    gws = []
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs", "sens_gw_*", "eval_results.txt"))):
        tag = os.path.basename(os.path.dirname(path))
        raw = tag.replace("sens_gw_", "", 1)
        gw = raw.replace("p", ".", 1) if "p" in raw else raw
        with open(path, encoding="utf-8", errors="replace") as f:
            t, d = extract_pair(f.read())
        gws.append((gw, t, d))
    for gw, t, d in gws:
        lines.append(f"{gw}\t{t}\t{d}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as w:
        w.write("\n".join(lines) + "\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
