#!/usr/bin/env python3
"""从各 output/*/eval_results.txt 抽取 test best ckpt 的 per_class_f1，写入 outputs/per_class_summary.txt"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SUM = os.path.join(ROOT, "outputs", "per_class_summary.txt")

JOBS = [
    ("Twitter-2017", "exp1_baseline"),
    ("Twitter-2017", "exp2_pp_ib"),
    ("Twitter-2017", "exp3_got_cac"),
    ("Twitter-2017", "grid_6"),
    ("Twitter-2015", "t15_exp1_baseline"),
    ("Twitter-2015", "t15_exp2_pp_ib"),
    ("Twitter-2015", "t15_exp3_got_cac"),
    ("Twitter-2015", "t15_exp4_both"),
]

ORDER_17 = ["PER", "LOC", "ORG", "MISC"]


def parse_block(text: str):
    """取最后一次 test best ckpt 块中的 overall span_f1 与 per_class_f1。"""
    parts = text.split("***** Eval results test best ckpt *****")
    if len(parts) < 2:
        return "", {}
    tail = parts[-1]
    om = re.search(r"span_f1 = ([\d.]+)", tail)
    overall = om.group(1) if om else ""
    f1s = {}
    pcm = re.search(r"per_class_f1:\n((?: .*\n)+)", tail)
    if pcm:
        for line in pcm.group(1).splitlines():
            mm = re.match(
                r"\s*(\w+):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+support=(\d+)",
                line,
            )
            if mm:
                f1s[mm.group(1)] = mm.group(4)
    return overall, f1s


def main():
    lines = [
        "# 数据集 | 组别 | PER-F1 | LOC-F1 | ORG-F1 | MISC/OTHER-F1 | Overall-F1",
        "# 2015 第四列为 OTHER（表头仍写 MISC/OTHER）",
    ]
    for ds, tag in JOBS:
        path = os.path.join(ROOT, "outputs", tag, "eval_results.txt")
        if not os.path.isfile(path):
            lines.append(f"{ds}\t{tag}\t(missing eval_results.txt)")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        overall, f1s = parse_block(text)
        if "Twitter-2015" in ds:
            keys = ["PER", "LOC", "ORG", "OTHER"]
        else:
            keys = ORDER_17
        cols = [f1s.get(k, "NA") for k in keys]
        lines.append("\t".join([ds, tag] + cols + [overall or "NA"]))
    os.makedirs(os.path.dirname(OUT_SUM), exist_ok=True)
    with open(OUT_SUM, "w", encoding="utf-8") as w:
        w.write("\n".join(lines) + "\n")
    print("Wrote", OUT_SUM)


if __name__ == "__main__":
    main()
