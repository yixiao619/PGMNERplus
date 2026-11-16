#!/usr/bin/env python3
"""从 eval_results.txt 取最后一次 dev/test best ckpt 的 span_f1；打印: test_f1<TAB>dev_f1"""
import re
import sys

DEV = "***** Eval results dev best ckpt *****"
TST = "***** Eval results test best ckpt *****"


def last_f1(text: str, marker: str) -> str:
    parts = text.split(marker)
    if len(parts) < 2:
        return "NA"
    m = re.search(r"span_f1 = ([\d.]+)", parts[-1])
    return m.group(1) if m else "NA"


def main():
    if len(sys.argv) < 2:
        print("NA\tNA")
        return
    try:
        text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    except OSError:
        print("NA\tNA")
        return
    tf = last_f1(text, TST)
    df = last_f1(text, DEV)
    print(f"{tf}\t{df}")


if __name__ == "__main__":
    main()
