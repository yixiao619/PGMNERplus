#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选：调用多模态 LLM 为每条样本生成简短视觉描述，写入 visual_suffix_jsonl，
供 main-mner-robert.py --visual_suffix_jsonl 使用。

需要: pip install openai
环境变量: OPENAI_API_KEY

数据：MINER 使用 CoNLL txt（非 train.json）。本脚本从 data/twitter2017/{split}.txt 读入。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

def read_conll_samples(txt_path: Path):
    """Yield dict tokens, labels, imgid."""
    words, labels, imgid = [], [], None
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if line == "":
                if words and imgid is not None:
                    yield {"tokens": list(words), "labels": list(labels), "imgid": str(imgid).strip()}
                words, labels = [], []
                imgid = None
                continue
            if "IMGID" in line:
                imgid = line.split(":")[-1].strip()
                continue
            sp = line.split("\t")
            if len(sp) not in (2, 4):
                continue
            words.append(sp[0])
            labels.append(sp[-1].replace("\n", ""))
    if words and imgid is not None:
        yield {"tokens": list(words), "labels": list(labels), "imgid": str(imgid).strip()}


def find_image(img_root: Path, img_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".gif", ""):
        p = img_root / (img_id + ext if ext else img_id)
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", help="train|dev|test")
    ap.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data/twitter2017"))
    ap.add_argument("--image_dir", type=str, default=str(REPO_ROOT / "data/twitter2017_images"))
    ap.add_argument("--out_jsonl", type=str, default=str(REPO_ROOT / "data/twitter2017/visual_suffix_train.jsonl"))
    ap.add_argument("--max_samples", type=int, default=0, help="0=all")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; exiting.", file=sys.stderr)
        sys.exit(2)

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    data_dir = Path(args.data_dir)
    img_root = Path(args.image_dir)
    txt = data_dir / f"{args.split}.txt"
    if not txt.is_file():
        print(f"Missing {txt}", file=sys.stderr)
        sys.exit(1)

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(outp, "w", encoding="utf-8") as outf:
        for sample in read_conll_samples(txt):
            if args.max_samples and n >= args.max_samples:
                break
            img_path = find_image(img_root, sample["imgid"])
            if not img_path:
                continue
            b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            text = " ".join(sample["tokens"])
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f'Briefly describe the image in one English sentence for NER context. '
                                        f'Tweet: "{text[:500]}"\n'
                                        f'Reply with JSON: {{"visual_context": "..."}}'
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )
                raw = r.choices[0].message.content or ""
                ctx = raw
                try:
                    lo = raw.find("{")
                    hi = raw.rfind("}") + 1
                    if lo >= 0 and hi > lo:
                        js = json.loads(raw[lo:hi])
                        ctx = js.get("visual_context", raw)
                except json.JSONDecodeError:
                    pass
                suffix = f"[Visual: {ctx.strip()}]"
                rec = {"img_id": sample["imgid"], "suffix": suffix}
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            except Exception as e:
                print(f"[WARN] {sample['imgid']}: {e}", file=sys.stderr)
    print(f"Wrote {n} lines to {outp}")


if __name__ == "__main__":
    main()
