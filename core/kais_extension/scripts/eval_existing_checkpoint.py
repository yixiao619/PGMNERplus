#!/usr/bin/env python
"""KAIS Phase A1: evaluate an existing PGMNER checkpoint without training.

This wrapper avoids main-mner-robert.py fast_evaluate(), which writes debug
artifacts under saved_weights/. It imports the original code after preparing
sys.argv, then calls evaluate(..., mode="test") directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--labels-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--model-name-or-path", default="pretrain/roberta-large")
    p.add_argument("--tokenizer-name", default="pretrain/roberta-large")
    p.add_argument("--vit-name", default="pretrain/clip-vit-base-patch16")
    p.add_argument("--gpu-id", default="-1")
    p.add_argument("--seed", default="42")
    return p.parse_args()


def write_result(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_with_legacy_clip_key_remap(model, state: dict):
    """Load current or legacy CLIPVisionModel wrapper key layouts."""
    try:
        return model.load_state_dict(state, strict=True), "strict"
    except RuntimeError:
        mapped = {}
        changed = 0
        old = "image_model._hf.encoder.vision_model."
        new = "image_model._hf.encoder."
        for k, v in state.items():
            if k.startswith(old):
                mapped[new + k[len(old):]] = v
                changed += 1
            else:
                mapped[k] = v
        if changed == 0:
            raise
        return model.load_state_dict(mapped, strict=True), f"legacy_clip_key_remap:{changed}"


def main() -> int:
    a = parse_args()
    repo = Path(a.repo_root).resolve()
    os.chdir(repo)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": "Twitter-2017",
        "checkpoint": a.checkpoint,
        "precision": None,
        "recall": None,
        "f1": None,
        "seed": a.seed,
        "status": "failed",
    }

    try:
        sys.path.insert(0, str(repo))
        sys.argv = [
            "main-mner-robert.py",
            "--data_dir",
            a.data_dir,
            "--output_dir",
            a.output_dir,
            "--model_name_or_path",
            a.model_name_or_path,
            "--tokenizer_name",
            a.tokenizer_name,
            "--vit_name",
            a.vit_name,
            "--gpu_id",
            a.gpu_id,
            "--do_train",
            "false",
            "--do_eval",
            "false",
            "--do_predict",
            "false",
            "--batch_size",
            "16",
            "--max_span_len",
            "4",
            "--max_seq_len",
            "128",
            "--seed",
            str(a.seed),
            "--use_pp_ib",
            "--use_got",
            "--got_mode",
            "partial_sinkhorn",
            "--ot_sinkhorn_iter",
            "30",
            "--partial_mass_init",
            "0.5",
            "--dust_cost",
            "1.0",
            "--entropy_eps",
            "0.1",
            "--ot_alpha",
            "0.05",
            "--auto_img_dims",
            "true",
            "--use_hf_clip_vision",
            "true",
        ]
        spec = importlib.util.spec_from_file_location("pgmner_main", repo / "main-mner-robert.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load main-mner-robert.py")
        pgmner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pgmner)

        pgmner.args.labels = a.labels_file
        labels = pgmner.get_labels(pgmner.args.labels)
        config = pgmner.AutoConfig.from_pretrained(
            pgmner.args.config_name if pgmner.args.config_name else pgmner.args.model_name_or_path,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=None,
        )
        tokenizer = pgmner.RobertaTokenizer.from_pretrained(
            pgmner.args.tokenizer_name if pgmner.args.tokenizer_name else pgmner.args.model_name_or_path
        )
        model = pgmner.BertSpanNerBN(config, args=pgmner.args, num_labels=len(labels))
        state = torch.load(a.checkpoint, map_location=pgmner.args.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        _, load_mode = load_with_legacy_clip_key_remap(model, state)
        result["load_mode"] = load_mode
        model.to(pgmner.args.device)
        metrics, _ = pgmner.evaluate(
            pgmner.args,
            model,
            tokenizer,
            labels,
            mode="test",
            prefix="kais_phaseA1_existing_checkpoint",
            only_test=False,
            split_tag="test",
        )
        result.update(
            {
                "precision": metrics.get("span_precision"),
                "recall": metrics.get("span_recall"),
                "f1": metrics.get("span_f1"),
                "status": "success",
            }
        )
        write_result(a.result_json, result)
        return 0
    except Exception as ex:
        result["error_type"] = type(ex).__name__
        result["error"] = str(ex)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
        write_result(a.result_json, result)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
