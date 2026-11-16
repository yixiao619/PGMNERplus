#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/_smoke_per_class_dc10_m05"
mkdir -p "$OUT"
rm -f "$OUT/per_class_history.jsonl" "$OUT/eval_results.txt"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python3 - <<'PY'
import importlib.util
import os
import sys
import torch
from transformers import AutoConfig, RobertaTokenizer
from utils.datasets import get_labels
from models.bn_bert_ner_OT_robert import BertSpanNerBN

spec = importlib.util.spec_from_file_location("mner", "main-mner-robert.py")
mner = importlib.util.module_from_spec(spec)
sys.modules["mner"] = mner
spec.loader.exec_module(mner)

out = "outputs/_smoke_per_class_dc10_m05"
ckpt = "outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth"

class A:
    pass

a = A()
a.output_dir = out
a.data_dir = "data/twitter2017"
a.model_name_or_path = "pretrain/roberta-large"
a.tokenizer_name = "pretrain/roberta-large"
a.vit_name = "pretrain/clip-vit-base-patch16"
a.use_hf_clip_vision = True
a.auto_img_dims = True
a.gpu_id = 0
a.batch_size = 16
a.max_seq_len = 128
a.max_span_len = 4
a.use_pp_ib = True
a.beta_polar = 0.01
a.delta_minus = 0.2
a.delta_plus = 0.8
a.pp_ib_lambda_balance = 1.0
a.beta_polar_warmup_epochs = 3
a.pp_ib_gate_init_std = 0.01
a.pp_ib_logvar_clamp = 0.0
a.use_got = True
a.got_mode = "partial_sinkhorn"
a.got_ot_backend = "partial_sinkhorn"
a.ot_alpha = 0.05
a.entropy_eps = 0.1
a.ot_eps = 0.1
a.ot_sinkhorn_iter = 30
a.dust_cost = 0.8
a.partial_mass_init = 0.5
a.ot_warmup_epochs = 3
a.use_ema = True
a.ema_decay = 0.999
a.hidden_dim = 1024
a.txt_global_dim = 1024
a.img_global_dim = 768
a.txt_local_dim = 1024
a.img_local_dim = 768
a.span_detection_weight = 0.0
a.span_head_dropout = 0.1
a.crf_misc_bias = 0.0
a.crf_entity_bias = 0.0
a.crf_o_penalty = 0.0
a.ib_consistency_weight = 0.0
a.ib_cons_temperature = 0.5
a.use_misc_weighted_scl = False
a.use_misc_focal_loss = False
a.use_got_cac = False
a.use_semantic_ot = False
a.got_lambda_sem = 0.5
a.got_struct_mode = "full"
a.labels = "data/twitter2017/./labels.txt"
a.visual_suffix_jsonl = ""
a.visual_suffix_map = None
a.eval_batch_size = 8
a.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels = get_labels(a.labels)
tokenizer = RobertaTokenizer.from_pretrained(a.tokenizer_name)
config = AutoConfig.from_pretrained(a.model_name_or_path)
model = BertSpanNerBN(config, args=a, num_labels=len(labels))
model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
model.to(a.device)
mner.evaluate(a, model, tokenizer, labels, mode="test", prefix="smoke", split_tag="test", epoch_num=0)
print("SMOKETEST wrote", out + "/per_class_history.jsonl")
PY
test -s "$OUT/per_class_history.jsonl"
head -1 "$OUT/per_class_history.jsonl"
