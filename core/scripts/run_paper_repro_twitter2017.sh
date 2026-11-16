#!/usr/bin/env bash
# ICME 匿名稿「Implementation Configurations」对齐的 Twitter-2017 主方法复现线。
# 文本: RoBERTa-base；视觉: CLIP ViT-B/32（与常见 ViT-base 同维 768，作论文 ViT-base 的本地实现）
# 维度: txt/hidden 全 768
# 论文超参: β=0.1, λ=0.5, p=0.3 → 代码映射见下方与 vlog 说明
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

OUT="${1:-outputs/paper_repro_2017_base}"

# --- 与论文/代码的对应关系（需写进实验笔记）---
# β=0.1  →  --ib_kl_weight 0.1  （MEC 中 VIB/ KL 项权重；旧代码硬编码 0.001）
# p=0.3  →  --ot_prompt_ratio 0.3  （OT.py 中 filter bucket 分位，默认即 0.3）
# λ=0.5  →  --ot_fusion_temperature 0.5  （CAC 里对 transport_plan 做 softmax 的温度；启发式对应论文的 λ，若与主表差较大可再试 0.25/0.1 等）
# 未开 --use_pp_ib / --use_got_cac：与你们「全模块默认前向」一致（MEC 的 dist+KL + CAC 的 OT 融合 + MTC 的 SCL）

mkdir -p "$OUT"
python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --output_dir "$OUT" \
  --model_name_or_path pretrain/roberta-base \
  --tokenizer_name pretrain/roberta-base \
  --vit_name pretrain/clip-vit-base-patch16 \
  --gpu_id 0 \
  --epoch 30 \
  --batch_size 8 \
  --max_seq_len 128 \
  --max_span_len 4 \
  --lr 3e-5 \
  --bert_lr 3e-5 \
  --patience 0 \
  --seed 42 \
  --logging_steps 100 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --txt_global_dim 768 \
  --txt_local_dim 768 \
  --img_global_dim 768 \
  --img_local_dim 768 \
  --hidden_dim 768 \
  --ib_kl_weight 0.1 \
  --ot_prompt_ratio 0.3 \
  --ot_fusion_temperature 0.5 \
  2>&1 | tee "$OUT/train.log"

echo "== Paper-align repro done. See $OUT/eval_results.txt and train.log =="
