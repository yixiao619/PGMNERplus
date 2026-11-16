#!/usr/bin/env bash
# run_towards_91.sh — 冲刺 91+（适配 main-mner-robert.py）
# 用法: cd MINER-My && mkdir -p logs outputs/towards_91 && nohup bash scripts/run_towards_91.sh > logs/towards_91_master.log 2>&1 &
# 环境变量:
#   GPU_ID (默认 0)  VIT_L (默认 openai/clip-vit-large-patch14，需可下载或换本地目录)
#   RUN_EXP_BCDE=true  默认只跑 EXP-A；设为 true 才跑 ViT-L 相关（需本地或网络有 CLIP-L）

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/outputs/towards_91}"
LOG="$ROOT/logs"
mkdir -p "$OUT" "$LOG"
GPU_ID="${GPU_ID:-0}"
VIT_L="${VIT_L:-openai/clip-vit-large-patch14}"
RUN_EXP_BCDE="${RUN_EXP_BCDE:-false}"

echo "========================================================"
echo " PGMNER towards 91+  start: $(date -Is)"
echo " ROOT=$ROOT OUT=$OUT GPU_ID=$GPU_ID"
echo " RUN_EXP_BCDE=$RUN_EXP_BCDE  VIT_L=$VIT_L"
echo "========================================================"

pip install -q torch-ema 2>/dev/null || echo "[WARN] torch-ema pip failed; use_ema 将跳过"

# 公共超参（与仓库惯例一致；可改）
COMMON=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id "$GPU_ID"
  --epoch 30
  --batch_size 8
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 3e-5
  --patience 0
  --seed 42
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_pp_ib
  --beta_polar 0.005
  --max_grad_norm 1.0
)

echo ""
echo "─── EXP-A: 基准 ViT-B/32 + node_sinkhorn + PP-IB ───"
python main-mner-robert.py "${COMMON[@]}" \
  --vit_name pretrain/clip-vit-base-patch16 \
  --use_got --got_mode node_sinkhorn \
  --ot_alpha 0.05 --entropy_eps 0.1 \
  --ot_sinkhorn_iter 30 \
  --output_dir "$OUT/EXPA_baseline_b32" \
  2>&1 | tee "$LOG/EXPA_baseline_b32.log"
echo "[EXP-A DONE] $(date -Is)"

if [[ "$RUN_EXP_BCDE" != "true" ]]; then
  echo "RUN_EXP_BCDE!=true，跳过 ViT-L / 两阶段实验（避免无权重时失败）。"
  echo "准备 ViT-L 权重后执行: RUN_EXP_BCDE=true bash scripts/run_towards_91.sh"
  exit 0
fi

echo ""
echo "─── EXP-B: ViT-L/14 (HF) + PP-IB only ───"
python main-mner-robert.py "${COMMON[@]}" \
  --vit_name "$VIT_L" \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --output_dir "$OUT/EXPB_vitl14_ppib_only" \
  2>&1 | tee "$LOG/EXPB_vitl14_ppib_only.log"
echo "[EXP-B DONE] $(date -Is)"

echo ""
echo "─── EXP-C: ViT-L/14 + node_sinkhorn ───"
python main-mner-robert.py "${COMMON[@]}" \
  --vit_name "$VIT_L" \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --use_got --got_mode node_sinkhorn \
  --ot_alpha 0.01 --entropy_eps 0.1 \
  --ot_sinkhorn_iter 30 \
  --output_dir "$OUT/EXPC_vitl14_node_sinkhorn" \
  2>&1 | tee "$LOG/EXPC_vitl14_node_sinkhorn.log"
echo "[EXP-C DONE] $(date -Is)"

echo ""
echo "─── EXP-D Stage1: ViT-L + PP-IB 8 epoch（无 GOT）───"
python main-mner-robert.py "${COMMON[@]}" \
  --vit_name "$VIT_L" \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --epoch 8 \
  --output_dir "$OUT/EXPD_stage1_ppib" \
  2>&1 | tee "$LOG/EXPD_stage1.log"
echo "[EXP-D Stage1 DONE] $(date -Is)"

ST1_CKPT="$OUT/EXPD_stage1_ppib/best_model.pth"
if [[ ! -f "$ST1_CKPT" ]]; then
  echo "FATAL: missing $ST1_CKPT"
  exit 1
fi

echo ""
echo "─── EXP-D Stage2: resume + partial_sinkhorn + OT warmup ───"
python main-mner-robert.py "${COMMON[@]}" \
  --vit_name "$VIT_L" \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --use_got --got_mode partial_sinkhorn \
  --ot_alpha 0.005 --entropy_eps 0.1 \
  --partial_mass_init 0.6 --dust_cost 0.5 \
  --ot_warmup_epochs 5 --ot_start_epoch 0 \
  --ot_sinkhorn_iter 30 \
  --resume_checkpoint "$ST1_CKPT" \
  --epoch 22 \
  --lr 3e-6 --bert_lr 1e-6 \
  --output_dir "$OUT/EXPD_stage2_partial_ot" \
  2>&1 | tee "$LOG/EXPD_stage2.log"
echo "[EXP-D DONE] $(date -Is)"

echo ""
echo "─── 可选: 多种子 (123, 2024)，按需取消注释 ───"
# for SEED in 123 2024; do
#   python main-mner-robert.py "${COMMON[@]}" \
#     --vit_name "$VIT_L" --use_hf_clip_vision true --auto_img_dims true \
#     --use_got --got_mode partial_sinkhorn \
#     --ot_alpha 0.005 --entropy_eps 0.1 \
#     --ot_warmup_epochs 5 --resume_checkpoint "$ST1_CKPT" \
#     --epoch 22 --lr 3e-6 --bert_lr 1e-6 \
#     --seed "$SEED" \
#     --output_dir "$OUT/EXPE_seed${SEED}" \
#     2>&1 | tee "$LOG/EXPE_seed${SEED}.log"
# done

echo "========================================================"
echo " 全部请求的阶段完成:                  $(date -Is)"
echo "========================================================"
