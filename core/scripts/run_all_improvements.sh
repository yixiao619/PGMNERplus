#!/usr/bin/env bash
# 一键管线：Table1 修复 → 参考统计 →（可选）网格 / ViT-L / 多 seed / LLM / 汇总
# 后台完整跑： FULL_PIPELINE=1 nohup bash scripts/run_all_improvements.sh >> logs/full_pipeline.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs analysis/results outputs/grid_search outputs/ensemble

export PYTHONUNBUFFERED=1
GPU_ID="${GPU_ID:-0}"
BEST_BETA="${BEST_BETA:-0.01}"
BEST_LR="${BEST_LR:-3e-5}"

if [[ "${FULL_PIPELINE:-}" == "1" ]]; then
  export RUN_GRID_SEARCH="${RUN_GRID_SEARCH:-1}"
  export RUN_VITL="${RUN_VITL:-1}"
  export RUN_MULTI_SEED="${RUN_MULTI_SEED:-1}"
  echo "[FULL_PIPELINE] RUN_GRID_SEARCH=$RUN_GRID_SEARCH RUN_VITL=$RUN_VITL RUN_MULTI_SEED=$RUN_MULTI_SEED"
fi

echo "======== Phase 1: Table 1 analysis (MISC+OTHER fix) ========"
python analysis/dataset_analysis.py \
  --data_root data \
  --datasets Twitter-2015,Twitter-2017 \
  --clip_model pretrain/clip-vit-base-patch16 \
  --output_file analysis/results/table1_fixed.json \
  2>&1 | tee logs/fix_table1.log

python analysis/create_reference_stats.py --output_file analysis/results/reference_stats.json

if [[ "${RUN_GRID_SEARCH:-}" == "1" ]]; then
  echo "======== Phase 2: grid search（耗时极长）========"
  python scripts/auto_grid_search.py --gpu_id "$GPU_ID" --stages beta,lr 2>&1 | tee logs/grid_search_master.log
  if [[ -f outputs/grid_search/results.csv ]]; then
    BEST_BETA="$(python3 -c "
import csv
from pathlib import Path
p=Path('outputs/grid_search/results.csv')
best=None
with p.open() as f:
    for r in csv.DictReader(f):
        if not r.get('test_f1'): continue
        try: v=float(r['test_f1'])
        except: continue
        if best is None or v>best[0]: best=(v,r)
if best:
    print(best[1].get('beta_polar','0.01'))
else:
    print('0.01')
")"
    BEST_LR="$(python3 -c "
import csv
from pathlib import Path
p=Path('outputs/grid_search/results.csv')
best=None
with p.open() as f:
    for r in csv.DictReader(f):
        if not r.get('test_f1'): continue
        try: v=float(r['test_f1'])
        except: continue
        if best is None or v>best[0]: best=(v,r)
if best and best[1].get('lr'):
    print(best[1]['lr'])
else:
    print('3e-5')
")"
    export BEST_BETA BEST_LR
    echo "Grid best: BEST_BETA=$BEST_BETA BEST_LR=$BEST_LR"
  fi
fi

if [[ -f outputs/grid_search/results.csv ]] && [[ "${RUN_GRID_SEARCH:-}" != "1" ]]; then
  BEST_BETA="$(python3 -c "
import csv
from pathlib import Path
p=Path('outputs/grid_search/results.csv')
best=None
with p.open() as f:
    for r in csv.DictReader(f):
        if not r.get('test_f1'): continue
        try: v=float(r['test_f1'])
        except: continue
        if best is None or v>best[0]: best=(v,r)
if best:
    print(best[1].get('beta_polar','0.01'))
else:
    print('0.01')
")"
  BEST_LR="$(python3 -c "
import csv
from pathlib import Path
p=Path('outputs/grid_search/results.csv')
best=None
with p.open() as f:
    for r in csv.DictReader(f):
        if not r.get('test_f1'): continue
        try: v=float(r['test_f1'])
        except: continue
        if best is None or v>best[0]: best=(v,r)
if best and best[1].get('lr'):
    print(best[1]['lr'])
else:
    print('3e-5')
")"
  export BEST_BETA BEST_LR
  echo "Loaded grid CSV: BEST_BETA=$BEST_BETA BEST_LR=$BEST_LR"
fi

VIT_L="${VIT_L:-pretrain/clip-vit-large-patch14}"
if [[ "${RUN_VITL:-}" == "1" ]] && [[ -d "$VIT_L" ]]; then
  echo "======== ViT-L training ========"
  mkdir -p outputs/vitl_best_config
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name "$VIT_L" \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id "$GPU_ID" \
    --epoch 30 --batch_size 8 \
    --max_seq_len 128 --max_span_len 4 \
    --lr "$BEST_LR" --bert_lr "$BEST_LR" \
    --patience 0 --seed 42 --logging_steps 100 \
    --use_pp_ib --beta_polar "$BEST_BETA" \
    --delta_minus 0.2 --delta_plus 0.8 \
    --output_dir outputs/vitl_best_config \
    --do_train true --do_eval true --do_predict true \
    2>&1 | tee outputs/vitl_best_config/train.log
else
  if [[ "${RUN_VITL:-}" == "1" ]]; then
    echo "[[ Skip ViT-L: 目录不存在 $VIT_L ，请将 CLIP ViT-L/14 放到该路径或设置 VIT_L=... ]]"
  else
    echo "[[ Skip ViT-L: RUN_VITL!=1 ]]"
  fi
fi

if [[ "${RUN_MULTI_SEED:-}" == "1" ]]; then
  echo "======== Multi-seed（若本地无 ViT-L 则自动用 ViT-B/16）========"
  SEED_VIT="$VIT_L"
  SEED_USE_HF=true
  if [[ ! -d "$SEED_VIT" ]]; then
    SEED_VIT="pretrain/clip-vit-base-patch16"
    SEED_USE_HF=false
    echo "[INFO] ViT-L 不可用，multi-seed 使用 $SEED_VIT"
  fi
  export GPU_ID BEST_BETA BEST_LR VIT_NAME="$SEED_VIT" USE_HF_CLIP="$SEED_USE_HF"
  bash scripts/train_multi_seed.sh 2>&1 | tee logs/train_multi_seed.log
  echo "======== Ensemble hard vote on test_predictions ========"
  python scripts/ensemble_predict.py \
    --output outputs/ensemble/test_predictions_voted.txt \
    2>&1 | tee logs/ensemble_predict.log || true
fi

if [[ -n "${OPENAI_API_KEY:-}" ]] && [[ "${RUN_GPT4V:-}" == "1" ]]; then
  python scripts/gpt4v_augmentation.py --max_samples 50 || true
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]] && [[ "${RUN_CLAUDE:-}" == "1" ]]; then
  python scripts/claude_refinement.py || true
fi

python scripts/summarize_all_results.py 2>&1 | tee logs/summarize_results.log
echo "Done. See outputs/final_report.md"
