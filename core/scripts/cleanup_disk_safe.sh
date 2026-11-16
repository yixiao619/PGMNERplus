#!/usr/bin/env bash
# Free disk while KEEPING: main checkpoints, ensemble, ALL ablation outputs, driver logs, artifacts/.
set -eu
LOG="/home/jyx/yixiao/MINER-My/logs/disk_cleanup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== disk cleanup $(date -Is) ==="
df -h /

# Exact output dir names always kept
KEEP_OUTPUTS_EXACT=(
  pp_ib_partial_ot_grid_dc10_m05_seed42
  t17_fig3_mass0.7_dc08_seed42
  ensemble
)

should_keep_output() {
  local base="$1"
  local k
  for k in "${KEEP_OUTPUTS_EXACT[@]}"; do
    [[ "$base" == "$k" ]] && return 0
  done
  case "$base" in
    t17_abl_*|t15_abl_*)
      return 0 ;;
    t17_fig3_*|t15_fig3_*)
      return 0 ;;
    driver_t17.*|driver_t15.*)
      return 0 ;;
    t17_abl_3seed_summary.*|t17_ablation_table.tex|t17_abl_max_summary.tsv)
      return 0 ;;
    exp[1-4]_*|exp_abl_*|got_ablation*|got_k*|got_thr*)
      return 0 ;;
    paper_repro_*)
      return 0 ;;
  esac
  return 1
}

rm_rf() {
  local p="$1"
  [[ -e "$p" ]] || return 0
  local sz
  sz=$(du -sh "$p" 2>/dev/null | cut -f1 || echo "?")
  echo "DELETE $sz $p"
  rm -rf "$p"
}

# --- /data/jyx/MINER-My/outputs: keep whitelist + ablation patterns ---
OUT_ROOT="/data/jyx/MINER-My/outputs"
if [[ -d "$OUT_ROOT" ]]; then
  for ent in "$OUT_ROOT"/*; do
    [[ -e "$ent" ]] || continue
    base=$(basename "$ent")
    if should_keep_output "$base"; then
      echo "KEEP outputs/$base"
      continue
    fi
    rm_rf "$ent"
  done
fi

# --- workspace: never delete ablation artifacts or vllm cache ---
echo "KEEP artifacts/ablation_twitter2017 (never delete)"
echo "KEEP data/twitter2017/vllm_desc"
echo "KEEP runs/pgmner_vllm"

# --- workspace results/ (fig3 sweep duplicates only) ---
rm_rf "/home/jyx/yixiao/MINER-My/results/runs"
rm_rf "/home/jyx/yixiao/MINER-My/results/fig3_sweep"
rm_rf "/home/jyx/yixiao/MINER-My/results/pp_ib_fix_stepA"
rm_rf "/home/jyx/yixiao/MINER-My/results/pp_ib_fix_stepA_v2"

# --- stub vllm desc backups ---
rm_rf "/home/jyx/yixiao/MINER-My/data/twitter2017/vllm_desc.stub_backup"
rm_rf /home/jyx/yixiao/MINER-My/data/twitter2017/vllm_desc_stub_*

# --- logs: keep ablation / train / cleanup related ---
LOG_DIR="/home/jyx/yixiao/MINER-My/logs"
for f in "$LOG_DIR"/*; do
  [[ -f "$f" ]] || continue
  base=$(basename "$f")
  case "$base" in
    disk_cleanup_*|pgmner_vllm_train_nohup.log|qwen35_desc_*nohup.log)
      echo "KEEP log $base"
      ;;
    t17_extra_*|t15_extra_*|driver_*|ablation_*|got_abl_*|exp_abl_*|nohup_t17*|nohup_t15*)
      echo "KEEP log $base"
      ;;
    *)
      rm_rf "$f"
      ;;
  esac
done

# --- checkpoints on /data: keep ablation_chain + any t17_abl mirror ---
CKPT="/data/jyx/MINER-My/checkpoints"
if [[ -d "$CKPT" ]]; then
  for ent in "$CKPT"/*; do
    [[ -e "$ent" ]] || continue
    base=$(basename "$ent")
    case "$base" in
      ablation_chain|t17_abl_*|t15_abl_*)
        echo "KEEP checkpoints/$base"
        ;;
      *)
        rm_rf "$ent"
        ;;
    esac
  done
fi

# --- misc under home/jyx (not LargeModels) ---
rm_rf "/home/jyx/yixiao/homework4.tar.gz"
rm_rf "/home/jyx/yixiao/MINER-My-claude-handoff"*.tar.gz
rm_rf "/home/jyx/yixiao/tmp"

echo "=== after cleanup ==="
df -h /
du -sh /data/jyx/MINER-My/outputs/* /home/jyx/yixiao/MINER-My/artifacts/ablation_twitter2017 \
  /home/jyx/yixiao/MINER-My/runs/pgmner_vllm /home/jyx/yixiao/MINER-My/data/twitter2017/vllm_desc 2>/dev/null || true
