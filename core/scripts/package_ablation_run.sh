#!/usr/bin/env bash
# Package one ablation (or any) run dir for reproducibility.
# Usage: bash scripts/package_ablation_run.sh outputs/t17_abl_nogota_seed123_v2 [--with-ckpt]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?run dir e.g. outputs/t17_abl_nogota_seed123_v2}"
WITH_CKPT=0
[[ "${2:-}" == "--with-ckpt" ]] && WITH_CKPT=1

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing run dir: $RUN_DIR" >&2
  exit 1
fi

base=$(basename "$RUN_DIR")
dest="artifacts/ablation_twitter2017/runs/${base}"
mkdir -p "$dest"

copy_if() {
  local f="$1"
  [[ -e "$RUN_DIR/$f" ]] || return 0
  cp -a "$RUN_DIR/$f" "$dest/"
  echo "  + $f"
}

echo "Packaging $RUN_DIR -> $dest"
copy_if train.log
copy_if eval_results.txt
copy_if per_class_history.jsonl
copy_if SKIPPED_GATED.txt
if [[ "$WITH_CKPT" -eq 1 ]]; then
  copy_if best_model.pth
fi

{
  echo "# packaged $(date -Is)"
  echo "source_dir=$RUN_DIR"
  echo "pwd=$ROOT"
  if [[ -f "$RUN_DIR/train.log" ]]; then
    echo "# first train line:"
    head -n 3 "$RUN_DIR/train.log" | sed 's/^/# /'
  fi
} >"$dest/MANIFEST.txt"

if [[ -f "$RUN_DIR/train.log" ]]; then
  grep -m1 'main-mner-robert' "$RUN_DIR/train.log" 2>/dev/null | head -1 >"$dest/RUN_CMD.txt" || true
fi

echo "Done: $dest ($(du -sh "$dest" | cut -f1))"
