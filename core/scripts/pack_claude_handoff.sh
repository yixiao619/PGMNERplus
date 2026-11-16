#!/usr/bin/env bash
# 打包「给 Claude 网页版 / 其它 LLM 读上下文」用的小体积归档（不含 outputs、pretrain、*.pth）。
# 用法：在 MINER-My 根目录执行   bash scripts/pack_claude_handoff.sh
# 生成：上级目录 MINER-My-claude-handoff-YYYYMMDD-HHMMSS.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
# 归档放在仓库上一级，避免打进自己
ARCHIVE="$(dirname "$ROOT")/MINER-My-claude-handoff-${STAMP}.tar.gz"

# 可选：yixiao 下的实验笔记（与 finalreport 互链）
YIXIAO_ROOT="$(dirname "$ROOT")"
VLOG_GLOB=( "$YIXIAO_ROOT"/vlog/*MINER*.md )

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STAGE="$TMP/stage"
mkdir -p "$STAGE/MINER-My-context/"{snapshots,vlog}

# --- 核心文档与代码 ---
for f in finalreport.md EXPERIMENT_LOG.md eval_metric.py engine_utils.py OT.py \
         OT_baseline_backup.py main-mner-robert.py main-mner-robert_baseline_backup.py \
         loader4image.py graph_got_toy.py readme.txt CLAUDE_WEB_upload_guide.md; do
  [[ -f "$f" ]] && cp -a "$f" "$STAGE/MINER-My-context/"
done

for d in models utils configs scripts analysis; do
  if [[ -d "$d" ]]; then
    mkdir -p "$STAGE/MINER-My-context/$d"
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' --exclude='logs/' \
      "$d/" "$STAGE/MINER-My-context/$d/"
  fi
done

# analysis/results 里小 json（仅顶层，避免误拷大文件）
if [[ -d analysis/results ]]; then
  mkdir -p "$STAGE/MINER-My-context/analysis/results"
  find analysis/results -maxdepth 1 -name '*.json' -exec cp -a {} "$STAGE/MINER-My-context/analysis/results/" \; 2>/dev/null || true
fi

# --- 实验数字快照（若存在；软链用 -L 跟到真文件）---
snap() {
  local src="$1" dest="$2"
  if [[ -e "$src" ]] || [[ -L "$src" ]]; then
    mkdir -p "$(dirname "$STAGE/MINER-My-context/$dest")"
    cp -L -f "$src" "$STAGE/MINER-My-context/$dest" 2>/dev/null || cp -f "$src" "$STAGE/MINER-My-context/$dest"
  fi
}

snap "outputs/grid_search/results.csv" "snapshots/grid_search_results.csv"
snap "outputs/final_report.md" "snapshots/final_report.md"
snap "outputs/all_results.csv" "snapshots/all_results.csv"
snap "outputs/per_class_summary.txt" "snapshots/per_class_summary.txt"
snap "outputs/sensitivity_summary.txt" "snapshots/sensitivity_summary.txt"
snap "scripts/EXPERIMENTS.txt" "snapshots/EXPERIMENTS.txt"

# vlog 里 MINER 相关 md
shopt -s nullglob
for f in "${VLOG_GLOB[@]}"; do
  cp -f "$f" "$STAGE/MINER-My-context/vlog/" || true
done
shopt -u nullglob

# 上传说明放进包内
cat >"$STAGE/MINER-My-context/CLAUDE_WEB_UPLOAD_GUIDE.txt" <<'EOF'
1) 在 Claude 网页版：可直接上传本 .tar.gz（解压后）或把 finalreport.md 全文粘贴进对话。
2) 需要它改代码时：可再说明「仓库根目录即 MINER-My-context」，并点名 models/bn_bert_ner_OT_robert.py、main-mner-robert.py。
3) 数字以 finalreport.md + snapshots/ 下 csv、md 为准；不必传 best_model.pth。
4) 单文件长度限制时：优先顺序 finalreport.md → snapshots/grid_search_results.csv → vlog/*.md → EXPERIMENT_LOG.md。
EOF

tar czvf "$ARCHIVE" -C "$STAGE" MINER-My-context

echo ""
echo "== OK: $ARCHIVE =="
ls -lh "$ARCHIVE"
echo "解压: tar xzvf $(basename "$ARCHIVE")"
