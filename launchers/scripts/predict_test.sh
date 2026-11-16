#!/usr/bin/env bash
# 仅 test 推理（写 test_predictions.txt）
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/hparams.sh"

OUT="${1:?output dir}"
GPU="${GPU:-0}"

SPLIT=test bash "$(dirname "$0")/eval_checkpoint.sh" "$OUT"
