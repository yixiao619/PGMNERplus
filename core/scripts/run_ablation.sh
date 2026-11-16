#!/usr/bin/env bash
# Usage: bash scripts/run_ablation.sh <GPU> <DATASET> <MATRIX>
# Example: bash scripts/run_ablation.sh 0 t17 scripts/ablation_matrix.tsv
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/baseline_hparams.sh"

GPU="${1:?GPU id}"
DATASET="${2:?t17|t15}"
MATRIX="${3:-scripts/ablation_matrix.tsv}"

BASELINE_F1=0.9064
SEEDS=(123 456 789 1024 2048)
SUMMARY="outputs/driver_${DATASET}.summary"
BORDERLINE="outputs/driver_${DATASET}.borderline"
: >"$SUMMARY"
: >"$BORDERLINE"

if [[ "$DATASET" == "t17" ]]; then
  DATA_DIR="data/twitter2017"
elif [[ "$DATASET" == "t15" ]]; then
  DATA_DIR="data/twitter2015"
else
  echo "Unknown DATASET=$DATASET" >&2
  exit 1
fi

build_run_args() {
  local key="$1"
  local extra="$2"
  RUN_ARGS=()
  local -a src=("${BASELINE_ARGS[@]}")
  local i=0
  local flag val

  should_drop() {
    local f="$1"
    case "$key" in
      noppib)
        case "$f" in
          --use_pp_ib|--beta_polar|--delta_minus|--delta_plus) return 0 ;;
        esac ;;
      nogota)
        case "$f" in
          --use_got|--got_mode|--ot_alpha|--entropy_eps|--ot_sinkhorn_iter|--partial_mass_init|--dust_cost|--ot_warmup_epochs) return 0 ;;
        esac ;;
      noema)
        case "$f" in
          --use_ema|--ema_decay) return 0 ;;
        esac ;;
    esac
    return 1
  }

  is_bool_flag() {
    case "$1" in
      --use_pp_ib|--use_got|--use_ema) return 0 ;;
    esac
    return 1
  }

  apply_extra_args() {
    local -a extra_arr=( $extra )
    local j=0 ef ev k
    while (( j < ${#extra_arr[@]} )); do
      ef="${extra_arr[j]}"
      [[ "$ef" != --* ]] && { ((j += 1)); continue; }
      if (( j + 1 < ${#extra_arr[@]} )) && [[ "${extra_arr[j + 1]}" != --* ]]; then
        ev="${extra_arr[j + 1]}"
        ((j += 2))
      else
        ev=""
        ((j += 1))
      fi
      local found=-1
      for ((k = 0; k < ${#RUN_ARGS[@]}; k++)); do
        if [[ "${RUN_ARGS[k]}" == "$ef" ]]; then
          found=$k
          break
        fi
      done
      if (( found >= 0 )); then
        if [[ -n "$ev" ]]; then
          RUN_ARGS[found + 1]="$ev"
        fi
      elif [[ -n "$ev" ]]; then
        RUN_ARGS+=("$ef" "$ev")
      else
        RUN_ARGS+=("$ef")
      fi
    done
  }

  while (( i < ${#src[@]} )); do
    flag="${src[$i]}"
    if should_drop "$flag"; then
      if is_bool_flag "$flag"; then
        ((i += 1))
      else
        ((i += 2))
      fi
      continue
    fi
    RUN_ARGS+=("$flag")
    if is_bool_flag "$flag"; then
      ((i += 1))
    else
      RUN_ARGS+=("${src[$i + 1]}")
      ((i += 2))
    fi
  done

  if [[ -n "$extra" && "$extra" != "REMOVE_PPIB" && "$extra" != "REMOVE_GOT" && "$extra" != "REMOVE_EMA" ]]; then
    apply_extra_args
  fi
}

mark_skipped_seeds() {
  local key="$1"
  local from_seed="$2"
  local started=0
  for s in "${SEEDS[@]}"; do
    if [[ "$started" == "1" ]]; then
      local o="outputs/${DATASET}_abl_${key}_seed${s}_v2"
      mkdir -p "$o"
      echo "gated after seed ${from_seed} (F1 < ${BASELINE_F1})" >"$o/SKIPPED_GATED.txt"
    fi
    [[ "$s" == "$from_seed" ]] && started=1
  done
}

read_test_f1() {
  local out="$1"
  python3 - <<PY
import json, sys
out = "$out"
try:
    rows = [json.loads(l) for l in open(f"{out}/per_class_history.jsonl")]
    test = [r for r in rows if r.get("split") == "test"]
    if test:
        print(f"{max(float(r['overall']['f1']) for r in test):.4f}")
    else:
        print("0.0000")
except Exception:
    print("0.0000")
PY
}

echo "[$(date -Is)] driver start DATASET=$DATASET GPU=$GPU" | tee -a "$SUMMARY"

while IFS=$'\t' read -r key extra; do
  key="${key//$'\r'/}"
  extra="${extra//$'\r'/}"
  [[ -z "${key:-}" || "$key" =~ ^# || "$key" == "key" ]] && continue
  echo "===== [$DATASET] $key =====" | tee -a "$SUMMARY"
  gated=0
  for seed in "${SEEDS[@]}"; do
    out="outputs/${DATASET}_abl_${key}_seed${seed}_v2"
    mkdir -p "$out"
    build_run_args "$key" "${extra:-}"
    echo "[$(date +%H:%M:%S)] launching $key seed=$seed -> $out" | tee -a "$SUMMARY"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python3 main-mner-robert.py \
      "${RUN_ARGS[@]}" \
      --data_dir "$DATA_DIR" \
      --seed "$seed" \
      --epoch 30 \
      --patience 5 \
      --output_dir "$out" \
      >>"$out/train.log" 2>&1
    rc=$?
    f1="$(read_test_f1 "$out")"
    echo "[$key seed=$seed] rc=$rc test_F1=$f1" | tee -a "$SUMMARY"
    bash "$ROOT/scripts/package_ablation_run.sh" "$out" 2>/dev/null || true
    if [[ "$rc" != "0" ]]; then
      echo "[$key seed=$seed] ERROR rc=$rc (training failed, not gating)" | tee -a "$SUMMARY"
      exit 1
    fi
    below="$(python3 -c "print(1 if float('$f1') < $BASELINE_F1 else 0)")"
    if [[ "$below" == "1" ]]; then
      bord="$(python3 -c "print(1 if 0.9000 <= float('$f1') < $BASELINE_F1 else 0)")"
      if [[ "$bord" == "1" ]]; then
        echo "[$key seed=$seed] F1=$f1 borderline" | tee -a "$BORDERLINE"
      fi
      echo "[$key] gated at seed=$seed (F1=$f1 < $BASELINE_F1)" | tee -a "$SUMMARY"
      mark_skipped_seeds "$key" "$seed"
      gated=1
      break
    fi
  done
  if [[ "$gated" == "0" ]]; then
    echo "[$key] completed all seeds" | tee -a "$SUMMARY"
  fi
done <"$MATRIX"

echo "[$(date -Is)] ALL DONE: $DATASET" | tee -a "$SUMMARY"
