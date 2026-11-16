# 由调用方先设置 ROOT 为 MINER-My 根目录后: source "$ROOT/scripts/pgmner_job_util.sh"
# 幂等判断：best_model + (results.txt 或 eval 已有 test F1)

extract_tf_df() {
  python3 "$ROOT/scripts/extract_best_ckpt_f1.py" "$1"
}

read_out_f1() {
  local out="$1"
  if [[ -f "$out/eval_results.txt" ]]; then
    extract_tf_df "$out/eval_results.txt"
    return
  fi
  python3 - "$out/results.txt" <<'PY'
import re, sys
try:
    t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
except OSError:
    print("NA\tNA")
    raise SystemExit(0)

def last_span_f1(blob, marker):
    parts = blob.split(marker)
    if len(parts) < 2:
        return "NA"
    m = re.search(r"span_f1 = ([\d.]+)", parts[-1])
    return m.group(1) if m else "NA"

tf = last_span_f1(t, "***** Eval results test best ckpt *****")
df = last_span_f1(t, "***** Eval results dev best ckpt *****")
print(f"{tf}\t{df}")
PY
}

is_job_complete() {
  local out="$1"
  [[ -f "$out/best_model.pth" ]] || return 1
  if [[ -f "$out/results.txt" ]]; then
    return 0
  fi
  [[ -f "$out/eval_results.txt" ]] || return 1
  read -r tf _ <<< "$(extract_tf_df "$out/eval_results.txt")"
  [[ "$tf" != "NA" ]]
}
