#!/usr/bin/env bash
set -o pipefail

EVAL_SCRIPT="${EVAL_SCRIPT:-hf_ppl.py}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-14B}"
BASELINE_MODEL="${BASELINE_MODEL:-meta-llama/Llama-2-13b-hf}"
HN_PATH="${HN_PATH:-./checkpoints/hn-ckpt-iter-030000.pt}"
SAVE_HF_DIR="${SAVE_HF_DIR:-./outputs/qwen3-14b-wiki-1-256}"

send_notification() {
  local title="$1"
  local message="$2"
  if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
  fi
}

torchrun \
  --master_port="${MASTER_PORT:-29503}" \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "$EVAL_SCRIPT" \
  --hf_model "$HF_MODEL" \
  --dataset "wikitext" \
  --semi_evaluate true \
  --eval_baseline false \
  --baseline_model_name "$BASELINE_MODEL" \
  --hn_path "$HN_PATH" \
  --hn_groups=1 \
  --block_size=1024 \
  --batch_limit=2000 \
  --groups_in_dim=1 \
  --groups_out_dim=256 \
  --simple_gate=false \
  --gate_mode="hard" \
  --save_hf_dir "$SAVE_HF_DIR" \
  --share_qk=false \
  --is_simple_gate=false

status=$?
if [ "$status" -eq 0 ]; then
  send_notification "Evaluation Success" "job finished"
else
  send_notification "Evaluation Failed" "job crashed"
fi

exit "$status"
