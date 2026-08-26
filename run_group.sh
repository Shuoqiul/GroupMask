#!/usr/bin/env bash
set -o pipefail

LOG_DIR="${LOG_DIR:-./traininglogs}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-recipe_train_groupsparsity.py}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-14B}"
GROUPMASK_OUTPUT_ROOT="${GROUPMASK_OUTPUT_ROOT:-./outputs}"

export GROUPMASK_OUTPUT_ROOT

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_group256_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

send_notification() {
  local title="$1"
  local message="$2"
  if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
  fi
}

torchrun \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --master_port="${MASTER_PORT:-29503}" \
  "$TRAIN_SCRIPT" \
  --exp_name="groupsparsity" \
  --batch_size=1 \
  --hf_model="$HF_MODEL" \
  --compile_flag=false \
  --use_fsdp=true \
  --use_bf16=true \
  --use_minipile=false \
  --dataset_seed=42 \
  --dataset_list=['wiki'] \
  --total_n_step=30000 \
  --save_interval=10000 \
  --groups_in_dim=1 \
  --groups_out_dim=256 \
  --hn_groups=1 \
  --gamma=0.01 \
  --kd_loss=true \
  --mix_loss=false \
  --scale_weight=false \
  --soft_rank=false \
  --use_reinmax=false \
  --hard_flag=false \
  --semi_params=false \
  --use_ddp=false \
  --hn_block_size=2048 \
  --simple_gate=false \
  --p=0.5 \
  --T=0.4 \
  --lam=16 \
  --share_qk=false \
  --hidden_kd=false \
  --adam_8bit=true \
  2>&1 | tee "$LOG_FILE"

status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then
  send_notification "Training Success" "job finished"
else
  send_notification "Training Failed" "job crashed"
fi

exit "$status"
