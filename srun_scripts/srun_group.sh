#!/usr/bin/env bash
#SBATCH --job-name=calib-sweep
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2          # 4 个 job，最多同时跑 2 个(按你的配额调，不限制就去掉 %2)
#SBATCH --output=calib_sweep_%A_%a.out
#SBATCH --error=calib_sweep_%A_%a.err

set -o pipefail
set -x

# ========================= 环境 =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1

# ================ 配置区（全部绝对路径）=================
BASE=/orange/sgao1/sli/semi-structure-merge/GroupMask
TRAIN_SCRIPT="${BASE}/recipe_train_groupsparsity.py"
export GROUPMASK_OUTPUT_ROOT="${BASE}/outputs"
LOG_DIR="${BASE}/traininglogs"

NPROC_PER_NODE=1
MASTER_PORT=29503
HF_MODEL="meta-llama/Llama-2-7b-hf"

# ---- array 索引 → n_calib_samples 映射 ----
CALIB_LIST=(100 200 400 800)
N_CALIB=${CALIB_LIST[$SLURM_ARRAY_TASK_ID]}
# ========================================================

mkdir -p "$LOG_DIR" "$GROUPMASK_OUTPUT_ROOT"
LOG_FILE="$LOG_DIR/train_group256_calib${N_CALIB}_$(date +%Y%m%d_%H%M%S).log"
echo "Array task $SLURM_ARRAY_TASK_ID | n_calib_samples=$N_CALIB"
echo "Logging to: $LOG_FILE"

send_notification() {
    local title="$1" message="$2"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
    fi
}

torchrun \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$TRAIN_SCRIPT" \
    --exp_name="groupsparsity-calib${N_CALIB}" \
    --batch_size=1 \
    --hf_model="$HF_MODEL" \
    --compile_flag=false \
    --use_fsdp=true \
    --use_bf16=true \
    --use_minipile=false \
    --dataset_seed=42 \
    --total_n_step=40000 \
    --save_interval=5000 \
    --groups_in_dim=1 \
    --groups_out_dim=256 \
    --hn_groups=1 \
    --gamma=0.01 \
    --n_calib_samples="$N_CALIB" \
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
    send_notification "Calib${N_CALIB} Success" "job finished"
else
    send_notification "Calib${N_CALIB} Failed" "job crashed"
fi
exit "$status"
