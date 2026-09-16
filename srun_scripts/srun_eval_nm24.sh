#!/usr/bin/env bash
#SBATCH --job-name=nm24-base
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=eval_nm24_%j.out
#SBATCH --error=eval_nm24_%j.err

# ============================================================================
# 阶段二 | 标准 N:M (2:4) magnitude baseline, 摆进主表用
#   同模型 (Llama-2-7b)、同稀疏度量级 (2:4 = 50%), wikitext + ptb 评估
#   纯评估 job, 不需要 hn checkpoint; 也把烘焙后的 HF 模型存下来备用
# ============================================================================
set -o pipefail
set -x

# ========================= 环境 =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1

# ========================= 配置区 =========================
BASE=/orange/sgao1/sli/semi-structure-merge/GroupMask
EVAL_SCRIPT="${BASE}/hf_ppl.py"

HF_MODEL="meta-llama/Llama-2-7b-hf"
DATASET="wikitext,ptb"
BLOCK_SIZE=2048
BATCH_LIMIT=2000
SAVE_HF_DIR="${BASE}/outputs/nm24_baseline/llama2-7b_2of4_magnitude"

NM_N=2
NM_M=4
MASTER_PORT=29521
# =========================================================

mkdir -p "$SAVE_HF_DIR"
LOG_FILE="${BASE}/traininglogs/eval_nm24_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${BASE}/traininglogs"

send_notification() {
    local title="$1" message="$2"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
    fi
}

torchrun \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node=1 \
    "$EVAL_SCRIPT" \
    --hf_model "$HF_MODEL" \
    --dataset "$DATASET" \
    --nm_prune true \
    --nm_n "$NM_N" \
    --nm_m "$NM_M" \
    --block_size "$BLOCK_SIZE" \
    --batch_limit "$BATCH_LIMIT" \
    --save_hf_dir "$SAVE_HF_DIR" \
    2>&1 | tee "$LOG_FILE"

status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then
    send_notification "NM24 Baseline Success" "see $LOG_FILE"
else
    send_notification "NM24 Baseline Failed" "see $LOG_FILE"
fi
exit "$status"
