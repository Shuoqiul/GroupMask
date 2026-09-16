#!/usr/bin/env bash
#SBATCH --job-name=prior-hn-20k
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=train_prior_hn_wanda_%j.out
#SBATCH --error=train_prior_hn_wanda_%j.err

# ============================================================================
# 阶段五 | Prior 注入迁移到 hypernetwork (档位1 wanda)
#   插入点 = tp_out 线性投影之后、train/eval 分支之前, train/eval 共用
#   前提: prior_simple_wanda 已验证 prior 信号有效再提交本脚本
#   注意: hypernetwork 的 prior 偏移是 forward 期的, 不进 state_dict;
#   训练会写 sidecar prior_scores.pt, eval 时 srun_eval_sweep.sh 自动带上
# ============================================================================
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

RUN_NAME="prior_hn_wanda_a1.0_20k"
OUT_DIR="${GROUPMASK_OUTPUT_ROOT}/groupsparsity/${RUN_NAME}"

NPROC_PER_NODE=1
MASTER_PORT=29518
HF_MODEL="meta-llama/Llama-2-7b-hf"
# ========================================================

mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
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
    --exp_name="groupsparsity" \
    --out_dir="${OUT_DIR}" \
    --batch_size=1 \
    --hf_model="$HF_MODEL" \
    --compile_flag=false \
    --use_fsdp=true \
    --use_bf16=true \
    --use_minipile=false \
    --dataset_seed=42 \
    --dataset_list=['c4'] \
    --total_n_step=20000 \
    --save_interval=2000 \
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
    --prior_mode=wanda \
    --prior_alpha=1.0 \
    --prior_n_samples=8 \
    --p=0.5 \
    --T=0.4 \
    --lam=16 \
    --share_qk=false \
    --hidden_kd=false \
    --adam_8bit=true \
    2>&1 | tee "$LOG_FILE"

status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then
    send_notification "PriorHN 20k Success" "$RUN_NAME finished"
else
    send_notification "PriorHN 20k Failed" "$RUN_NAME crashed"
fi
exit "$status"
