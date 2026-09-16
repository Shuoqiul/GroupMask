#!/usr/bin/env bash
#SBATCH --job-name=uniform-40k
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=train_uniform_alloc_%j.out
#SBATCH --error=train_uniform_alloc_%j.err

# ============================================================================
# 阶段四 | Uniform vs Adaptive (全文最核心的 novelty 消融)
#   同样的 group 大小 (1x256), 同一个 hypernetwork 架构、lr、数据、步数,
#   唯一区别: --uniform_alloc=true -> 正则从"全局预算"改为"每层各自 pin 到 50%"
#   (组内 mask 放置仍然可学, 只锁死跨层分配)
#   对照组 = 现有 hypernetwork 40k run (全局预算自适应分配), 不用重跑
#   训练中每 500 步写 layer_allocation.csv, 直接画跨层分配对比图
#   正面回应 ProxSparse Appendix E (他们证明 N:M 细颗粒度下无效)
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

RUN_NAME="uniform_alloc_40k"
OUT_DIR="${GROUPMASK_OUTPUT_ROOT}/groupsparsity/${RUN_NAME}"

NPROC_PER_NODE=1
MASTER_PORT=29514
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
    --total_n_step=40000 \
    --save_interval=5000 \
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
    --uniform_alloc=true \
    --p=0.5 \
    --T=0.4 \
    --lam=16 \
    --share_qk=false \
    --hidden_kd=false \
    --adam_8bit=true \
    2>&1 | tee "$LOG_FILE"

status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then
    send_notification "UniformAlloc 40k Success" "$RUN_NAME finished"
else
    send_notification "UniformAlloc 40k Failed" "$RUN_NAME crashed"
fi
exit "$status"
