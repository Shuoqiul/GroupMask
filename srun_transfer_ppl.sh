#!/usr/bin/env bash
#SBATCH --job-name=ppl-sweep
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=ppl_sweep_%j.out
#SBATCH --error=ppl_sweep_%j.err

set -o pipefail

# ========================= 环境 =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export PYTHONUNBUFFERED=1          # log 实时刷进 tee，不再假死
export MASTER_ADDR=127.0.0.1       # torchrun rendezvous 显式指定，防挂起
set -x                             # 每条命令 trace 进 .out，卡住能看到位置

# ========================= 配置区 =========================
CKPT_DIR="/orange/sgao1/sli/semi-structure-merge/GroupMask/outputs/groupsparsity/2026-09-06_23-29-16"
STEPS_LIST="10000 20000 30000 40000 50000 60000 70000 80000 90000 100000"
MASTER_PORT_BASE=29510
EVAL_SCRIPT="/orange/sgao1/sli/semi-structure-merge/GroupMask/hf_ppl.py"
SAVE_HF_DIR_ROOT="/orange/sgao1/sli/semi-structure-merge/GroupMask/outputs/ppl-sweep"

HF_MODEL="meta-llama/Llama-2-7b-hf"
BASELINE_MODEL="meta-llama/Llama-2-7b-hf"
CKPT_GLOB="hn-ckpt-iter-*.pt"

# 每 step 的 eval 硬超时（防止单个卡死烧光 walltime）
EVAL_TIMEOUT=2h

# 与训练一致的 eval 配置（block_size 必须 = hn_block_size）
HN_GROUPS=1; BLOCK_SIZE=2048; BATCH_LIMIT=2000
GROUPS_IN_DIM=1; GROUPS_OUT_DIM=256
GATE_MODE="hard"; SIMPLE_GATE="false"; IS_SIMPLE_GATE="false"
SHARE_QK="false"; EVAL_BASELINE="false"
# =========================================================

send_notification() {
    local title="$1" message="$2"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
    fi
}

# ---------- 收集 checkpoint 列表 ----------
CKPTS=()
for s in $STEPS_LIST; do
    f="${CKPT_DIR}/hn-ckpt-iter-$(printf '%06d' "$s").pt"
    if [ -f "$f" ]; then
        CKPTS+=("$f")
    else
        echo "[WARN] missing checkpoint: $f"
    fi
done

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[ERROR] no checkpoints found in ${CKPT_DIR}" >&2
    exit 1
fi

echo "[INFO] ${#CKPTS[@]} checkpoints to evaluate:"
printf '  %s\n' "${CKPTS[@]}"

# ---------- 逐个评测 ----------
RESULT_LOG="ppl_results_${SLURM_JOB_ID:-manual}.log"
: > "$RESULT_LOG"

FAILED=0
for ckpt in "${CKPTS[@]}"; do
    bname=$(basename "$ckpt" .pt)                       # e.g. hn-ckpt-iter-020000
    step=$(echo "$bname" | grep -oE '[0-9]+')
    step=$((10#$step))                                  # 强制十进制，避开八进制坑
    save_dir="${SAVE_HF_DIR_ROOT}/${bname}"

    echo ""
    echo "=================================================================="
    echo "[EVAL] step=${step}  ckpt=${ckpt}"
    echo "=================================================================="

    timeout --signal=KILL "${EVAL_TIMEOUT}" \
    torchrun \
        --master_addr="$MASTER_ADDR" \
        --master_port=$(( MASTER_PORT_BASE + step % 1000 )) \
        --nproc_per_node=1 \
        "$EVAL_SCRIPT" \
        --hf_model "$HF_MODEL" \
        --dataset "wikitext" \
        --semi_evaluate true \
        --eval_baseline "$EVAL_BASELINE" \
        --baseline_model_name "$BASELINE_MODEL" \
        --hn_path "$ckpt" \
        --hn_groups="$HN_GROUPS" \
        --block_size="$BLOCK_SIZE" \
        --batch_limit="$BATCH_LIMIT" \
        --groups_in_dim="$GROUPS_IN_DIM" \
        --groups_out_dim="$GROUPS_OUT_DIM" \
        --simple_gate="$SIMPLE_GATE" \
        --gate_mode="$GATE_MODE" \
        --save_hf_dir "$save_dir" \
        --share_qk="$SHARE_QK" \
        --is_simple_gate="$IS_SIMPLE_GATE" \
        2>&1 | tee "eval_step_${step}.log"

    status=${PIPESTATUS[0]}
    if [ "$status" -eq 137 ]; then
        echo "[TIMEOUT] step=${step} eval exceeded ${EVAL_TIMEOUT}, killed"
        echo -e "${step}\tTIMEOUT" >> "$RESULT_LOG"
        FAILED=$((FAILED+1))
        continue
    fi
    if [ "$status" -ne 0 ]; then
        echo "[FAIL] step=${step} exited with $status"
        echo -e "${step}\tFAIL" >> "$RESULT_LOG"
        FAILED=$((FAILED+1))
        continue
    fi

    # 抓 PPL：精确匹配 "Perplexity on wikitext: <float>"
    ppl=$(grep -oE "Perplexity on wikitext: [0-9.]+" "eval_step_${step}.log" \
          | grep -oE "[0-9]+\.[0-9]+" | tail -1)

    if [ -z "$ppl" ]; then
        echo "[WARN] could not parse PPL for step=${step}, check eval_step_${step}.log"
        echo -e "${step}\tPARSE_FAIL" >> "$RESULT_LOG"
        FAILED=$((FAILED+1))
    else
        echo "[OK] step=${step}  wikitext_ppl=${ppl}"
        echo -e "${step}\t${ppl}" >> "$RESULT_LOG"
    fi
done

# ---------- 汇总表 ----------
echo ""
echo ""
echo "#################### SUMMARY ####################"
printf "%-12s %-15s\n" "STEP" "WIKITEXT_PPL"
echo "---------------------------------------------"
sort -n "$RESULT_LOG" | awk -F'\t' '
    $2=="FAIL"        {printf "%-12s %-15s\n", $1, "RUN_FAILED"}
    $2=="TIMEOUT"     {printf "%-12s %-15s\n", $1, "TIMEOUT"}
    $2=="PARSE_FAIL"  {printf "%-12s %-15s\n", $1, "PARSE_FAILED"}
    {printf "%-12s %-15s\n", $1, $2}
'
echo "#################################################"
echo "(raw results in: $RESULT_LOG ; full per-step logs: eval_step_*.log)"

if [ "$FAILED" -gt 0 ]; then
    send_notification "PPL Sweep Partial Failure" "${FAILED} evals failed, see log"
else
    send_notification "PPL Sweep Success" "all ${#CKPTS[@]} evals finished"
fi
