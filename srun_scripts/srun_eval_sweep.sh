#!/usr/bin/env bash
#SBATCH --job-name=ppl-sweep2
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=ppl_sweep2_%j.out
#SBATCH --error=ppl_sweep2_%j.err

# ============================================================================
# 通用 checkpoint -> PPL sweep (token-vs-PPL 曲线 / 步数消融 / C4->wiki 泛化)
#
# 用法:
#   1) 直接 sbatch: 自动选 outputs/groupsparsity 下最新的 run, 评它全部
#      hn-ckpt-iter-*.pt (wikitext + ptb)
#   2) 指定 run 和步数:
#      sbatch --export=CKPT_DIR=/path/to/run,STEPS_LIST="2000 4000 6000" \
#            srun_eval_sweep.sh
#   3) 评 simple_gate 的 run: sbatch --export=CKPT_DIR=...,SIMPLE_GATE=true \
#            srun_eval_sweep.sh
#   hypernetwork+prior 的 run 会自动读 run 目录下的 prior_scores.pt 挂回偏移
#   (simple_gate 的 prior 烘焙在 ckpt 里, 不需要也不允许再挂)
# ============================================================================
set -o pipefail

# ========================= 环境 =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
set -x

# ========================= 配置区 =========================
BASE=/orange/sgao1/sli/semi-structure-merge/GroupMask
EVAL_SCRIPT="${BASE}/hf_ppl.py"
export GROUPMASK_OUTPUT_ROOT="${BASE}/outputs"
SAVE_HF_DIR_ROOT="${BASE}/outputs/ppl-sweep"

HF_MODEL="meta-llama/Llama-2-7b-hf"
BASELINE_MODEL="meta-llama/Llama-2-7b-hf"

CKPT_DIR="${CKPT_DIR:-}"            # 空 = 自动选最新 run
STEPS_LIST="${STEPS_LIST:-}"        # 空 = 评该 run 全部 iter checkpoint
DATASET="${DATASET:-wikitext,ptb}"  # MaskLLM 协议: c4 训练, wikitext 主评; ptb 看泛化

# 与训练一致的 eval 配置 (block_size 必须 = hn_block_size)
HN_GROUPS=1; BLOCK_SIZE=2048; BATCH_LIMIT=2000
GROUPS_IN_DIM=1; GROUPS_OUT_DIM=256
GATE_MODE="hard"
SIMPLE_GATE="${SIMPLE_GATE:-false}"
IS_SIMPLE_GATE="${IS_SIMPLE_GATE:-$SIMPLE_GATE}"
SHARE_QK="false"; EVAL_BASELINE="false"
PRIOR_SCORES_PATH="${PRIOR_SCORES_PATH:-}"   # 留空则自动探测

EVAL_TIMEOUT=2h
MASTER_PORT_BASE=29510
# =========================================================

send_notification() {
    local title="$1" message="$2"
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -H "Title: $title" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
    fi
}

# ---------- 定位 run 目录 ----------
if [ -z "$CKPT_DIR" ]; then
    ROOT="${GROUPMASK_OUTPUT_ROOT}/groupsparsity"
    CKPT_DIR=$(ls -dt "$ROOT"/*/ 2>/dev/null | head -1)
    CKPT_DIR="${CKPT_DIR%/}"
fi
if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ]; then
    echo "[ERROR] no run directory found under ${GROUPMASK_OUTPUT_ROOT}/groupsparsity (or CKPT_DIR invalid)" >&2
    exit 1
fi
echo "[INFO] evaluating run dir: $CKPT_DIR"

# ---------- hypernetwork prior sidecar 自动探测 ----------
if [ -z "$PRIOR_SCORES_PATH" ] && [ "$SIMPLE_GATE" != "true" ] && [ -f "${CKPT_DIR}/prior_scores.pt" ]; then
    PRIOR_SCORES_PATH="${CKPT_DIR}/prior_scores.pt"
    echo "[INFO] detected prior sidecar: $PRIOR_SCORES_PATH"
fi
PRIOR_ARGS=()
if [ -n "$PRIOR_SCORES_PATH" ]; then
    PRIOR_ARGS=(--prior_scores_path "$PRIOR_SCORES_PATH")
fi

# ---------- 收集 checkpoint 列表 ----------
CKPTS=()
if [ -n "$STEPS_LIST" ]; then
    for s in $STEPS_LIST; do
        f="${CKPT_DIR}/hn-ckpt-iter-$(printf '%06d' "$s").pt"
        if [ -f "$f" ]; then
            CKPTS+=("$f")
        else
            echo "[WARN] missing checkpoint: $f"
        fi
    done
else
    # 文件名 6 位零填充, 字典序即步数序
    for f in $(ls "${CKPT_DIR}"/hn-ckpt-iter-*.pt 2>/dev/null | sort); do
        CKPTS+=("$f")
    done
fi

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[ERROR] no checkpoints found in ${CKPT_DIR}" >&2
    exit 1
fi

echo "[INFO] ${#CKPTS[@]} checkpoints to evaluate:"
printf '  %s\n' "${CKPTS[@]}"

# ---------- 逐个评测 ----------
RUN_TAG=$(basename "$CKPT_DIR")
RESULT_LOG="ppl_results_${RUN_TAG}_${SLURM_JOB_ID:-manual}.log"
: > "$RESULT_LOG"

FAILED=0
for ckpt in "${CKPTS[@]}"; do
    bname=$(basename "$ckpt" .pt)                       # e.g. hn-ckpt-iter-020000
    step=$(echo "$bname" | grep -oE '[0-9]+')
    step=$((10#$step))                                  # 强制十进制，避开八进制坑
    save_dir="${SAVE_HF_DIR_ROOT}/${RUN_TAG}/${bname}"
    mkdir -p "$save_dir"

    echo ""
    echo "=================================================================="
    echo "[EVAL] step=${step}  ckpt=${ckpt}  dataset=${DATASET}"
    echo "=================================================================="

    timeout --signal=KILL "${EVAL_TIMEOUT}" \
    torchrun \
        --master_addr="$MASTER_ADDR" \
        --master_port=$(( MASTER_PORT_BASE + step % 1000 )) \
        --nproc_per_node=1 \
        "$EVAL_SCRIPT" \
        --hf_model "$HF_MODEL" \
        --dataset "$DATASET" \
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
        "${PRIOR_ARGS[@]}" \
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

    # 抓 PPL：精确匹配 "Perplexity on <ds>: <float>"
    row="${step}"
    ok=1
    for ds in $(echo "$DATASET" | tr ',' ' '); do
        ppl=$(grep -oE "Perplexity on ${ds}: [0-9.]+" "eval_step_${step}.log" \
              | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        if [ -z "$ppl" ]; then
            echo "[WARN] could not parse ${ds} PPL for step=${step}"
            ok=0
            row="${row}\tNA"
        else
            row="${row}\t${ppl}"
        fi
    done
    if [ "$ok" -eq 1 ]; then
        echo "[OK] step=${step} parsed"
    else
        FAILED=$((FAILED+1))
    fi
    echo -e "$row" >> "$RESULT_LOG"
done

# ---------- 汇总表 ----------
echo ""
echo ""
echo "#################### SUMMARY (${RUN_TAG}) ####################"
printf "%-12s %-15s\n" "STEP" "PPLs (dataset order: ${DATASET})"
echo "---------------------------------------------"
sort -n "$RESULT_LOG" | awk -F'\t' '
    {printf "%-12s", $1; for (i=2; i<=NF; i++) printf " %-12s", $i; print ""}
'
echo "#################################################"
echo "(raw results in: $RESULT_LOG ; full per-step logs: eval_step_*.log)"

if [ "$FAILED" -gt 0 ]; then
    send_notification "PPL Sweep Partial Failure" "${FAILED} evals failed (${RUN_TAG})"
else
    send_notification "PPL Sweep Success" "all ${#CKPTS[@]} evals finished (${RUN_TAG})"
fi
