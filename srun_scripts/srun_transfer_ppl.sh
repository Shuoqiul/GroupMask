#!/usr/bin/env bash
#SBATCH --job-name=calib-ppl-sweep
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --array=0-3%2
#SBATCH --output=calib_ppl_sweep_%A_%a.out
#SBATCH --error=calib_ppl_sweep_%A_%a.err

set -o pipefail

# ========================= 环境 =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /orange/sgao1/ZhengaoLi/Envs/torch2.7.0

export HF_HOME=/orange/sgao1/sli/hf_cache
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
set -x

# ========================= 配置区 =========================
BASE="/orange/sgao1/sli/semi-structure-merge/GroupMask"
EVAL_SCRIPT="${BASE}/hf_ppl.py"
SAVE_HF_DIR_ROOT="${BASE}/outputs/ppl-sweep"

# ---- array 索引 → calib 目录映射 ----
CALIB_LIST=(100 200 400 800)
CALIB=${CALIB_LIST[$SLURM_ARRAY_TASK_ID]}

# ---- 自动解析“最新且含 hn-ckpt”的时间戳目录 ----
CKPT_ROOT="${BASE}/outputs/groupsparsity-calib${CALIB}"
if [ -n "${CKPT_DIR_OVERRIDE:-}" ]; then
    CKPT_DIR="${CKPT_DIR_OVERRIDE}"
else
    CKPT_DIR=""
    # 按 mtime 从新到旧逐个检查，第一个含 hn-ckpt-iter-*.pt 的目录即中选
    while IFS= read -r d; do
        if compgen -G "${d}/hn-ckpt-iter-*.pt" > /dev/null; then
            CKPT_DIR="$d"
            break
        fi
    done < <(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | cut -d' ' -f2-)

    if [ -z "$CKPT_DIR" ]; then
        echo "[ERROR] no run dir containing hn-ckpt-iter-*.pt under ${CKPT_ROOT}" >&2
        exit 1
    fi
fi
echo "[INFO] calib=${CALIB} resolved CKPT_DIR=${CKPT_DIR}"



STEPS_LIST="5000 10000 15000 20000 25000 30000 35000 40000"   # 每 5000 一步，共 8 个
CKPT_GLOB="hn-ckpt-iter-*.pt"

MASTER_PORT_BASE=29510
EVAL_TIMEOUT=2h

# 与训练一致的 eval 配置（block_size 必须 = hn_block_size=2048）
HF_MODEL="meta-llama/Llama-2-7b-hf"
BASELINE_MODEL="meta-llama/Llama-2-7b-hf"
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

# 文件名全部带 calib 标签，4 个 array task 并行不互相覆盖
RESULT_LOG="ppl_results_calib${CALIB}_${SLURM_ARRAY_JOB_ID:-manual}.log"
: > "$RESULT_LOG"

# ---------- 收集 checkpoint 列表 ----------
CKPTS=()
for s in $STEPS_LIST; do
    f="${CKPT_DIR}/hn-ckpt-iter-$(printf '%06d' "$s").pt"
    if [ -f "$f" ]; then
        CKPTS+=("$f")
    else
        echo "[WARN] calib${CALIB} missing checkpoint: $f"
    fi
done

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[ERROR] no checkpoints found in ${CKPT_DIR}" >&2
    exit 1
fi
echo "[INFO] calib=${CALIB}: ${#CKPTS[@]} checkpoints to evaluate"
printf ' %s\n' "${CKPTS[@]}"

# ---------- 逐个评测 ----------
FAILED=0
for ckpt in "${CKPTS[@]}"; do
    bname=$(basename "$ckpt" .pt)                     # e.g. hn-ckpt-iter-020000
    step=$(echo "$bname" | grep -oE '[0-9]+')
    step=$((10#$step))                                # 强制十进制，避开八进制坑
    save_dir="${SAVE_HF_DIR_ROOT}/calib${CALIB}/${bname}"
    eval_log="eval_calib${CALIB}_step_${step}.log"

    echo ""
    echo "=================================================================="
    echo "[EVAL] calib=${CALIB} step=${step} ckpt=${ckpt}"
    echo "=================================================================="

    timeout --signal=KILL "${EVAL_TIMEOUT}" \
    torchrun \
        --master_addr="$MASTER_ADDR" \
        --master_port=$(( MASTER_PORT_BASE + SLURM_ARRAY_TASK_ID * 100 + step % 100 )) \
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
        2>&1 | tee "$eval_log"
    status=${PIPESTATUS[0]}

    if [ "$status" -eq 137 ]; then
        echo "[TIMEOUT] calib=${CALIB} step=${step} exceeded ${EVAL_TIMEOUT}"
        echo -e "${step}\tTIMEOUT" >> "$RESULT_LOG"; FAILED=$((FAILED+1)); continue
    fi
    if [ "$status" -ne 0 ]; then
        echo "[FAIL] calib=${CALIB} step=${step} exited with $status"
        echo -e "${step}\tFAIL" >> "$RESULT_LOG"; FAILED=$((FAILED+1)); continue
    fi

    ppl=$(grep -oE "Perplexity on wikitext: [0-9.]+" "$eval_log" \
          | grep -oE "[0-9]+\.[0-9]+" | tail -1)
    if [ -z "$ppl" ]; then
        echo "[WARN] could not parse PPL for calib=${CALIB} step=${step}"
        echo -e "${step}\tPARSE_FAIL" >> "$RESULT_LOG"; FAILED=$((FAILED+1))
    else
        echo "[OK] calib=${CALIB} step=${step} wikitext_ppl=${ppl}"
        echo -e "${step}\t${ppl}" >> "$RESULT_LOG"
    fi
done

# ---------- 汇总表 ----------
echo ""
echo "################ SUMMARY: calib=${CALIB} ################"
printf "%-12s %-15s\n" "STEP" "WIKITEXT_PPL"
echo "---------------------------------------------"
sort -n "$RESULT_LOG" | awk -F'\t' '
    $2=="FAIL"        {printf "%-12s %-15s\n", $1, "RUN_FAILED"}
    $2=="TIMEOUT"     {printf "%-12s %-15s\n", $1, "TIMEOUT"}
    $2=="PARSE_FAIL"  {printf "%-12s %-15s\n", $1, "PARSE_FAILED"}
    {printf "%-12s %-15s\n", $1, $2}
'
echo "#########################################################"

if [ "$FAILED" -gt 0 ]; then
    send_notification "Calib${CALIB} PPL Sweep Partial Failure" "${FAILED} evals failed"
else
    send_notification "Calib${CALIB} PPL Sweep Success" "all ${#CKPTS[@]} evals finished"
fi
