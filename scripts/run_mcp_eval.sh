#!/usr/bin/env bash
# run_mcp_eval.sh — Baseline 2: GUI + MCP inference (no training), test_all_no_internet.json 309 tasks
#
# Usage:
#   # Thinking model (default, aligned with OSWorld-MCP)
#   nohup bash scripts/run_mcp_eval.sh > /tmp/mcp_eval.log 2>&1 &
#
#   # Instruct model
#   MODEL_TYPE=instruct bash scripts/run_mcp_eval.sh
#
#   # Skip vLLM startup (when a server is already running)
#   SKIP_VLLM=1 bash scripts/run_mcp_eval.sh
#
# Environment variables (all have defaults):
#   MODEL_TYPE    thinking | instruct (default thinking)
#   MODEL         model path (overrides the MODEL_TYPE auto-selection)
#   MAX_TOKENS    max output tokens (thinking default 2048, instruct default 2048)
#   RESULT_DIR    results directory (default <repo_root>/baselines/b2_mcp_<model_type>)
#   TEST_META     task set (default test_nogdrive.json, 361 tasks)
#   MAX_STEPS     max steps per task (default 30)
#   NUM_ENVS      number of concurrent environments (default 60, reduces docker lock contention)
#   SKIP_VLLM     =1 to skip vLLM startup
#   OSWORLD_BOOT_SLOTS  host-level VM concurrent-boot limit (default 8, prevents dockerd bursts)
#   CONTEXT_POLICY       baseline | skip_on_mcp_success | skip_on_no_change (default baseline)
#   MAX_CONSECUTIVE_SKIPS  max consecutive skips for skip_on_no_change (default 2)
#   MAX_IMAGE_HISTORY_LENGTH  screenshot history window size (default 4)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-thinking}"
NUM_GPU_PER_MODEL=1         # 8 GPU / 1 = 8 vLLM instances (aligned with run_puregui_eval.sh)
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.90
MAX_STEPS="${MAX_STEPS:-50}"
NUM_ENVS="${NUM_ENVS:-96}"
TEST_META="${TEST_META:-evaluation_examples/test_all_no_internet.json}"
SKIP_VLLM="${SKIP_VLLM:-0}"
export OSWORLD_BOOT_SLOTS="${OSWORLD_BOOT_SLOTS:-8}"
CONTEXT_POLICY="${CONTEXT_POLICY:-baseline}"
MAX_CONSECUTIVE_SKIPS="${MAX_CONSECUTIVE_SKIPS:-2}"
MAX_IMAGE_HISTORY_LENGTH="${MAX_IMAGE_HISTORY_LENGTH:-4}"

# MCP tool set injected into the VM. Default = the vendored tool set under
# mcp_tools/ (the set behind the paper's numbers). Set MCP_SRC_ROOT=0 to fall
# back to a stock OSWorld-MCP checkout, or point it at another tools dir.
_REPO_ROOT_MCP="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${MCP_SRC_ROOT:-}" == "0" ]]; then
    unset MCP_SRC_ROOT
    echo "[mcp] using stock OSWorld-MCP tools (MCP_SRC_ROOT=0)"
else
    export MCP_SRC_ROOT="${MCP_SRC_ROOT:-${_REPO_ROOT_MCP}/mcp_tools}"
    echo "[mcp] using MCP tools from ${MCP_SRC_ROOT}"
fi

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_INSTRUCT="${MODEL_INSTRUCT:-${MODEL_DIR}/Qwen3-VL-8B-Instruct}"
MODEL_THINKING="${MODEL_THINKING:-${MODEL_DIR}/Qwen3-VL-8B-Thinking}"

if [[ -z "${MODEL:-}" ]]; then
    if [[ "$MODEL_TYPE" == "thinking" ]]; then
        MODEL="$MODEL_THINKING"
        MAX_TOKENS="${MAX_TOKENS:-8192}"
    else
        MODEL="$MODEL_INSTRUCT"
        MAX_TOKENS="${MAX_TOKENS:-4096}"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-encode context strategy into RESULT_DIR suffix
_CTX_SUFFIX=""
if [[ "$CONTEXT_POLICY" != "baseline" ]]; then
    _CTX_SUFFIX="_${CONTEXT_POLICY}"
fi
_IMG_SUFFIX=""
if [[ "$MAX_IMAGE_HISTORY_LENGTH" != "4" ]]; then
    _IMG_SUFFIX="_img${MAX_IMAGE_HISTORY_LENGTH}"
fi
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/baselines/b2_mcp_${MODEL_TYPE}${_CTX_SUFFIX}${_IMG_SUFFIX}}"

# ── Change into the OSWorld-main directory ───────────────────────────────────
cd "$REPO_ROOT/OSWorld-main"

set +e
__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; fi
unset __conda_setup
conda activate rlanything 2>/dev/null || true
set -e
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "======================================================"
echo "  Baseline 2: GUI + MCP Eval (full eval, no training)"
echo "  MODEL_TYPE   : $MODEL_TYPE"
echo "  MODEL        : $MODEL"
echo "  MAX_TOKENS   : $MAX_TOKENS"
echo "  MAX_STEPS    : $MAX_STEPS"
echo "  NUM_ENVS     : $NUM_ENVS"
echo "  TEST_META    : $TEST_META"
echo "  RESULT_DIR   : $RESULT_DIR"
echo "  CONTEXT_POLICY: $CONTEXT_POLICY"
echo "  IMG_HIST_LEN : $MAX_IMAGE_HISTORY_LENGTH"
echo "  MCP_SRC_ROOT : ${MCP_SRC_ROOT:-OSWorld-MCP default}"
echo "======================================================"

# ── Step 1: clean up leftover Docker containers ─────────────────────────────
echo ""
echo "[1/4] Cleaning up Docker containers..."
removed=$(docker ps -aq | xargs docker rm -f 2>/dev/null | wc -l || echo 0)
echo "      Removed ${removed} containers"

# ── Step 2: clean up leftover vLLM/GPU processes ───────────────────────────
echo ""
echo "[2/4] Cleaning up vLLM processes..."
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "rl_rollout_local_qwen3vl" 2>/dev/null || true
sleep 5
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo "?")
echo "      GPU 0 free memory: ${free_mem} MiB"

# ── Step 3: start vLLM ──────────────────────────────────────────────────────
if [[ "$SKIP_VLLM" == "1" ]]; then
    echo ""
    echo "[3/4] Skipping vLLM startup (SKIP_VLLM=1)"
    if [[ -f .env ]]; then source .env; fi
else
    echo ""
    echo "[3/4] Starting vLLM (${NUM_GPU_PER_MODEL} GPU/model × 8 instances)..."
    MODEL="$MODEL" \
    NUM_GPU_PER_MODEL="$NUM_GPU_PER_MODEL" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    SERVED_MODEL_NAME="$MODEL" \
    bash start_8gpus_qwen3vl.sh
    source .env
fi

export QWEN3VL_LOCAL_ENDPOINTS="${OPENCUA_LOCAL_ENDPOINTS}"
export QWEN3VL_API_KEY="${OPENCUA_API_KEY:-dummy}"

# ── Step 4: loop the eval until every task has a result ─────────────────────
echo ""
echo "[4/4] Starting eval (missing tasks are re-run automatically)..."

_count_done() {
    local model_tag
    model_tag=$(python3 -c "import re; print(re.sub(r'[\\\\/]+','_','$MODEL').lstrip('_'))")
    ( set +o pipefail; find "$RESULT_DIR/hybrid/screenshot/$model_tag" -name "result.txt" 2>/dev/null | wc -l )
}

_total_tasks() {
    python3 -c "
import json
d = json.load(open('$TEST_META'))
print(sum(len(v) for v in d.values()))
"
}

TOTAL=$(_total_tasks)
echo "      Total tasks: $TOTAL"

RUN=0
while true; do
    RUN=$((RUN + 1))
    DONE=$(_count_done)
    echo ""
    echo "  --- Run $RUN: completed ${DONE}/${TOTAL} ---"

    if [[ "$DONE" -ge "$TOTAL" ]]; then
        echo "      All tasks completed!"
        break
    fi

    docker ps -aq | xargs -r docker rm -f 2>/dev/null | wc -l | xargs -I{} echo "      Docker cleanup: removed {} containers" || true

    NO_PROXY='localhost,127.0.0.1' no_proxy='localhost,127.0.0.1' \
    python rl_rollout_local_qwen3vl.py \
        --headless \
        --action_space hybrid \
        --observation_type screenshot \
        --max_steps "$MAX_STEPS" \
        --max_image_history_length "$MAX_IMAGE_HISTORY_LENGTH" \
        --context_policy "$CONTEXT_POLICY" \
        --max_consecutive_skips "$MAX_CONSECUTIVE_SKIPS" \
        --model "$MODEL" \
        --temperature 0 \
        --top_p 0.9 \
        --max_tokens "$MAX_TOKENS" \
        --test_config_base_dir evaluation_examples \
        --test_all_meta_path "$TEST_META" \
        --result_dir "$RESULT_DIR" \
        --num_envs "$NUM_ENVS" \
        --num_rollout_per_trial 1 \
        --rollout_type evaluation \
        --coevolveenv FALSE \
        --path_to_vm /dev/shm/Ubuntu-MCP.qcow2 \
        --provider_name docker \
        --domain all \
        --example all \
        || true

    DONE_AFTER=$(_count_done)
    if [[ "$DONE_AFTER" -le "$DONE" ]]; then
        echo ""
        echo "  [WARNING] Run $RUN completed no new tasks (${DONE_AFTER} vs ${DONE}), exiting."
        break
    fi
done

# ── Print final results ──────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Final results"
echo "======================================================"

python3 - "$MODEL" "$RESULT_DIR" "$TEST_META" <<'PYEOF'
import os, glob, json, re, sys

model_arg, result_dir, test_meta = sys.argv[1], sys.argv[2], sys.argv[3]
model_tag = re.sub(r'[\\/]+', '_', model_arg).lstrip('_')

base = os.path.join(result_dir, 'hybrid', 'screenshot', model_tag)
results = glob.glob(os.path.join(base, '*', '*', '0', 'result.txt'))

by_app = {}
for r in results:
    app = r.split('/')[-4]
    try:
        s = float(open(r).read().strip())
        by_app.setdefault(app, []).append(s)
    except:
        pass

all_s = [s for ss in by_app.values() for s in ss]
total = len(all_s)
total_tasks = sum(len(v) for v in json.load(open(test_meta)).values())
strict = sum(s == 1.0 for s in all_s)
partial = sum(all_s)

print(f"Done   : {total} / {total_tasks}")
print(f"Strict : {strict/total_tasks*100:.1f}%  ({strict} tasks)")
print(f"Partial: {partial/total_tasks*100:.1f}%")
print()
print("Per-app breakdown:")
for app, ss in sorted(by_app.items()):
    s_cnt = sum(s == 1.0 for s in ss)
    print(f"  {app:<25s} {len(ss):3d} tasks  strict={s_cnt/len(ss)*100:5.1f}%")
PYEOF
