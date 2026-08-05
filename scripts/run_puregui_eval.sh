#!/usr/bin/env bash
# run_puregui_eval.sh — one-shot script for the pure-GUI eval (no tools)
#
# Usage:
#   # Instruct model (default)
#   bash scripts/run_puregui_eval.sh
#
#   # Thinking model
#   MODEL_TYPE=thinking bash scripts/run_puregui_eval.sh
#
#   # Custom model path and results directory
#   MODEL=/path/to/model RESULT_DIR=/path/to/results bash scripts/run_puregui_eval.sh
#
# Environment variables (all have defaults):
#   MODEL_TYPE    instruct | thinking (default instruct)
#   MODEL         absolute model path (overrides the MODEL_TYPE auto-selection)
#   MAX_TOKENS    max output tokens (instruct default 4096, thinking default 8192)
#   RESULT_DIR    results directory (default <repo_root>/baselines/b1_puregui_<model_type>)
#   TEST_META     task set JSON (default test_all_no_internet.json, 309 tasks)
#   MAX_STEPS     max steps per task (default 50, aligned with ToolCUA)
#   NUM_ENVS      number of concurrent environments (default 96)
#   SKIP_VLLM     =1 to skip vLLM startup (when a server is already running)

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-instruct}"
NUM_GPU_PER_MODEL=1
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.85
MAX_STEPS="${MAX_STEPS:-50}"
NUM_ENVS="${NUM_ENVS:-96}"
TEST_META="${TEST_META:-evaluation_examples/test_all_no_internet.json}"
SKIP_VLLM="${SKIP_VLLM:-0}"

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_INSTRUCT="${MODEL_INSTRUCT:-${MODEL_DIR}/Qwen3-VL-8B-Instruct}"
MODEL_THINKING="${MODEL_THINKING:-${MODEL_DIR}/Qwen3-VL-8B-Thinking}"

if [[ -z "${MODEL:-}" ]]; then
    if [[ "$MODEL_TYPE" == "thinking" ]]; then
        MODEL="$MODEL_THINKING"
        MAX_TOKENS="${MAX_TOKENS:-8192}"    # thinking models get a larger output budget
    else
        MODEL="$MODEL_INSTRUCT"
        MAX_TOKENS="${MAX_TOKENS:-4096}"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/baselines/b1_puregui_${MODEL_TYPE}}"

# ── Change into the OSWorld-main directory ───────────────────────────────────
cd "$REPO_ROOT/OSWorld-main"

# Two spots conflict with `set -euo pipefail`, and the script already does manual
# error handling via `|| true` plus the loop's exit checks:
#   1) source ~/.bashrc → /etc/bashrc references the undefined $BASHRCSOURCED,
#      so set -u kills the shell on the spot (`|| true` cannot save it: the exit
#      happens inside `source`, before control reaches the outer ||).
#   2) In the eval loop, _count_done runs find on a results directory that may
#      not exist yet, returning nonzero; combined with pipefail this makes
#      `DONE=$(_count_done)` silently exit the whole script under set -e.
# So from here on disable errexit/nounset/pipefail (same set +e approach as
# run_mcp_eval.sh).
set +euo pipefail
source ~/.bashrc 2>/dev/null || true
source activate rlanything 2>/dev/null || conda activate rlanything 2>/dev/null || true

echo "======================================================"
echo "  Pure GUI Eval"
echo "  MODEL_TYPE : $MODEL_TYPE"
echo "  MODEL      : $MODEL"
echo "  MAX_TOKENS : $MAX_TOKENS"
echo "  MAX_STEPS  : $MAX_STEPS"
echo "  NUM_ENVS   : $NUM_ENVS"
echo "  TEST_META  : $TEST_META"
echo "  RESULT_DIR : $RESULT_DIR"
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
    find "$RESULT_DIR/pyautogui/screenshot/$model_tag" -name "result.txt" 2>/dev/null | wc -l
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

    # Clean up Docker before each run
    docker ps -aq | xargs docker rm -f 2>/dev/null | wc -l | xargs -I{} echo "      Docker cleanup: removed {} containers"

    python rl_rollout_local_qwen3vl.py \
        --headless \
        --action_space pyautogui \
        --observation_type screenshot \
        --max_steps "$MAX_STEPS" \
        --max_image_history_length 4 \
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
        || true   # a worker crash must not abort the outer loop; the next run picks up the slack

    DONE_AFTER=$(_count_done)
    if [[ "$DONE_AFTER" -le "$DONE" ]]; then
        echo ""
        echo "  [WARNING] Run $RUN completed no new tasks (${DONE_AFTER} vs ${DONE})"
        echo "         Possible vLLM issue or persistently crashing tasks, exiting."
        break
    fi
done

# ── Print final results ──────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Final results"
echo "======================================================"

MODEL_LOCAL="$MODEL"  # passed into python
python3 - <<PYEOF
import os, glob, json, re

model_tag = re.sub(r'[\\\\/]+', '_', '$MODEL_LOCAL').lstrip('_')
base = os.path.join('$RESULT_DIR', 'pyautogui', 'screenshot', model_tag)
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
total_tasks = sum(len(v) for v in json.load(open('$TEST_META')).values())
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
