#!/usr/bin/env bash
# stop_training.sh — One-shot kill for the entire RL training pipeline.
#
# Stops (in order):
#   1. Main orchestrator (osworld_rl.py)
#   2. Rollout worker processes (rl_rollout_local_*.py)
#   3. Preprocess / merge / train subprocesses
#   4. vLLM inference servers (ports 8000-8007)
#   5. Docker containers (osworld-rl labeled)
#
# Usage:
#   bash stop_training.sh          # stop everything
#   bash stop_training.sh --wait   # stop and wait for GPU memory release (15s)

set -uo pipefail

WAIT=0
[[ "${1:-}" == "--wait" ]] && WAIT=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "== 1. Killing main orchestrator (osworld_rl.py)"
pids=$(pgrep -f "python osworld_rl.py" 2>/dev/null || true)
if [[ -n "$pids" ]]; then
    echo "  PIDs: $pids"
    kill -9 $pids 2>/dev/null || true
else
    echo "  Not running."
fi

echo ""
echo "== 2. Killing rollout workers (rl_rollout_local_*.py)"
pids=$(pgrep -f "rl_rollout_local_" 2>/dev/null || true)
if [[ -n "$pids" ]]; then
    echo "  PIDs: $pids"
    kill -9 $pids 2>/dev/null || true
else
    echo "  Not running."
fi

echo ""
echo "== 3. Killing preprocess / merge / train subprocesses"
for pattern in \
    "train.osworld_vlm_preprocess_shards" \
    "train.osworld_vlm_merge_preproc_shards" \
    "accelerate launch"; do
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "  [$pattern] PIDs: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
done

echo ""
echo "== 3c. Killing orphaned osworld_train.py workers (accelerate GPU processes)"
train_pids=$(pgrep -f "osworld_train.py" 2>/dev/null || true)
if [[ -n "$train_pids" ]]; then
    n=$(echo "$train_pids" | wc -w)
    echo "  Found $n orphaned train workers, killing..."
    kill -9 $train_pids 2>/dev/null || true
else
    echo "  None found."
fi

echo ""
echo "== 3b. Killing orphaned multiprocessing children (rlanything env)"
mp_pids=$(pgrep -f "multiprocessing\.(spawn|resource_tracker)" 2>/dev/null || true)
if [[ -n "$mp_pids" ]]; then
    n=$(echo "$mp_pids" | wc -w)
    mem=$(ps -p $(echo $mp_pids | tr ' ' ',') -o rss= 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1024/1024}')
    echo "  Found $n orphaned multiprocessing children (${mem} GB RSS), killing..."
    kill -9 $mp_pids 2>/dev/null || true
else
    echo "  None found."
fi

echo ""
echo "== 4. Killing vLLM servers (ports 8000-8007)"
# Method A: PID files
for port in $(seq 8000 8007); do
    pidfile="/tmp/opencua_pids/${port}.pid"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile" 2>/dev/null || true)
        if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
            echo "  Port $port → PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# Method B: process name patterns
for pattern in "vllm serve" "VLLM::EngineCore" "vllm.engine" "ray::IDLE"; do
    pkill -9 -f "$pattern" 2>/dev/null || true
done

# Method C: anything still holding ports 8000-8007
for port in $(seq 8000 8007); do
    pids=$(ss -tlnp 2>/dev/null \
        | awk -v p=":${port} " '$0~p{while(match($0,/pid=([0-9]+)/,a)){print a[1];$0=substr($0,RSTART+RLENGTH)}}' \
        | sort -u)
    for pid in $pids; do
        echo "  Port $port still held by PID $pid, killing..."
        kill -9 "$pid" 2>/dev/null || true
    done
done
echo "  vLLM cleanup done."

echo ""
echo "== 4b. Cleaning /dev/shm vLLM checkpoint copies"
# [TMPFS] copy of policy ckpt (~17GB each); leftovers from a killed run cause
# vLLM boot timeout on next start (incomplete safetensors) or fill up tmpfs.
shm_leftover=$(ls -d /dev/shm/vllm_ckpt* 2>/dev/null || true)
if [[ -n "$shm_leftover" ]]; then
    echo "  Removing: $shm_leftover"
    rm -rf /dev/shm/vllm_ckpt* 2>/dev/null || true
else
    echo "  Nothing to clean."
fi
df -h /dev/shm | tail -1 | awk '{print "  /dev/shm now: used " $3 " / " $2}'

echo ""
echo "== 5. Removing Docker containers (label=osworld-rl)"
container_ids=$(docker ps -aq --filter "label=osworld-rl=true" 2>/dev/null || true)
if [[ -n "$container_ids" ]]; then
    n=$(echo "$container_ids" | wc -w)
    echo "$container_ids" | xargs -r docker rm --force 2>/dev/null || true
    echo "  Removed $n container(s)."
else
    echo "  No containers found."
fi

if [[ $WAIT -eq 1 ]]; then
    echo ""
    echo "== Waiting 15s for GPU memory release..."
    sleep 15
    echo "  Done."
fi

echo ""
echo "[DONE] All training processes stopped."
echo ""
echo "Next steps:"
echo "  # Archive failed run (if needed):"
echo "  mv runs/<project> runs/archive/<project>_runN_reason"
echo ""
echo "  # Restart training:"
echo "  OSWORLD_LOCAL_TEMP=/tmp/osworld_temp TQDM_DISABLE=1 nohup python osworld_rl.py config=configs/experiments/<config>.yaml > logs/<name>.log 2>&1 &"
