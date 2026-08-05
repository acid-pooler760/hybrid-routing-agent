#!/usr/bin/env bash
# Copy to env.sh and `source env.sh` before running training / eval.
# These are the machine-specific paths that used to be hardcoded. Every value
# has a sensible default in the configs/scripts, so you only need to set what
# differs on your host.

# --- model weights ---
# Directory that CONTAINS Qwen3-VL-8B-Thinking/ (and optionally -Instruct/).
# Download from HuggingFace: Qwen/Qwen3-VL-8B-Thinking
export MODEL_DIR="./models"

# --- repo root (absolute path recommended for multi-node) ---
export RL_BASE_DIR="$(pwd)"

# --- HuggingFace cache ---
export HF_HOME="$HOME/.cache/huggingface"

# --- local scratch for transient per-step .pt artifacts ---
# RL rewrites 0.4-1.8 TB of .pt per step; keep this on a FAST LOCAL disk, not a
# shared/network filesystem. Unset = keep everything under runs/ (network fs).
export OSWORLD_LOCAL_TEMP="/tmp/osworld_temp"

# --- MCP tools ---
# Defaults to the vendored tool set in mcp_tools/ (the set behind the paper's
# numbers). Set MCP_SRC_ROOT=0 to fall back to a stock OSWorld-MCP checkout.
export MCP_SRC_ROOT="$PWD/mcp_tools"
# export MCP_TOOLS_DIR="mcp_tools/mcp/mcp_server/tools/apis"   # for prepare_splits_v2.py tool counts

# --- multi-node training ---
export SSH_USER="root"          # user for `ssh <user>@<host>` across nodes
export OSWORLD_BOOT_SLOTS="8"   # host-wide cap on concurrent VM boots

# --- optional HTTP proxy (leave unset if your host has direct internet) ---
# export download_proxy="http://<PROXY_HOST>:<PORT>"
