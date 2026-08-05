# Environment setup

The evaluation and RL rollouts run real desktop tasks inside virtual machines. This document covers the VM provider, the MCP-in-VM architecture, the external assets you must fetch, and hardware requirements.

`setup/environment_rlanything.yml` is the exact conda environment.

## Architecture

```
osworld_rl.py / run_mcp_eval.sh
  └─ rl_rollout_local_qwen3vl.py
       ├─ N × DesktopEnv (parallel)  →  Docker container → QEMU → Ubuntu VM
       │                                    ├─ osworld.service :5000  (screenshot / pyautogui)
       │                                    └─ mcp_server      :9292  (MCP tools)
       └─ 8 × vLLM (Qwen3-VL-8B, ports 8001-8008) behind a round-robin proxy :8000
```

- **Provider = Docker wrapping QEMU** (`--provider_name docker`). Containers are labelled `osworld-rl=true`; the rollout driver cleans them between runs.
- **`OSWORLD_BOOT_SLOTS`** (default 8) throttles concurrent VM boots host-wide via file locks under `/dev/shm/osworld_boot_slots/`.
- **`OSWORLD_DISABLE_RECORDING=1`** (default) disables trajectory video.

## External assets (not in this repo)

| Asset | Where to get it | Points via |
|-------|-----------------|------------|
| **Policy model** | HuggingFace `Qwen/Qwen3-VL-8B-Thinking` (and `-Instruct`) | `$MODEL_DIR` |
| **VM base image** | Docker image `happysixd/osworld-docker` (upstream OSWorld) | `--provider_name docker` |
| **VM disk** | OSWorld `Ubuntu.qcow2` (upstream) → patch → `Ubuntu-MCP.qcow2` | `--path_to_vm`, `vm_path` |
| **MCP tools** | vendored in-repo under `mcp_tools/` (stock OSWorld-MCP is a fallback) | `MCP_SRC_ROOT` |

### The VM image (`Ubuntu-MCP.qcow2`)

The runtime VM is the upstream OSWorld `Ubuntu.qcow2` with a **small patch**: the MCP runtime dependency (`fastmcp`) is pip-installed into it. Everything else is stock OSWorld:
- `osworld.service` (:5000, screenshot / pyautogui) — already in the stock image.
- The **MCP server code is NOT baked in** — the qcow2's `/home/user/mcp_server/` is an empty stub; it is injected at boot from `MCP_SRC_ROOT` (the vendored `mcp_tools/`) by `desktop_env.py::_inject_mcp_files`, then started on :9292.

So you do **not** download the MCP tools — they ship in-repo and inject at runtime. You only need the (lightly patched) VM disk.

**What the ~13 GB image contains** (≈99% is the stock OSWorld VM):

| Layer | Contents | Origin |
|-------|----------|--------|
| OS + apps | Ubuntu desktop; LibreOffice (calc/impress/writer), GIMP, VLC, Thunderbird, VS Code, Chrome; node/npm (nvm) + python-UNO | **OSWorld** |
| control | `osworld.service` :5000 (screenshot / pyautogui) + task fixtures / preset accounts | **OSWorld** |
| patch | `pip install fastmcp` (the MCP runtime dep) | this repo |
| — | `/home/user/mcp_server/` = empty stub (MCP code injected at boot) | runtime |

Two ways to get the disk:

1. **Build it** (recommended; no large download): take the stock OSWorld `Ubuntu.qcow2`, `pip install fastmcp`, snapshot — a few minutes. See [`scripts/build_mcp_vm.py`](../scripts/build_mcp_vm.py).
2. **Download** a prebuilt copy (GitHub Release attachment, ~13 GB, split archive
   + checksum). Note this is essentially the **OSWorld VM** (its OS/apps/fixtures) plus `fastmcp`, so it is redistributed under **OSWorld's terms** — prefer option 1 (fetch OSWorld's own image, apply the tiny patch) when in doubt.

Runtime copies it to fast tmpfs for boot speed:
```bash
cp /path/to/Ubuntu-MCP.qcow2 /dev/shm/
# scripts default to --path_to_vm /dev/shm/Ubuntu-MCP.qcow2
```

### MCP tools

Running `--action_space hybrid` needs an MCP server + tool implementations inside the VM. **This ships in-repo** under `mcp_tools/` — no external component required.

**In this repo:**
- `mcp_tools/mcp/mcp_server/` — the FastMCP **server** + executable tool implementations (calc / impress / writer / code / chrome / vlc / os). This is the tool set behind the paper's numbers (derived from OSWorld-MCP; see `mcp_tools/NOTICE`).
- `mcp_tools/mcp/osworld_mcp_client.py` — the client injected into the guest.
- `agents/tool_retriever.py` — BM25 tool retriever (top-18).
- `tools/tools_registry.json` — retrieval corpus (tool names / descriptions / param schemas; mirrors `mcp_tools/.../tools/apis/`).
- `OSWorld-main/desktop_env/desktop_env.py::_inject_mcp_files` — copies the server + client from `MCP_SRC_ROOT` into the guest at boot.

**`MCP_SRC_ROOT`** selects which tool set to inject:
- **default `./mcp_tools`** (set in `configs/env.example.sh`) → the vendored set.
- **`0`** → fall back to a stock OSWorld-MCP checkout (must expose `mcp/mcp_server/` and `mcp/osworld_mcp_client.py`).
- **`/path/to/other`** → any alternative tools dir with the same layout.

Note: the patched VM image also has a server baked in; injection is the runtime override used to swap tool sets without rebuilding the image.

## Hardware

- 8× GPUs, 80 GB-class (Qwen3-VL-8B fits on one GPU → 8 parallel vLLM workers).
- Docker + **`/dev/kvm`** (nested virt) for QEMU.
- **Fast local disk** for `OSWORLD_LOCAL_TEMP`: RL rewrites 0.4–1.8 TB of `.pt` per step — keep it OFF any shared/network filesystem.
- Large `/dev/shm` (tmpfs) for the VM disk copy.

## Known gotchas

- Set `OSWORLD_LOCAL_TEMP` to a local path for any RL run (network-fs write bandwidth otherwise saturates).
- After a crashed run, clear `/dev/shm/vllm_ckpt` before restarting (partial checkpoints hang vLLM boot).
- `resume_step.txt` is not cleared by `start_from_scratch`; archive an old `runs/<proj>/` dir before re-running with the same project name.
