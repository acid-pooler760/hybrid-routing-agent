# Reproducing the paper

Paper: **[Screenshots or Tools? Eliciting Tool Use and Managing Multimodal Context in Hybrid GUI–MCP Computer-Use Agents](https://arxiv.org/abs/2608.03327)** ([arXiv:2608.03327](https://arxiv.org/abs/2608.03327)).

This maps each paper claim to a command and states the measurement caveats. **Read the caveats** — several results are negative or narrowly scoped, and the aggregate curves are explicitly *not* the learning criterion.

All eval uses `test_all_no_internet.json` (309 tasks), temp=0, max_steps=50, 4-image history (`img4`), unless noted.

## Measurement discipline (applies everywhere)

- **Aggregate `[TRAIN]` / held-out curves are NOT a learning signal** — in-band sampling noise masquerades as improvement. Judge with **per-task classification**: number of CLIMBs (persistent fail→pass flips) + accuracy on the *movable* subset.
- **Held-out 48** is pre-classified into `dead 25 / movable 18 / solved 5` (`held_out_v2_layers.json`); read out on the **movable 18** only. Aggregate metrics cannot resolve ≤5pp.
- Single-point σ: held-out 48×3 ≈ ±3.9pp; train-74×1 ≈ ±5.8pp. Require two consecutive same-direction probes.
- Three `outcome` denominators are NOT interchangeable: `rollout/outcome_rate` (step-weighted) ≠ `outcome_rate_traj` (post-filter, primary) ≠ `[TRAIN] ALL`.

## §A — Inference characterization (B1 / B2)

```bash
bash scripts/run_puregui_eval.sh      # B1: pure GUI (pyautogui)
bash scripts/run_mcp_eval.sh          # B2: GUI + MCP (thinking model)
MODEL_TYPE=instruct bash scripts/run_mcp_eval.sh   # B2-Instruct
python OSWorld-main/show_result.py --result_dir baselines/b2_mcp_thinking
```

**Claim: MCP injection helps Thinking (+4.0pp) but hurts Instruct (−5.9pp), both beyond 2 SE (5-run means).** The headline table is in the [README](README.md); per-run numbers are in [`baselines/REPEATS.md`](baselines/REPEATS.md) (the best single Thinking run reaches 37.9%, +7.5pp over its paired GUI-only run).

### Inference context-policy sweep

The observation rule is fully controllable at inference via env vars on `run_mcp_eval.sh` — no retraining needed. Results auto-land in `baselines/b2_mcp_<model>[_<policy>][_img<N>]` and are summarised in [`baselines/COMPARISON.md`](baselines/COMPARISON.md).

```bash
#   CONTEXT_POLICY:           baseline | skip_on_mcp_success | skip_on_no_change  (default baseline)
#   MAX_IMAGE_HISTORY_LENGTH: 4 (default) | 2        # img4 vs img2
#   MAX_CONSECUTIVE_SKIPS:    2 (default)            # for skip_on_no_change

bash scripts/run_mcp_eval.sh                                              # B2: img4, no skip
MAX_IMAGE_HISTORY_LENGTH=2 bash scripts/run_mcp_eval.sh                   # img2 (halved history)
CONTEXT_POLICY=skip_on_mcp_success bash scripts/run_mcp_eval.sh          # skip screenshot after a successful tool call
CONTEXT_POLICY=skip_on_no_change  bash scripts/run_mcp_eval.sh           # skip when the screen did not change
CONTEXT_POLICY=skip_on_mcp_success MAX_IMAGE_HISTORY_LENGTH=2 \
  bash scripts/run_mcp_eval.sh                                           # the compressed inference config (matches context_rl training)
```

The last line is exactly the observation rule the **context-RL** checkpoint was trained under (`context_rl.yaml`); the ~−3.9pp it costs at inference is what that run recovers (see §B).

**Caveat — most of the MCP gain is a prompt effect, not a tool-execution effect** (decomposition measured on the original single-run pair, where the gain was +7.5pp). Pairing B2 vs B1 per task: tasks where MCP was *never called* (252) still show B2 +8.3pp; tasks where MCP *was called* (56) show only +3.6pp, and calc-with-MCP is −10pp. Tool `exec_ok` is 98–100% (infra is fine); failures are semantic (e.g. `find_and_replace` returns `success:true` with "0 replacements"). Net per-task value of calling a tool ≈ 0 at base skill level. So the adoption gap (tools present but unused) and a **competence** gap ("can't use the tool correctly") coexist — and closing adoption alone does not move accuracy, which is exactly the dense-bonus result below.

## §B — RL results

Run an experiment (self-documenting YAML headers hold hypotheses + red lines):
```bash
OSWORLD_LOCAL_TEMP="$OSWORLD_LOCAL_TEMP" TQDM_DISABLE=1 \
  nohup python osworld_rl.py config=configs/experiments/<run>.yaml > logs/<run>.log 2>&1 &
```

| Claim | Config | Read-out |
|-------|--------|----------|
| **Dense tool bonus moves adoption, not competence** (paper §5.2): a post-normalization bonus (λ_mcp=0.1) raises spreadsheet adoption **0.03→0.33** within 23 steps on the 24-task subset, carries into greedy (**0.02→0.29**), but held-out accuracy stays at base — behavior is steerable, competence is not; bottleneck = tool-call semantics. | `dense_bonus_rl.yaml` | adoption vs held-out acc ([results/dense_bonus](results/dense_bonus/README.md)) |
| **Outcome-only can't learn competence** (four compounding factors: 1-bit/traj signal ÷T, missing credit assignment, exploration wall, data scale). The pure outcome-only run also drifts to the same adoption endpoint (~0.33) while held-out stays flat. | `outcome_only_rl.yaml` | held/greedy acc flat ([results/outcome_only](results/outcome_only/README.md)) |
| **Context-compression RL** (train=eval consistent observation rule): skipping the redundant post-tool screenshot + halving image history cuts input tokens ~⅓ at a small accuracy cost; retraining under the same rule removes that cost. Compressed agent reaches **37.8% vs 33.0%** (uncompressed operating point) at **53% of the input cost**, and closes the rich–lean gap on a pre-registered degraded subset to **zero**. In-distribution only. | `context_rl.yaml` | operating-point acc, input-token cost, degraded-subset gap ([results/context_rl](results/context_rl/PROVENANCE.md)) |

> **Note on the numbers.** The paper reports an operating-point comparison (37.8 vs 33.0 @ 53% input cost). An intermediate step-30 probe is a different snapshot: recomputed from its surviving trajectories, input tokens are 189.0M vs 309.8M (**−39%**) and Peak-Ctx p95 is 7093 vs 11358 (**−38%**), with accuracy 33.6% (img2) vs 34.0% (img4). See `results/context_rl/` for the full curve and the recompute script.

> **Released checkpoints.** You can evaluate without retraining: the outcome-only and context-RL policies are on the HuggingFace Hub — see the README. `MODEL=<downloaded_ckpt_dir> bash scripts/run_mcp_eval.sh`.

**Other ablations discussed in the paper** (stepwise vs groupwise normalization, format reward, the rest of the tool-bonus 2×2, and the positive-advantage-only collapse) were hyperparameter-sweep runs; their configs are not shipped here (dev artifacts), but the findings are reported in §B.

**Probes / profiles** (decision-point measurement recipes):
```bash
# train-74 greedy anchor / gap curve
python osworld_rl.py config=configs/experiments/probe_train_greedy.yaml
# held-out 3-layer classification
python osworld_rl.py config=configs/experiments/profile_heldout_tempk.yaml
```
Probe recipe = `total_step=0 + step0_held_out_eval=true` + `policy_model` → a checkpoint dir (`ckpt/epoch-N-policy`, the real snapshot — not `ckpt_history/`).

**Caveat.** The context-RL win is strictly *in-distribution behavior adaptation*; it is not evidence of improved generalization. Negative/flat RL results are the point, not a failure to reproduce — they bound what outcome-only RLVR does on an 8B model.

## Pre-run self-check

Before trusting any RL run, confirm: `[DS-CHECK] global_steps` increments per window, `[CKPT-SYNC]` appears, first window has `clip=0 / ratio=1.0`.
