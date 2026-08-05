# Outcome-Only outcome-only RL — results & provenance

`outcome_only_rl` is the **pure outcome-only RLVR** run (74-task gradient band, 33 steps, symmetric GRPO, no dense tool bonus, no pos-adv-only). It is the "outcome-only" released checkpoint (epoch-30).

## What it shows: behavior is steerable, competence is not

| signal | step 1 | step ~33 | reading |
|---|---|---|---|
| **train ALL** (temp=1.0, 74-task) | 0.58 | **0.71–0.73** | train-side sampled accuracy rises |
| **spreadsheet MCP adoption** (calc, train) | **0.02** | **0.33** | tool-use behavior is learned |
| impress / writer MCP adoption | 0.04 / 0.06 | 0.22 / 0.26 | same trend |
| **held-out accuracy** (temp=0 greedy, 48×3) | 0.34 | 0.30–0.37 (bouncing) | **flat — no competence gain** |

The held-out greedy curve over steps 0→30 is `0.34, 0.28, 0.34, 0.32, 0.35, 0.31, 0.37` — noise around ~0.34, no trend. So outcome-only RL **raises tool adoption ~0.02→0.33 and lifts the temp=1.0 train curve, but does not move held-out competence.** This is the paper's "behavior is steerable; competence is not" — visible within a single run.

> Aggregate/train curves are NOT the learning criterion (in-band sampling noise masquerades as improvement). The competence read-out is the temp=0 held-out curve above and per-task CLIMB counts, which stay flat.

## Note on attribution

The paper phrases the action-level result around the *dense tool bonus* run (paper §5.2) — shipped as `configs/experiments/dense_bonus_rl.yaml` with its training record in [`results/dense_bonus/`](../dense_bonus/README.md). This released outcome-only run reaches the same adoption endpoint (~0.33) with no bonus, and is the cleaner outcome-only baseline; the dense-bonus run is the one that establishes the bonus→adoption causal link on the 24-task subset.

## Files

- `train-reward.csv` — per-step training reward (the rich-observation control curve of paper §5.3 / Figure 4a; paused at step 30 after probing).
- `held-out-eval.jsonl` — per-step held-out (48×3) accuracy + resp_len + frac_hit_cap.
- `rl-results.txt` — per-step train-side accuracy by domain ([TRAIN N]).
- `rollout-stats.jsonl` — per-step rollout stats incl. `domain_mcp_rates` (adoption), outcome rates, termination signals.
- `train-loss.jsonl` — per-step training loss / KL / entropy.

Raw rollout trajectories are not shipped; these per-step summaries are the monitoring record produced during training.
