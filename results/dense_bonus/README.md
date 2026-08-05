# Dense tool-bonus RL — results & provenance

`dense_bonus_rl` is the **dense-tool-bonus** run of paper §5.2: outcome-only reward plus a post-normalization MCP tool bonus (λ_mcp = 0.1, constant over the run), trained on the 24-task fast-iteration subset of the gradient band (`train_v2_signal_v2_top24.json`). Config: `configs/experiments/dense_bonus_rl.yaml`.

## What it shows: the tool decision is fully steerable

| signal | step 1 | step 23 | reading |
|---|---|---|---|
| **spreadsheet (calc) MCP adoption** (temp=1.0 rollouts) | **0.029** | **0.326** | the paper's 0.03 → 0.33 |
| greedy calc adoption (temp=0 probes) | 0.02 | 0.29 | learned policy, not sampling noise |
| **held-out accuracy** (48×3, temp=0) | 0.36 | 0.27–0.32 (bouncing) | **flat at base level — no competence gain** |

One post-normalization reward term is enough to change the tool decision by an order of magnitude, and the shift survives into greedy decoding. Held-out accuracy does not follow — this is the paper's **adoption–competence decoupling** (§5.2, Figure 2).

The per-step calc adoption comes from `rollout-stats.jsonl` → `domain_mcp_rates.libreoffice_calc` (extracted to `adoption_calc.csv`); the held-out curve is `held-out-eval.jsonl` (extracted to `heldout_acc.csv`).

## Files

- `adoption_calc.csv` — per-step calc adoption at temp=1.0 (Figure 2, top).
- `heldout_acc.csv` — held-out (48×3) greedy accuracy every 5 steps (Figure 2, bottom).
- `held-out-eval.jsonl` — full held-out probe record (acc + resp_len + frac_hit_cap).
- `rl-results.txt` — per-step train-side accuracy by domain (`[TRAIN N]`).
- `rollout-stats.jsonl` — per-step rollout stats incl. `domain_mcp_rates` (adoption), `n_tool_bonus` / `mcp_bonus_dense` (bonus firing), outcome rates, termination signals.
- `train-loss.jsonl` — per-step training loss / KL / entropy.

Raw rollout trajectories are not shipped; these per-step summaries are the monitoring record produced during training.
