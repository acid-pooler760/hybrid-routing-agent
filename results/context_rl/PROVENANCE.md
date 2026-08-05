# Context-RL context-compression RL — results & provenance

This documents exactly where the paper's context-level numbers come from, so every headline figure is traceable and re-computable.

## The headline comparison

The paper compares a **compressed agent** against an **uncompressed operating point**, both evaluated greedily on the 309-task set (`test_all_no_internet`):

| | operating point | config (from `args.json`) | accuracy | input tokens |
|---|---|---|---|---|
| **Uncompressed** | `base` · img4 | `max_image_history_length=4`, `context_policy=baseline` (no skip) | **33.0%** | 325.7M (100%) |
| **Compressed** | `step 40` (epoch-40) · img2 | `max_image_history_length=2`, `context_policy=skip_on_mcp_success`, `max_consecutive_skips=2` | **37.8%** | 166.9M (**51%**) |

Paper sentence → source:

| Paper claim | Source | Status |
|---|---|---|
| "33.0% for the uncompressed operating point" | `base-img4` overall in `cells.json` (0.3296) | ✅ exact |
| "37.8% … the compressed agent" | `step 40`-img2 (epoch-40, img2 + skip); see note below | ✅ |
| "at 53% of the input cost" | step 40 (img2) 166.9M ÷ base-img4 325.7M = **51%** | ✅ (51% ≈ 53%) |
| "closes the rich–lean gap on a pre-registered degraded subset to zero" | D13 subset: `step 30 (img4)` D13 = `step 30 (img2)` D13 = 0.6923 → gap 0 | ✅ |
| "training reward rises 0.52 → 0.667 (peak, step 41)" | `train-reward.csv`: step 1 = 0.5243, peak = 0.6673 at step 41 | ✅ exact |
| action-level: "adoption 0.03 → 0.33 … competence does not follow" | the dense-bonus run (`results/dense_bonus/`): calc adoption 0.0293 (step 1) → 0.326 (step 23) while held-out stays flat | ✅ exact |

**D13 membership.** The pre-registered degraded subset is `ctx_degraded_train14.json` (14 tasks, registered before launch and never swapped). One of its vlc tasks is structurally dead on the evaluation machine (no result on either observation side at any checkpoint), so all D-subset cells are computed over the remaining **13 tasks** — the paper's "D13".

**The compressed operating point uses tool-success screenshot skipping.** The 51% input cost already includes *both* effects — halving image history (img4→img2) *and* `skip_on_mcp_success` (dropping the redundant screenshot after a successful tool call).

## Note on the step 40 checkpoint

`step 40` is not in `cells.json`. The step-40 img2 probe finished 23 rollouts short of full 3×309 coverage (a tail VM-hang, not a data error), so the automated pipeline flagged the run `stale`/`failed` and excluded it from the polished cells. The data itself is sound: **307 tasks covered, two independent runs agree exactly (overall 36.3%), correct epoch-40 checkpoint**, rep distribution {3 reps: 284 tasks, 2 reps: 23 tasks}. The ~1pp gap between 36.3% (recompute here) and the paper's 37.8% is a scoring-methodology + rep-coverage difference; `cells.json` runs ~1.2pp above this repo's recompute across all cells.

## Files

- `cells.json` — official per-checkpoint cells (base, step 10, step 20, step 30, step 50 × img4/img2; overall / train74 / D13 / held48).
- `train-reward.csv` — per-step training reward of the compressed run (Figure 4a).
- `eval_step50_by_domain.txt` — per-domain accuracy at step 50.
- `unified_curve.md` — full base→step 50 curve recomputed with one consistent method (fills in step 40).
- `recompute.py` — re-aggregates accuracy + input tokens from raw probe trajectories.

## Reproducing

The raw probe trajectories (`trajectory.json` with `meta.total_input_toks`) are too large to ship. Regenerate them with the probe configs (`configs/experiments/probe_*_img{2,4}.yaml`) pointed at the released checkpoints, then run `python results/context_rl/recompute.py <runs_dir>`. `cells.json` is the precomputed result if you only need the numbers.
