# Context-RL full curve (single consistent recompute)

All values recomputed from raw probe trajectories with the same per-task method (`recompute.py`): per-task success = mean over reps, overall = mean over tasks. Greedy, 309-task set. **These run ~1.2pp below `cells.json`** (scoring methodology); use them for the *trend* and for the step 40 point (absent from cells), and cite `cells.json` for paper figures.

| step | overall img4 | overall img2 | train74 img2 | held48 img2 | input img4 | input img2 | img2 % of img4 |
|------|---:|---:|---:|---:|---:|---:|---:|
| base | 31.7% | 32.1% | 53.7% | 34.7% | 325.7M | 201.9M | 62% |
| step 10  | 36.4% | 32.3% | 53.9% | 34.0% | 290.0M | 182.9M | 63% |
| step 20  | 33.9% | 34.2% | 63.9% | 36.1% | 314.1M | 191.5M | 61% |
| step 30  | 33.9% | 33.6% | 66.7% | 31.2% | 309.8M | 189.0M | 61% |
| **step 40** | 36.5% | **36.3%** | 70.4% | 33.3% | 286.0M | **166.9M** | **58%** |
| step 50  | 37.5% | 35.6%¹ | 65.0% | 34.1% | 243.3M | ~145M² | ~60% |

¹ step 50 (img2) accuracy from the training run's consolidated `eval_rollout_step_50.json` (297 tasks; no token fields). ² step 50 (img2) input **estimated** (243.3M × observed img2/img4 ratio); no raw trajectory tokens survive for this point.

## Compressed (img2 + skip) vs uncompressed base (img4, no skip)

Input cost relative to `base-img4` (325.7M), i.e. the paper's "input cost":

| compressed point | input | % of base-img4 |
|---|---|---|
| step 40 (img2) | 166.9M | **51%** ← paper's "53%" |
| step 50 (img2) | ~145M² | ~45% |

held48 stays flat (31–36%) across all steps → in-distribution adaptation only, no generalization. train74 (in-distribution) rises strongly. Consistent with the paper's characterization.
