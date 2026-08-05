# Task splits

The split files come in a few families — not many versions of one thing, but **five different roles**. This explains each and why they exist.

The authoritative split JSONs (consumed by the code) live in `OSWorld-main/evaluation_examples/`. The `*.txt` here are provenance id lists; `task_metadata.jsonl` carries per-task tags. `v1_legacy/` holds the id lists of the deprecated v1 split (whole-app train on calc/impress/writer, app-level transfer buckets) — kept only so the early B1/B2 report tables in `baselines/` remain traceable; nothing consumes them.

## TL;DR — which file do I use?

| To… | Use | Wired in |
|-----|-----|----------|
| Reproduce any paper eval number | `test_all_no_internet.json` (309) | default `TEST_META` in `scripts/run_{mcp,puregui}_eval.sh`; `dataset.evaluation` in every experiment yaml |
| Train RL with the paper recipe | `train_v2_signal_v2.json` (74) | `dataset.train` in `outcome_only_rl.yaml` / `context_rl.yaml` |
| Reproduce the dense-bonus run (paper §5.2) | `train_v2_signal_v2_top24.json` (24) | `dataset.train` in `dense_bonus_rl.yaml` |
| Monitor generalization during training | `held_out_v2_eval.json` (48) — judge on the `held_out_v2_movable.json` (18) subset | `dataset.held_out_meta` in every experiment yaml |
| Re-curate the gradient band yourself | start from `train_v2_all_apps.json` (172) | `scripts/curate_trainable_tasks.py` |
| Reproduce the D13 DiD read-out (paper §5.3) | `ctx_degraded_train14.json` (13 usable of 14; see `results/context_rl/PROVENANCE.md`) | `results/context_rl/recompute.py` |
| Smoke-test the harness end-to-end | `smoke_test.json` | `TEST_META=... bash scripts/run_mcp_eval.sh` |

Everything not in this table — `test_all.json`, `test_nogdrive.json`, `held_out_v2_layers.json`, `ctx_degraded_{held6,d20}.json`, `align_smoke_libreoffice23.json`, `smoke_test_5tasks.json`, and the `data/splits/*.txt` id lists — is **provenance / measurement metadata**: kept for traceability (or historical B1/B2 bucket semantics), consumed by no training or eval entry point.

## 1. Final eval — nested filtering (for reproducibility)

`test_all (369) ⊃ test_nogdrive (361) ⊃ test_all_no_internet (309)`

Each level drops tasks that need internet / a Google-Drive account (not reproducible without credentials). **All baselines and eval in the paper use the 309-task `test_all_no_internet.json`** (includes 89 true-OOD chrome/multi_apps tasks never touched during development).

## 2. Training — two splits

| File | Tasks | Role |
|------|------:|------|
| `train_v2_all_apps.json` | 172 | the full v2 train pool (per-task split of 8 apps); source for profiling + curation |
| **`train_v2_signal_v2.json`** | **74** | the **gradient band** used by every RL run |
| `train_v2_signal_v2_top24.json` | 24 | fast-iteration subset of the band (the dense-bonus run, `dense_bonus_rl.yaml`) |

**Why curate 172 → 74 (the "signal band"):** outcome-only GRPO gets **zero gradient** from tasks that are always-fail (p=0) or always-pass (p=1) — no variance within the rollout group. So the RL set is curated to the mixed-outcome band (0.1 < p < 0.9), the only tasks that carry a learning signal. The band **drifts with the policy**, so it is re-profiled periodically (`configs/experiments/profile_train_tempk.yaml`) and re-curated (`scripts/curate_trainable_tasks.py`).

## 3. Held-out — one set, sub-classified for measurement sensitivity

| File | Tasks | Role |
|------|------:|------|
| `held_out_v2_eval.json` | 48 | the held-out set (monitored every few steps) |
| `held_out_v2_layers.json` | 48 (tagged) | dead 25 / movable 18 / solved 5 |
| `held_out_v2_movable.json` | 18 | the **primary read-out** |

**Why the sub-classification:** of the 48 held-out tasks, 25 are always-0 and 5 always-1 — they can't move, so aggregate accuracy hides small changes. The 18 "movable" tasks are the real read-out. Held-out trajectories **never** enter training data.

## 4. Pre-registered ablation subsets

`ctx_degraded_{train14, held6, d20}.json` — the context-compression degradation subsets for the difference-in-differences read-out, pre-registered (frozen after selection) to avoid forking-paths.

## 5. Smoke tests

`smoke_test.json`, `smoke_test_5tasks.json` — a handful of tasks for end-to-end sanity checks.

---

Per-app train/held counts: calc 37/10, impress 37/10, writer 18/5, gimp 20/6, os 19/5, vlc 13/4, vs_code 16/5, thunderbird 12/3. train ∩ held = ∅. `chrome` + `multi_apps` appear only in the 309 eval → the only true OOD.

Regenerate the v2 splits with `scripts/prepare_splits_v2.py`; curate the gradient band with `scripts/curate_trainable_tasks.py`.
