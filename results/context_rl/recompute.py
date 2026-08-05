#!/usr/bin/env python3
"""Re-aggregate context-RL accuracy + input-token cost from raw probe trajectories.

One consistent method: per-task success = mean over reps, overall = mean over
tasks. Input cost = sum of meta.total_input_toks over all trajectories.

The raw probe trajectories are not shipped (too large). Regenerate them with the
probe configs pointed at the released checkpoints, then run:

    python results/context_rl/recompute.py /path/to/runs

Produces the table in unified_curve.md. cells.json is the precomputed
official result if you only need the numbers.
"""
import json
import glob
import os
import sys
from collections import defaultdict

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
EX = os.environ.get("EVAL_EXAMPLES", "OSWorld-main/evaluation_examples")


def load_ids(path):
    try:
        d = json.load(open(path))
        return set(x for v in d.values() for x in v) if isinstance(d, dict) else set(d)
    except Exception:
        return set()


TRAIN74 = load_ids(f"{EX}/train_v2_signal_v2.json")
HELD48 = load_ids(f"{EX}/held_out_v2_eval.json")

# operating point -> expected probe-output dir name(s) under runs/ or runs/archive/.
# Regenerate these by running the probe configs against the released checkpoints;
# the first matching dir found wins.
PLAN = {
    "base-img4": ["probe_base_img4"],
    "base-img2": ["probe_base_img2"],
    "step30-img4": ["probe_context_rl_step30_img4"],
    "step30-img2": ["probe_context_rl_step30_img2"],
    "step40-img4": ["probe_context_rl_step40_img4"],
    "step40-img2": ["probe_context_rl_step40_img2"],
    "step50-img4": ["probe_context_rl_step50_img4"],
}


def resolve(cands):
    best, best_n = None, -1
    for c in cands:
        for base in (RUNS, os.path.join(RUNS, "archive")):
            p = os.path.join(base, c)
            if os.path.isdir(p):
                n = len(glob.glob(f"{p}/rollouts/**/trajectory.json", recursive=True))
                if n > best_n:
                    best, best_n = p, n
    return best


def aggregate(root):
    by_task = defaultdict(list)
    tot_in = 0
    for p in glob.glob(f"{root}/rollouts/**/trajectory.json", recursive=True):
        tid = p.split("/")[-3]
        try:
            m = json.load(open(p)).get("meta", {})
        except Exception:
            continue
        r = m.get("result")
        try:
            ok = 1.0 if float(r) >= 1.0 else 0.0
        except (TypeError, ValueError):
            ok = 1.0 if r is True else 0.0
        by_task[tid].append(ok)
        tot_in += m.get("total_input_toks", 0) or 0

    def rate(ids=None):
        ts = [t for t in by_task if ids is None or t in ids]
        return (sum(sum(by_task[t]) / len(by_task[t]) for t in ts) / len(ts) * 100) if ts else float("nan")

    return dict(overall=rate(), train74=rate(TRAIN74), held48=rate(HELD48),
                n_tasks=len(by_task), tot_in=tot_in)


def main():
    print(f"{'operating point':16} {'overall':>8} {'train74':>8} {'held48':>8} {'input':>9} {'n':>4}")
    for op, cands in PLAN.items():
        root = resolve(cands)
        if not root:
            print(f"{op:16}  (no trajectories found under {RUNS})")
            continue
        r = aggregate(root)
        print(f"{op:16} {r['overall']:7.1f}% {r['train74']:7.1f}% {r['held48']:7.1f}% "
              f"{r['tot_in']/1e6:7.1f}M {r['n_tasks']:>4}")


if __name__ == "__main__":
    main()
