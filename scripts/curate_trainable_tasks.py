#!/usr/bin/env python3
"""Curate a "signal-dense" training meta from per-task rollout outcomes.

Motivation: an early diagnostic showed outcome-only RL is signal-starved —
`n_tasks_kept` median 6, because so many train tasks are always-fail
(thunderbird, most calc) or always-pass (easy gimp). Those groups carry zero
within-group variance → z-score 0 → no gradient. This script drops them and
keeps only tasks with *mixed* empirical pass rate (lo < rate < hi), so nearly
every sampled task contributes a real gradient.

Input: one or more rollout-results json files in the standard rollout dump
format (a list of {domain, example, trajctory_list:[{result, ...}]}), e.g.
  - temp=1.0 profiling sweep dump  (PREFERRED — matches training distribution)
  - runs/<exp>/results/eval_rollout_step_N.json  (greedy temp=0 BOOTSTRAP;
    greedy under-counts hard-but-occasionally-solvable tasks, so it may drop
    "golden hard" tasks. Pass --from-eval-json and treat the result as
    provisional; write the limitation into the run notes.)

Output: an OSWorld meta json {domain: [example_id, ...]} that is a subset of
--base-meta, ready to drop into dataset.train.environment_data_dir.

The curated set is a subset of train_v2_all_apps, so it stays ⊆
test_all_no_internet (no-internet compliance is automatic).
"""
import argparse
import json
import os
from collections import defaultdict


def load_outcomes(paths):
    """Aggregate per-(domain, example) outcomes across all input files.

    Returns dict[(domain, example)] -> {"succ": int, "total": int}.
    A trajectory's `result` field is the binary outcome (0 / 1).
    """
    agg = defaultdict(lambda: {"succ": 0, "total": 0})
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            dom = entry["domain"]
            ex = entry["example"]
            for traj in entry.get("trajctory_list", []):
                res = traj.get("result", 0)
                try:
                    ok = float(res) > 0.0
                except (TypeError, ValueError):
                    ok = bool(res)
                agg[(dom, ex)]["total"] += 1
                agg[(dom, ex)]["succ"] += 1 if ok else 0
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-rollout-json", nargs="*", default=[],
                    help="temp=1.0 rollout dump(s) (preferred source)")
    ap.add_argument("--from-eval-json", nargs="*", default=[],
                    help="greedy eval dump(s) (bootstrap source; under-counts hard tasks)")
    ap.add_argument("--base-meta",
                    default="OSWorld-main/evaluation_examples/train_v2_all_apps.json",
                    help="full train meta to subset")
    ap.add_argument("--out",
                    default="OSWorld-main/evaluation_examples/train_v2_signal_v2.json")
    ap.add_argument("--lo", type=float, default=0.1,
                    help="drop tasks with pass rate <= lo (always-fail)")
    ap.add_argument("--hi", type=float, default=0.9,
                    help="drop tasks with pass rate >= hi (always-pass)")
    ap.add_argument("--min-rollouts", type=int, default=2,
                    help="tasks observed fewer times than this are KEPT (insufficient evidence to drop)")
    ap.add_argument("--min-kept", type=int, default=20,
                    help="warn if fewer than this many tasks survive")
    args = ap.parse_args()

    sources = list(args.from_rollout_json) + list(args.from_eval_json)
    if not sources:
        ap.error("provide at least one of --from-rollout-json / --from-eval-json")
    if args.from_eval_json and not args.from_rollout_json:
        print("[WARN] greedy-only source: hard-but-occasionally-solvable tasks may "
              "be mis-dropped as dead. Result is PROVISIONAL.\n")

    agg = load_outcomes(sources)
    with open(args.base_meta, "r", encoding="utf-8") as f:
        base = json.load(f)

    out = {}
    # per-app tallies: total / kept / dead / saturated / kept-insufficient
    tally = defaultdict(lambda: {"total": 0, "kept": 0, "dead": 0, "sat": 0, "insuf": 0})
    rows = []  # (domain, ex, total, succ, rate, decision)

    for dom, ids in base.items():
        for ex in ids:
            tally[dom]["total"] += 1
            st = agg.get((dom, ex))
            if st is None or st["total"] < args.min_rollouts:
                total = 0 if st is None else st["total"]
                succ = 0 if st is None else st["succ"]
                decision = "keep-insuf"
                tally[dom]["insuf"] += 1
                tally[dom]["kept"] += 1
                out.setdefault(dom, []).append(ex)
                rows.append((dom, ex, total, succ, None, decision))
                continue
            rate = st["succ"] / st["total"]
            if rate <= args.lo:
                decision, key = "drop-dead", "dead"
            elif rate >= args.hi:
                decision, key = "drop-sat", "sat"
            else:
                decision, key = "keep", "kept"
            tally[dom][key if key in ("dead", "sat") else "kept"] += 1
            if decision == "keep":
                out.setdefault(dom, []).append(ex)
            rows.append((dom, ex, st["total"], st["succ"], rate, decision))

    # preserve base domain ordering, drop now-empty domains
    out = {d: out[d] for d in base if d in out and out[d]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- report ----
    total_kept = sum(len(v) for v in out.values())
    total_base = sum(len(v) for v in base.values())
    print(f"sources: {sources}")
    print(f"filter : keep {args.lo} < rate < {args.hi}  "
          f"(min_rollouts={args.min_rollouts})\n")
    print(f"{'app':<22}{'base':>5}{'kept':>6}{'dead':>6}{'sat':>5}{'insuf':>7}")
    print("-" * 51)
    for dom in base:
        t = tally[dom]
        print(f"{dom:<22}{t['total']:>5}{t['kept']:>6}{t['dead']:>6}{t['sat']:>5}{t['insuf']:>7}")
    print("-" * 51)
    print(f"{'TOTAL':<22}{total_base:>5}{total_kept:>6}"
          f"{sum(t['dead'] for t in tally.values()):>6}"
          f"{sum(t['sat'] for t in tally.values()):>5}"
          f"{sum(t['insuf'] for t in tally.values()):>7}")
    print(f"\nwrote {total_kept} tasks -> {args.out}")
    if total_kept < args.min_kept:
        print(f"[WARN] only {total_kept} tasks survived (< {args.min_kept}). "
              f"Consider widening --lo/--hi or using a temp=1.0 source.")


if __name__ == "__main__":
    main()
