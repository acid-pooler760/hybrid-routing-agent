"""
prepare_splits_v2.py

New split: train on ALL apps except multi_apps/chrome, OOD eval on multi_apps+chrome.
Source: test_all_no_internet.json (309 tasks, no internet).
Within each train app, 80% train / 20% held-out (seed=42).

Outputs:
  data/splits/train_v2_task_ids.txt
  data/splits/held_out_v2_task_ids.txt
  data/splits/ood_v2_multiapps_task_ids.txt
  OSWorld-main/evaluation_examples/train_v2_all_apps.json    (for environment_data_dir)
  OSWorld-main/evaluation_examples/held_out_v2_eval.json     (for run_held_out_eval)

Existing v1 split files are NOT modified.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OSWORLD_DIR = REPO_ROOT / "OSWorld-main" / "evaluation_examples"
OUTPUT_DIR = REPO_ROOT / "data" / "splits"

SOURCE_JSON = OSWORLD_DIR / "test_all_no_internet.json"

OOD_APPS = {"multi_apps", "chrome"}

SEED = 42
TRAIN_RATIO = 0.8


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    train_by_domain: dict[str, list[str]] = defaultdict(list)
    held_out_by_domain: dict[str, list[str]] = defaultdict(list)
    ood_by_domain: dict[str, list[str]] = defaultdict(list)

    rng = random.Random(SEED)

    for domain, task_ids in sorted(all_tasks.items()):
        if domain in OOD_APPS:
            ood_by_domain[domain] = list(task_ids)
            continue

        ids = list(task_ids)
        rng.shuffle(ids)
        split_idx = int(len(ids) * TRAIN_RATIO)
        train_by_domain[domain] = ids[:split_idx]
        held_out_by_domain[domain] = ids[split_idx:]

    train_ids = [tid for ids in train_by_domain.values() for tid in ids]
    held_out_ids = [tid for ids in held_out_by_domain.values() for tid in ids]
    ood_ids = [tid for ids in ood_by_domain.values() for tid in ids]

    # --- Write task ID files ---
    def write_ids(filename: str, ids: list[str]):
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for tid in sorted(ids):
                f.write(tid + "\n")
        print(f"  {filename}: {len(ids)} tasks")

    print("Writing v2 split files:")
    write_ids("train_v2_task_ids.txt", train_ids)
    write_ids("held_out_v2_task_ids.txt", held_out_ids)
    write_ids("ood_v2_multiapps_task_ids.txt", ood_ids)

    # --- Write training JSON (for environment_data_dir) ---
    train_json_path = OSWORLD_DIR / "train_v2_all_apps.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(dict(train_by_domain), f, indent=2)
    print(f"  train_v2_all_apps.json: {len(train_ids)} tasks -> {train_json_path}")

    # --- Write held-out eval JSON (for run_held_out_eval) ---
    held_out_json_path = OSWORLD_DIR / "held_out_v2_eval.json"
    with open(held_out_json_path, "w", encoding="utf-8") as f:
        json.dump(dict(held_out_by_domain), f, indent=2)
    print(f"  held_out_v2_eval.json: {len(held_out_ids)} tasks -> {held_out_json_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"V2 Split Summary (source: test_all_no_internet.json, {sum(len(v) for v in all_tasks.values())} tasks)")
    print(f"{'='*60}")
    print(f"{'App':<25} {'Train':>6} {'Held-out':>9} {'OOD':>5}")
    print(f"{'-'*25} {'-'*6} {'-'*9} {'-'*5}")
    all_domains = sorted(set(list(train_by_domain.keys()) + list(ood_by_domain.keys())))
    for domain in all_domains:
        t = len(train_by_domain.get(domain, []))
        h = len(held_out_by_domain.get(domain, []))
        o = len(ood_by_domain.get(domain, []))
        print(f"  {domain:<23} {t:>6} {h:>9} {o:>5}")
    print(f"{'-'*25} {'-'*6} {'-'*9} {'-'*5}")
    print(f"  {'TOTAL':<23} {len(train_ids):>6} {len(held_out_ids):>9} {len(ood_ids):>5}")
    print(f"\nGrand total: {len(train_ids) + len(held_out_ids) + len(ood_ids)}")


if __name__ == "__main__":
    main()
