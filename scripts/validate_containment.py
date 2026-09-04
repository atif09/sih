"""
Validates the Containment Window Estimator (Phase 9, Feature 2) two ways:

1. Leave-one-day-out across the 7 non-held-out training days: for each day
   that has attack episodes, build duration stats from the OTHER 6 days and
   check whether that day's own episode durations fall inside the estimated
   range. This is the statistically meaningful check (7 folds), since the
   project only has one true held-out day.
2. The real held-out day (Friday-Afternoon-DDoS) against stats built from
   ALL 7 training days -- the actual PS-relevant generalization test, but
   explicitly disclosed as a sample size of one (a single 21-window DDoS
   episode), not something to over-interpret statistically.

Usage:
    python scripts/validate_containment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from containment.estimator import ground_truth_stage, build_stage_duration_stats
from graph.io import load_day
from features.datasets import CICIDS2017_FILES

WINDOW_SIZE_SECONDS = 60.0
HELD_OUT_DAY = "Friday-Afternoon-DDoS"


def day_episode_durations(snapshots) -> dict[str, list[float]]:
    stages = [ground_truth_stage(s) for s in snapshots]
    out: dict[str, list[float]] = {}
    i = 0
    while i < len(stages):
        if stages[i] == "Benign / No Stage":
            i += 1
            continue
        j = i
        while j < len(stages) and stages[j] == stages[i]:
            j += 1
        out.setdefault(stages[i], []).append((j - i) * WINDOW_SIZE_SECONDS / 60.0)
        i = j
    return out


def main() -> None:
    day_snapshots = {meta.day: load_day(meta.day, "data/processed/cicids2017")[0] for meta in CICIDS2017_FILES}
    training_days = [d for d in day_snapshots if d != HELD_OUT_DAY]

    print(f"{'=' * 78}\nLeave-one-day-out validation across {len(training_days)} training days\n{'=' * 78}")
    hits_typical, hits_full, total = 0, 0, 0
    for held_day in training_days:
        held_durations = day_episode_durations(day_snapshots[held_day])
        if not held_durations:
            print(f"{held_day}: no attack episodes, skipped")
            continue
        stats = build_stage_duration_stats(day_snapshots, WINDOW_SIZE_SECONDS, exclude_days={held_day, HELD_OUT_DAY})
        for stage, actual_durations in held_durations.items():
            stage_stats = stats.get(stage)
            if stage_stats is None:
                print(f"{held_day} [{stage}]: no reference data from other days, skipped")
                continue
            lo_typ, hi_typ = stage_stats.likely_range_minutes
            lo_full, hi_full = stage_stats.full_observed_range_minutes
            for actual in actual_durations:
                total += 1
                in_typical = lo_typ <= actual <= hi_typ
                in_full = lo_full <= actual <= hi_full
                hits_typical += in_typical
                hits_full += in_full
            mean_actual = sum(actual_durations) / len(actual_durations)
            print(f"{held_day} [{stage}]: {len(actual_durations)} actual episode(s), mean={mean_actual:.1f}min | "
                  f"reference (n={stage_stats.n_episodes}) typical=[{lo_typ:.1f},{hi_typ:.1f}] full=[{lo_full:.1f},{hi_full:.1f}]")

    print(f"\nAcross {total} held-out episode instances (leave-one-day-out):")
    print(f"  Fell within the 'typical' (p25-p90) range: {hits_typical}/{total} ({100*hits_typical/total:.0f}%)")
    print(f"  Fell within the full observed range:        {hits_full}/{total} ({100*hits_full/total:.0f}%)")

    print(f"\n{'=' * 78}\nTrue held-out day: {HELD_OUT_DAY} (sample size = 1 episode -- disclosed, not oversold)\n{'=' * 78}")
    real_stats = build_stage_duration_stats(day_snapshots, WINDOW_SIZE_SECONDS, exclude_days={HELD_OUT_DAY})
    held_durations = day_episode_durations(day_snapshots[HELD_OUT_DAY])
    for stage, actual_durations in held_durations.items():
        actual = actual_durations[0]
        stage_stats = real_stats.get(stage)
        if stage_stats is None:
            print(f"[{stage}]: no reference data available")
            continue
        lo_typ, hi_typ = stage_stats.likely_range_minutes
        lo_full, hi_full = stage_stats.full_observed_range_minutes
        print(f"True duration: {actual:.1f} min")
        print(f"Estimated typical range (p25-p90, n={stage_stats.n_episodes}): [{lo_typ:.1f}, {hi_typ:.1f}] min "
              f"-> {'CONTAINS' if lo_typ <= actual <= hi_typ else 'MISSES'} the true value")
        print(f"Estimated full observed range: [{lo_full:.1f}, {hi_full:.1f}] min "
              f"-> {'CONTAINS' if lo_full <= actual <= hi_full else 'MISSES'} the true value")
        print(f"Point estimate (median) would have said {stage_stats.median_minutes:.1f} min "
              f"(error vs. true: {abs(stage_stats.median_minutes - actual):.1f} min)")


if __name__ == "__main__":
    main()
