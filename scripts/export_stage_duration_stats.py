"""Exports the Containment Window Estimator's episode-duration statistics to
checkpoints/phase3/stage_duration_stats.json so deployment doesn't need the
214MB processed day caches. Run locally (where data/processed exists):

    python scripts/export_stage_duration_stats.py
"""

import json
from pathlib import Path

import yaml

from containment.estimator import build_stage_duration_stats
from models.dataset import load_all_days

CHECKPOINT_DIR = Path("checkpoints/phase3")

run_config = yaml.safe_load(open(CHECKPOINT_DIR / "run_config.yaml"))
window_size = yaml.safe_load(open("configs/pipeline.yaml"))["feature_extraction"]["window_size_seconds"]
stats = build_stage_duration_stats(load_all_days(run_config["data"]["processed_dir"]), window_size)
out = {stage: s.durations_minutes for stage, s in stats.items()}
(CHECKPOINT_DIR / "stage_duration_stats.json").write_text(json.dumps(out))
print({stage: len(d) for stage, d in out.items()})
