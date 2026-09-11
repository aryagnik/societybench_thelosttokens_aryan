#!/usr/bin/env python3
"""run-pipeline-parallel (1:1 of skills/pipeline-run/SKILL.md, predict portion).

Run predict-step3B-brier and predict-step3F-time in parallel for the given
models, then aggregate via predict-step4-scorecard.

Usage:
    python run_pipeline_parallel.py \\
        --workspace <path> \\
        --event-name "<event>" \\
        --models "model1,model2,..." \\
        [--points all|P02,P15,...] \\
        [--pipeline-config <path>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from common import load_pipeline_config

THIS_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> int:
    print(f"\n========== [{name}] {' '.join(cmd)} ==========")
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="Run brier + time evaluation in parallel, then aggregate")
    p.add_argument("--workspace", required=True)
    p.add_argument("--event-name", required=True)
    p.add_argument("--models", default=None,
                   help="comma-separated; defaults to pipeline_config.evaluation_models")
    p.add_argument("--points", default="all")
    p.add_argument("--run-id", default=None,
                   help="Reuse a shared run_<ts> dir (orchestrator pins this)")
    p.add_argument("--batch-size", type=int, default=999999,
                   help="Brier batch size; 999999 means full-point/unlimited and records metadata as null")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    models = args.models or ",".join(cfg.get("evaluation_models", []))
    if not models:
        raise SystemExit("No models supplied (set --models or pipeline_config.evaluation_models)")

    # Generate ONE shared run id so step3B and step3F write into the same
    # results/run_<ts>/ directory; otherwise step4 only sees one half.
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    py = sys.executable
    common_args = ["--workspace", args.workspace,
                   "--event-name", args.event_name,
                   "--models", models,
                   "--points", args.points,
                   "--run-id", run_id]
    brier_cmd = [py, str(THIS_DIR / "predict_step3B_brier.py")] + common_args + [
        "--batch-size", str(args.batch_size)
    ]
    time_cmd  = [py, str(THIS_DIR / "predict_step3F_time.py")]  + common_args

    results: dict[str, int] = {}
    threads = [
        threading.Thread(target=lambda: results.update({"brier": run_step("3B-brier", brier_cmd)})),
        threading.Thread(target=lambda: results.update({"time":  run_step("3F-time",  time_cmd)})),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if results.get("brier", 1) != 0 or results.get("time", 1) != 0:
        print(json.dumps({"step3_results": results, "status": "step3_failed"}, ensure_ascii=False, indent=2))
        sys.exit(1)

    score_cmd = [py, str(THIS_DIR / "predict_step4_scorecard.py"),
                 "--workspace", args.workspace,
                 "--run-id", f"run_{run_id}"]
    code = run_step("step4-scorecard", score_cmd)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
