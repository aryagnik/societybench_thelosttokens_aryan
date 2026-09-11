#!/usr/bin/env python3
"""media-pipeline (1:1 of skills/media-pipeline/SKILL.md).

Runs the Media timeline chain: step0 → step1 → step2 → step3 → step4 → step5.

Each step checks whether its output already exists and skips (unless --force).
Use --start-step N to skip earlier steps explicitly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent


def run(name: str, cmd: list[str]) -> None:
    print(f"\n========== [{name}] {' '.join(cmd)} ==========")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def lines_in(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    p = argparse.ArgumentParser(description="media-pipeline: full social-media timeline pipeline")
    p.add_argument("input_csv", help="Input CSV from media-crawl (post + comments)")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--start-step", type=int, default=0, help="Skip steps below this index")
    p.add_argument("--stages", default=None, help="Optional stages JSON for step0/step5")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    a = p.parse_args()

    py = sys.executable
    base = ["--event-name", a.event_name]
    if a.model:
        base += ["--model", a.model]
    if a.pipeline_config:
        base += ["--pipeline-config", a.pipeline_config]

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s0 = out / "posts_aggregated.jsonl"
    s1 = out / "media_step1.jsonl"
    s2 = out / "media_step2.jsonl"
    s3 = out / "media_step3.jsonl"
    s4 = out / "media_step4.jsonl"
    s5 = out / "media_timeline_short.md"

    def maybe(idx: int, name: str, target: Path, cmd: list[str]) -> None:
        if a.start_step > idx:
            print(f"[Step {idx}/5] skipped (start-step={a.start_step})")
            return
        if not a.force and target.exists() and target.stat().st_size > 0:
            n = lines_in(target) if target.suffix == ".jsonl" else 0
            print(f"[Step {idx}/5] {name} already exists at {target} ({n} lines) — skipping (use --force to rerun)")
            return
        t = time.time()
        run(name, cmd)
        dur = time.time() - t
        if target.exists():
            n = lines_in(target) if target.suffix == ".jsonl" else target.stat().st_size
            print(f"[Step {idx}/5] {name} done — {n} {'lines' if target.suffix=='.jsonl' else 'bytes'} in {dur:.1f}s")

    # step0 is purely algorithmic (no LLM); takes input_csv directly
    step0_cmd = [py, str(D / "media_step0_aggregate.py"), a.input_csv, str(out)]
    if a.pipeline_config:
        step0_cmd += ["--pipeline-config", a.pipeline_config]
    if a.stages:
        step0_cmd += ["--stages", a.stages]
    maybe(0, "media-step0", s0, step0_cmd)

    maybe(1, "media-step1", s1, [py, str(D / "media_step1_summarize.py"), str(s0), str(out), *base])
    maybe(2, "media-step2", s2, [py, str(D / "media_step2_clean.py"),     str(s1), str(out), *base])
    maybe(3, "media-step3", s3, [py, str(D / "media_step3_dedup.py"),     str(s2), str(out), *base])
    maybe(4, "media-step4", s4, [py, str(D / "media_step4_extract.py"),   str(s3), str(out), *base])

    step5_cmd = [py, str(D / "media_step5_short.py"), str(s4), str(out), *base]
    if a.stages:
        step5_cmd += ["--stages", a.stages]
    maybe(5, "media-step5", s5, step5_cmd)

    print("\n═══════════════════════════════════")
    print(f"  Media Pipeline 完成 — {a.event_name}")
    print("═══════════════════════════════════")
    for label, path in (("step0", s0), ("step1", s1), ("step2", s2), ("step3", s3), ("step4", s4), ("short", s5)):
        if path.exists():
            n = lines_in(path) if path.suffix == ".jsonl" else path.stat().st_size
            print(f"  {label:6} → {path}  ({n} {'lines' if path.suffix=='.jsonl' else 'bytes'})")


if __name__ == "__main__":
    main()
