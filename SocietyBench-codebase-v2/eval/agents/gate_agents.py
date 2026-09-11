#!/usr/bin/env python3
"""M3 agent evaluation coverage gate (hard code-level gate to keep "full coverage" from being silently downscaled).

Given workspace + run_id + the expected frameworks x bases x axes x points, check each cell:
  - output file exists;
  - score_100 is not None;
  - (brier) parse_fail ratio <= threshold;
  - each config x axis has an aggregated.json.
Any gap -> list it and exit with a non-zero code.

Usage:
  python gate_agents.py --workspace <ws> --run-id <id> \
      --frameworks langgraph,autogen,mirofish \
      --bases doubao-seed-2-0-pro-260215,qwen3.5-plus-2026-02-15 --axes brier,time
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


def model_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--frameworks", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--axes", default="brier,time")
    ap.add_argument("--points", default="all")
    ap.add_argument("--max-parse-fail", type=float, default=0.02)
    args = ap.parse_args()

    ws = pathlib.Path(args.workspace)
    run = ws / "results" / f"run_{args.run_id}"
    pts = [f.stem.replace("_questionbank", "")
           for f in sorted((ws / "questionbank").glob("P*_questionbank.json"))]
    if args.points != "all":
        want = set(args.points.split(","))
        pts = [p for p in pts if p in want]

    fws = [x.strip() for x in args.frameworks.split(",") if x.strip()]
    bases = [x.strip() for x in args.bases.split(",") if x.strip()]
    axes = [x.strip() for x in args.axes.split(",") if x.strip()]

    gaps = []
    for fw in fws:
        for b in bases:
            slug = f"{fw}__{model_slug(b)}"
            for axis in axes:
                for pid in pts:
                    f = run / axis / slug / f"{pid}_{axis}.json"
                    if not f.exists():
                        gaps.append(f"缺文件 {axis}/{slug}/{pid}")
                        continue
                    try:
                        d = json.loads(f.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        gaps.append(f"坏JSON {axis}/{slug}/{pid}")
                        continue
                    if d.get("score_100") is None and d.get("events_total", 1) != 0:
                        gaps.append(f"score=None {axis}/{slug}/{pid}")
                    if d.get("agent_fallback"):
                        gaps.append(f"agent_fallback {axis}/{slug}/{pid}")
                    if axis == "brier":
                        n = d.get("n", 0) or 0
                        pf = d.get("parse_fail_count", 0) or 0
                        if n and pf / n > args.max_parse_fail:
                            gaps.append(f"parse_fail过高 {pf}/{n} {slug}/{pid}")
                if not (run / axis / slug / "aggregated.json").exists():
                    gaps.append(f"缺 aggregated {axis}/{slug}")

    total = len(fws) * len(bases) * len(axes) * len(pts)
    print(f"期望覆盖:{len(fws)}框架 × {len(bases)}底座 × {len(axes)}轴 × {len(pts)}点 = {total} 格")
    if gaps:
        print(f"❌ 覆盖门禁未通过,{len(gaps)} 处缺漏:")
        for g in gaps[:60]:
            print("  -", g)
        if len(gaps) > 60:
            print(f"  …… 还有 {len(gaps) - 60} 处")
        sys.exit(1)
    print("✅ 覆盖门禁通过:全部格子有结果、score 非空、parse_fail 达标、aggregated 齐全。")


if __name__ == "__main__":
    main()
