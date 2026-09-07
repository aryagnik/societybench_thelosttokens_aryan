#!/usr/bin/env python3
"""One-shot M3 evaluation: three agents + coverage gate, sharing one run-id.

**Decided 2026-07-01 ("global optimization"): M3 runs only the three agents by default,
no longer piggybacking the bare baseline.**
The bare Doubao 3B/3F baseline comes from main experiment M1 (run_M1); rerunning it in M3
is pure duplication/waste. The step4 scorecard reads "M1 bare baseline + M3 agents" across
runs for comparison. If you really need the bare line in the same run (old behavior), add
`--with-bare`. All results land in the same results/run_<id>/.

Usage (run with the agents venv python):
  ~/societybench_agents_venv/bin/python code/eval/agents/run_full_m3.py --event smci --run-id M3
  ~/societybench_agents_venv/bin/python code/eval/agents/run_full_m3.py --event us_iran --run-id M3
Smoke (a few points + truncated questions):
  ... --event smci --points P01,P07 --axes brier --mirofish-agents 4 --mirofish-steps 2
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent      # .../code/eval/agents
_EVAL = _HERE.parent                                 # .../code/eval
_ROOT = _EVAL.parent.parent                          # .../societybench

# Canonical finalized path <event>/final/<lang>/ (2026-06 refactor: final/ splits into 中文/ and 英文/
# as two independent test sets; question banks live in final/<lang>/questionbank/, so
# workspace = final/<lang>, decided by --lang)
_EVENTS = {
    "library": ("runs_new/event1_library/final", "library"),
    "trump_tariff": ("runs_new/event2_trump_tariff/final", "trump_tariff"),
    "tiktok": ("runs_new/event3_tiktok/final", "tiktok"),
    "us_iran": ("runs_new/event4_us_iran/final", "us_iran"),
    "smci": ("runs_new/event5_smci/final", "smci"),
}
_BASES_DEFAULT = "doubao-seed-2-0-pro-260215"


def _run(cmd, env=None) -> int:
    print("\n» " + " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot full M3: bare + three agents + gate")
    ap.add_argument("--event", required=True, choices=list(_EVENTS))
    ap.add_argument("--lang", default="中文", choices=["中文", "英文"],
                    help="Which set to test: 中文 (Chinese) for domestic models / 英文 (English) for English models (two independent test sets)")
    ap.add_argument("--bases", default=_BASES_DEFAULT)
    ap.add_argument("--frameworks", default="langgraph,autogen,mirofish")
    ap.add_argument("--axes", default="brier,time")
    ap.add_argument("--points", default="all")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--mirofish-agents", default="8")
    ap.add_argument("--mirofish-steps", default="3")
    ap.add_argument("--bare-max-parallel", default="2",
                    help="Concurrency for the bare-baseline 3B/3F; use low concurrency in production runs to fit the global API quota")
    ap.add_argument("--agent-max-parallel", default="auto",
                    help="Agent session concurrency. 'auto' (default) = all framework x axis units in parallel (after splitting B/F, units = n_frameworks x n_axes); an integer uses that value. Points within a session stay serial (continuous rollout).")
    ap.add_argument("--max-tokens", default="32000")
    ap.add_argument("--thinking-budget", default="24000")
    ap.add_argument("--batch-size", default="999999",
                    help="Brier batch size; 999999 means full-point/unlimited and records metadata as null")
    ap.add_argument("--with-bare", action="store_true",
                    help="Additionally run the bare baseline (3B/3F). Default OFF — the bare Doubao baseline comes from main experiment M1; M3 runs only the three agents to avoid duplication (decided 2026-07-01, global optimization).")
    a = ap.parse_args()

    # Decided 2026-07-01 ("framework parallelism"): by default run all framework x axis units in
    # parallel (after splitting B/F, units = n_frameworks x n_axes) instead of one by one
    # (old default 1 = serial; 3 frameworks couldn't finish in 25 minutes). An integer overrides.
    _n_fw = len([x for x in a.frameworks.split(",") if x.strip()])
    _n_ax = len([x for x in a.axes.split(",") if x.strip()])
    agent_par = str(max(1, _n_fw * _n_ax)) if a.agent_max_parallel == "auto" else a.agent_max_parallel

    rel, ev = _EVENTS[a.event]
    ws = _ROOT / rel / a.lang          # final/中文 or final/英文
    if not (ws / "questionbank").exists():
        raise SystemExit(f"工作区无 questionbank/:{ws}（应为 final/中文 或 final/英文）")
    rid = a.run_id or "full_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    py = sys.executable
    print(f"=== M3 一键全量  event={a.event}  ws={ws}  run_id={rid} ===")

    # 1) Bare-model baseline: OFF by default (bare Doubao baseline comes from main experiment M1; M3 avoids duplication); only run with --with-bare
    if a.with_bare:
        if "brier" in a.axes:
            _run([py, _EVAL / "predict_step3B_brier.py", "--workspace", ws, "--event-name", ev,
                  "--models", a.bases, "--points", a.points, "--run-id", rid,
                  "--max-parallel", a.bare_max_parallel,
                  "--max-tokens", a.max_tokens,
                  "--thinking-budget", a.thinking_budget,
                  "--batch-size", a.batch_size])
        if "time" in a.axes:
            _run([py, _EVAL / "predict_step3F_time.py", "--workspace", ws, "--event-name", ev,
                  "--models", a.bases, "--points", a.points, "--run-id", rid,
                  "--runs-per-point", "1",
                  "--max-parallel", a.bare_max_parallel,
                  "--max-tokens", a.max_tokens,
                  "--thinking-budget", a.thinking_budget])

    # 2) Three agents (the MiroFish subprocess uses the oasis venv; society size is tunable)
    env = dict(os.environ)
    env["MIROFISH_AGENTS"] = a.mirofish_agents
    env["MIROFISH_STEPS"] = a.mirofish_steps
    env["AGENT_MAX_TOKENS"] = a.max_tokens
    env["AGENT_THINKING_BUDGET"] = a.thinking_budget
    _run([py, _HERE / "run_agents.py", "--workspace", ws, "--event-name", ev,
          "--frameworks", a.frameworks, "--bases", a.bases,
          "--axes", a.axes, "--points", a.points, "--run-id", rid,
          "--runs-per-point", "1",
          "--max-parallel", agent_par,
          "--max-tokens", a.max_tokens,
          "--thinking-budget", a.thinking_budget,
          "--batch-size", a.batch_size], env=env)

    # 3) Coverage gate (agent configs; missing cells -> non-zero exit)
    rc = _run([py, _HERE / "gate_agents.py", "--workspace", ws, "--run-id", rid,
               "--frameworks", a.frameworks, "--bases", a.bases,
               "--axes", a.axes, "--points", a.points])

    print(f"\n=== 完成。结果在 {ws}/results/run_{rid}/  (门禁退出码={rc}) ===")
    sys.exit(rc)


if __name__ == "__main__":
    main()
