#!/usr/bin/env python3
"""Phase 1: rollout only (strict two-phase design, decided 2026-07-05):
Start one continuous rollout session, and along the timeline P1->P25 only advance and only
save opinion snapshots — **no answering at all**.
Rollout output = per-point opinion snapshots {snapshot_dir}/{cutoff}.json; phase 2 reads them offline to answer.

Usage: python miro_phase1_advance.py --workspace WS --event-name library --base doubao-... --run-id RID
Environment variables (rollout params): MIROFISH_STEPS / MIROFISH_MEM_KEEP (window) / MIROFISH_HEDGE_* / MIROFISH_SNAPSHOT_DIR / MIROFISH_LEDGER ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter import make_session          # noqa: E402
from run_agents import compute_delta      # noqa: E402  reuse the same delta slicing as bare models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--event-name", required=True)
    ap.add_argument("--base", default="doubao-seed-2-0-pro-260215")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--thinking-budget", type=int, default=24000)
    args = ap.parse_args()

    ws = Path(args.workspace)
    qb_dir, ctx_dir = ws / "questionbank", ws / "contexts"
    snap_dir = os.environ.get("MIROFISH_SNAPSHOT_DIR") or str(ws / "mirofish_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    os.environ["MIROFISH_SNAPSHOT_DIR"] = snap_dir  # the fw layer auto-saves a snapshot after advance

    qb_files = sorted(qb_dir.glob("P*_questionbank.json"))
    if not qb_files:
        raise SystemExit(f"无 questionbank/ 于 {ws}")

    # Pre-scan: skip points that already have snapshots (resume from checkpoint, don't re-spend)
    plan = []
    for qbf in qb_files:
        pid = qbf.stem.replace("_questionbank", "")
        try:
            cutoff = json.loads(qbf.read_text(encoding="utf-8")).get("cutoff_date") or ""
        except Exception:
            cutoff = ""
        ctxf = ctx_dir / f"{pid}_context.md"
        safe = cutoff.replace("/", "-").replace(" ", "_")
        done = os.path.exists(os.path.join(snap_dir, f"{safe}.json"))
        plan.append({"pid": pid, "cutoff": cutoff, "ctxf": ctxf, "done": done})

    n_done = sum(1 for r in plan if r["done"])
    print(f"[phase1] {args.event_name}: {len(plan)} 点, 已有快照 {n_done}, 待推 {len(plan)-n_done}", flush=True)

    sim_params = {"n_agents": int(os.environ.get("MIROFISH_AGENTS", "8")),
                  "n_steps": int(os.environ.get("MIROFISH_STEPS", "2"))}
    sess = make_session("mirofish", args.base, max_tokens=args.max_tokens,
                        thinking_budget=args.thinking_budget, sim_params=sim_params)
    print(f"[phase1] session started, steps={sim_params['n_steps']}, "
          f"mem_keep={os.environ.get('MIROFISH_MEM_KEEP','0')}", flush=True)

    ADV_MAX = int(os.environ.get("SB_AGENT_ADVANCE_MAX_ATTEMPTS", "500"))
    COOLDOWN = int(os.environ.get("SB_AGENT_ADVANCE_COOLDOWN", "15"))
    prev_ctx, prev_cutoff = "", ""
    try:
        for rec in plan:
            pid, cutoff, ctxf = rec["pid"], rec["cutoff"], rec["ctxf"]
            cur_ctx = ctxf.read_text(encoding="utf-8") if ctxf.exists() else ""
            # Note: even if this point already has a snapshot, prev_ctx must be advanced to keep
            # the delta chain continuous (and advance really runs).
            # Resume: points with existing snapshots still need advance so the social state is in
            # place (unless it's a continuous run from the start).
            # Simple + correct: really advance every point (rollout can't skip, or the social state
            # gets a gap); an existing snapshot only means it ran before, and rerunning here
            # overwrites it — for strict correctness, a restart rolls from the beginning
            # (snapshots overwritten under the same name).
            delta = compute_delta(prev_ctx, cur_ctx, prev_cutoff, cutoff)
            _err = None
            for att in range(1, ADV_MAX + 1):
                try:
                    sess.advance(delta, cutoff)   # the fw layer auto-saves a snapshot after a successful advance
                    break
                except Exception as e:  # noqa: BLE001
                    _err = e
                    print(f"[phase1] 推演到 {pid}({cutoff}) 失败 第{att}/{ADV_MAX}次: {e}", flush=True)
                    if att < ADV_MAX:
                        time.sleep(COOLDOWN)
            else:
                print(f"[phase1] ⚠️事故 {pid} 推演 {ADV_MAX} 次失败,链断,停: {_err}", flush=True)
                sys.exit(2)
            prev_ctx, prev_cutoff = cur_ctx, cutoff
            print(f"[phase1] ✓ {pid} ({cutoff}) 推演完成+快照已落 {time.strftime('%H:%M:%S')}", flush=True)
    finally:
        try:
            sess.close()
        except Exception:
            pass
    print(f"[phase1] ALL_DONE {args.event_name} {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
