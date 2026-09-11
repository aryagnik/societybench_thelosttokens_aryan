#!/usr/bin/env python3
"""M4 non-LLM baselines: frequency + momentum (paper Table 8). Pure code, no API calls.

Produces one probability per calibration question, written to
results/run_<id>/brier/{frequency,momentum}/, read by the scorecard as two
"models". Both produce probabilities only, no dates → the time axis is fixed
at 50 (midpoint; M4 produces no dates). Reuses 3B's scoring.

Baseline definitions (tunable, see _rate):
  - frequency: estimate a daily occurrence rate λ from the "event density" in
    C_P (context) before the cutoff: λ = number of dated nodes before cutoff / span in days;
    probability that a question "occurs" within its W-day window ≈ clamp(λ·W, 0.02, 0.98).
  - momentum: same, but λ is estimated from only the last 7 days before the
    cutoff (captures "recent trend continuation" — the LLM failure mode called out in the paper).
Note: the plan did not pin an exact definition for the "similar-event empirical
base rate"; this is a reasonable implementation — to change it, edit _rate only.

Usage (on one event's final workspace):
  python3 eval/baseline/baseline_freq_momentum.py --workspace runs_new/event5_smci/final \\
      --event-name smci --run-id M4
  (Run once per event, or pass the same --run-id to collect all 5 events' results in one place.)
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # eval/

import predict_step3B_brier as B  # noqa: E402
from common import load_pipeline_config, parse_date_obj  # noqa: E402


def _context_dates(ctx):
    ds = []
    for m in re.finditer(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", ctx):
        try:
            ds.append(datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    return sorted(set(ds))


def _rate(dates, cutoff, window_days=None):
    """Daily occurrence rate λ. window_days=None → use all pre-cutoff history; =7 → only the last 7 days."""
    cd = parse_date_obj(cutoff)
    pre = [d for d in dates if cd and d <= cd]
    if window_days:
        lo = cd - datetime.timedelta(days=window_days)
        pre = [d for d in pre if d >= lo]
    if not pre:
        return 0.0
    span = max(1, (max(pre) - min(pre)).days)
    return len(pre) / span


def main():
    ap = argparse.ArgumentParser(description="M4 non-LLM baselines: frequency + momentum (Table 8)")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--event-name", required=True)
    ap.add_argument("--points", default="all")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    cfg = load_pipeline_config()
    cal = cfg.get("scoring_calibration", {})
    ws = pathlib.Path(a.workspace)
    qbd, ctxd = ws / "questionbank", ws / "contexts"
    if not qbd.exists():
        raise SystemExit(f"无 questionbank/:{ws}")
    rid = a.run_id or "M4_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    qbs = sorted(qbd.glob("P*_questionbank.json"))
    if a.points != "all":
        want = set(a.points.split(","))
        qbs = [f for f in qbs if f.stem.replace("_questionbank", "") in want]

    per = {"frequency": {}, "momentum": {}}
    for qf in qbs:
        pid = qf.stem.replace("_questionbank", "")
        qb = json.loads(qf.read_text(encoding="utf-8"))
        cutoff = qb.get("cutoff_date") or ""
        qs = qb.get("brier_questions") or []
        if not qs:
            continue
        ctxf = ctxd / f"{pid}_context.md"
        dates = _context_dates(ctxf.read_text(encoding="utf-8") if ctxf.exists() else "")
        lam = {"frequency": _rate(dates, cutoff), "momentum": _rate(dates, cutoff, 7)}
        cutd = parse_date_obj(cutoff)
        for name in ("frequency", "momentum"):
            L = lam[name]
            probs = [min(0.98, max(0.02, L * int(q.get("window_days", 14) or 14))) for q in qs]
            # 2026-07-01, same fix as agents/A3: non-LLM baselines produce a real
            # probability for every question (no parse failures), so score_point
            # must receive answer_attempts=all 1s; otherwise per-question
            # answer_attempt=None → the aggregate no-signal gate sees
            # default_frac=1.0 → frequency/momentum totals wrongly forced to N/A.
            sc = B.score_point(qs, probs, cal, answer_attempts=[1] * len(qs))
            enr = []
            for q, p in zip(qs, probs):
                dt = parse_date_obj(q.get("d_target"))
                dd = (dt - cutd).days if (dt and cutd) else 0
                enr.append({**q, "prob": p, "delta_days": dd,
                            "gt": int((q.get("answer") if q.get("answer") is not None else q.get("gt", 0)) or 0)})
            out = ws / "results" / f"run_{rid}" / "brier" / name / f"{pid}_brier.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"point_id": pid, "model": name, "cutoff_date": cutoff,
                                       "questions": enr, **sc}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            per[name][pid] = sc

    for name in per:
        agg = B.aggregate_across_points(per[name], cal)
        (ws / "results" / f"run_{rid}" / "brier" / name / "aggregated.json").write_text(
            json.dumps({"model": name, "n_points": len(per[name]), "batch_size": None, "aggregated": agg},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}: 合池 Cal = {agg.get('score_100')}")
    print(f"完成。结果在 {ws}/results/run_{rid}/brier/(frequency, momentum)")
    print("注:时间轴固定 50(中点),M4 不产日期 → 时间分另行按 50 记。")


if __name__ == "__main__":
    main()
