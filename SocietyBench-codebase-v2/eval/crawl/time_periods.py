#!/usr/bin/env python3
"""web-search-crawl Step 2: time-period design (1:1 of skills/web-search-crawl/step2-timeperiods/SKILL.md).

SKILL §2.2 rules:
  1. Each event date → ±5 day burst window
  2. Merge overlapping / <7-day-gap burst windows
  3. Burst windows: 7-day granularity (per7d=1 round)
  4. Gap windows: 30-day granularity (entire gap = 1 round)
  5. Pre-roll 60 days before earliest, 30-day granularity
  6. Post-roll latest → today, 30-day granularity
  7. Adjacent 30-day segments merge into 1 round

Input  : list of event dates (YYYY-MM-DD) OR a timeline JSON
Output : <output_dir>/<project>_segments.json (used by search_bulk.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")


def parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def load_event_dates(args: argparse.Namespace) -> List[date]:
    dates: List[date] = []
    if args.event_dates:
        for s in args.event_dates.split(","):
            s = s.strip()
            if s:
                dates.append(parse_date(s))
    if args.event_dates_json:
        data = json.loads(Path(args.event_dates_json).read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    dates.append(parse_date(item))
                elif isinstance(item, dict) and item.get("date"):
                    dates.append(parse_date(item["date"]))
        elif isinstance(data, dict) and "events" in data:
            for ev in data["events"]:
                if isinstance(ev, dict) and ev.get("date"):
                    dates.append(parse_date(ev["date"]))
    return sorted(set(dates))


def merge_bursts(bursts: List[tuple]) -> List[tuple]:
    """Merge bursts whose gap is <7 days. Input/output: list of (start, end)."""
    if not bursts:
        return []
    bursts = sorted(bursts)
    merged: List[List[Any]] = [list(bursts[0])]
    for start, end in bursts[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end).days < 7:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step2: time-period segmentation")
    p.add_argument("--project", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--event-dates", default=None,
                   help="comma-separated event dates, e.g. 2023-10-13,2025-07-20")
    p.add_argument("--event-dates-json", default=None,
                   help="JSON file: list of dates or list of {date: ...}")
    p.add_argument("--today", default=None, help="override 'today' for reproducibility")
    args = p.parse_args()

    event_dates = load_event_dates(args)
    if not event_dates:
        sys.exit("must supply --event-dates or --event-dates-json with at least one date")

    today = parse_date(args.today) if args.today else date.today()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SKILL §2.2 rule 1: each event ±5 days
    raw_bursts = [(d - timedelta(days=5), d + timedelta(days=5)) for d in event_dates]
    # rule 2: merge
    bursts = merge_bursts(raw_bursts)

    earliest = min(b[0] for b in bursts)
    latest = max(b[1] for b in bursts)

    # Build full segment list
    segments: List[Dict[str, Any]] = []
    # rule 5: pre-roll 60 days before earliest, 30-day granularity
    pre_start = earliest - timedelta(days=60)
    pre_end = earliest - timedelta(days=1)
    if pre_end >= pre_start:
        segments.append({
            "start": pre_start.isoformat(),
            "end": pre_end.isoformat(),
            "period_days": 30,
            "period_count": 1,
            "type": "前溯",
        })

    # bursts + gaps
    for i, (b_start, b_end) in enumerate(bursts):
        # gap before this burst (if not the first)
        if i > 0:
            prev_end = bursts[i - 1][1]
            gap_start = prev_end + timedelta(days=1)
            gap_end = b_start - timedelta(days=1)
            if gap_end > gap_start:
                segments.append({
                    "start": gap_start.isoformat(),
                    "end": gap_end.isoformat(),
                    "period_days": 30,
                    "period_count": 1,
                    "type": "间隙",
                })
        # burst itself: 7-day granularity
        days = (b_end - b_start).days + 1
        period_count = max(1, (days + 6) // 7)
        segments.append({
            "start": b_start.isoformat(),
            "end": b_end.isoformat(),
            "period_days": 7,
            "period_count": period_count,
            "type": f"事件{i+1}",
        })

    # rule 6: post-roll from latest+1 to today
    post_start = latest + timedelta(days=1)
    if today > post_start:
        segments.append({
            "start": post_start.isoformat(),
            "end": today.isoformat(),
            "period_days": 30,
            "period_count": 1,
            "type": "后推",
        })

    total_rounds = sum(s["period_count"] for s in segments)
    output = {
        "project": args.project,
        "event_dates": [d.isoformat() for d in event_dates],
        "segments": segments,
        "summary": {
            "segment_count": len(segments),
            "total_rounds": total_rounds,
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
        },
    }
    out_path = out_dir / f"{args.project}_segments.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "segments": len(segments),
        "total_rounds": total_rounds,
        "earliest": earliest.isoformat(),
        "latest": latest.isoformat(),
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
