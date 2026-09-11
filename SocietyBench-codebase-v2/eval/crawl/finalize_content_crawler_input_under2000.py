#!/usr/bin/env python3
"""finalize_content_crawler_input_under2000.py — Step 5c (1:1 of skills/web-search-crawl/step5-qe-filter/SKILL.md §5.2.1).

If QE-passed URL count > 2000 (SKILL line 21: hard ceiling), do a single
final convergence pass that:
  - excludes URLs already attempted (if a prior attempted-urls file is given)
  - ranks remaining by (importance × time-density × QE-score)
  - emits ≤2000 final URL list

Per SKILL line 99-120: importance = topic-relevance signal from title+desc;
time-density = denser around event dates (loaded from segments JSON if given).

Input  : <output_dir>/content_enrich/v23_qe_passed_unique_urls.csv
         optional: segments JSON (for time-density), attempted URL list
Output : <output_dir>/content_enrich/final_crawl_urls.json   (≤2000)
         <output_dir>/content_enrich/final_crawl_urls.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def importance_score(title: str, topic_terms: List[str]) -> float:
    if not title:
        return 0.0
    s = 0.0
    title_l = title.lower()
    for t in topic_terms:
        if t and t.lower() in title_l:
            s += 1.0
    return s


def time_density_score(date: datetime | None, event_dates: List[datetime]) -> float:
    """Higher score for URLs whose date is closer to any event date."""
    if date is None or not event_dates:
        return 0.5  # neutral default
    days = [abs((date - ed).days) for ed in event_dates]
    min_dist = min(days)
    # Gaussian-ish around event dates; sigma=14 days
    return math.exp(-(min_dist ** 2) / (2 * 14 ** 2))


def load_attempted(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix == ".csv":
        urls: set[str] = set()
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                u = row.get("url_norm") or row.get("url") or ""
                if u:
                    urls.add(u.strip())
        return urls
    # treat as txt one per line
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step5c: converge to <2000 URLs")
    p.add_argument("--qe-passed-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--segments-json", default=None, help="for event-date time-density signal")
    p.add_argument("--keywords-json", default=None, help="for topic-importance signal")
    p.add_argument("--attempted-urls", default=None,
                   help="optional CSV / TXT of URLs already attempted (excluded)")
    p.add_argument("--max-urls", type=int, default=2000, help="SKILL hard ceiling")
    args = p.parse_args()

    # Topic terms
    topic_terms: List[str] = []
    if args.keywords_json and Path(args.keywords_json).exists():
        k = json.loads(Path(args.keywords_json).read_text(encoding="utf-8"))
        for kw in k.get("keywords", []):
            m = re.search(r'"([^"]+)"', kw)
            if m:
                topic_terms.append(m.group(1))

    # Event dates from segments
    event_dates: List[datetime] = []
    if args.segments_json and Path(args.segments_json).exists():
        sd = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
        for s in sd.get("event_dates", []):
            d = parse_date(s)
            if d:
                event_dates.append(d)

    # Load candidates + score
    attempted = load_attempted(Path(args.attempted_urls) if args.attempted_urls else None)
    candidates: List[Dict[str, Any]] = []
    with Path(args.qe_passed_csv).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url_norm = (row.get("url_norm") or "").strip()
            if not url_norm or url_norm in attempted:
                continue
            d = parse_date(row.get("date"))
            imp = importance_score(row.get("title", ""), topic_terms)
            tdens = time_density_score(d, event_dates)
            qe = float(row.get("qe_score") or 0)
            score = imp * 2.0 + tdens * 3.0 + qe * 0.5
            candidates.append({
                **row,
                "importance": round(imp, 3),
                "time_density": round(tdens, 3),
                "qe_score": qe,
                "final_score": round(score, 4),
            })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    final = candidates[:args.max_urls]

    out_dir = Path(args.output_dir) / "content_enrich"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (Apify crawler input ready)
    json_path = out_dir / "final_crawl_urls.json"
    json_path.write_text(json.dumps({
        "total_candidates": len(candidates),
        "kept": len(final),
        "max_urls_cap": args.max_urls,
        "startUrls": [{"url": r["url_norm"]} for r in final],
        "url_meta": [{k: v for k, v in r.items() if k in (
            "url_norm", "url", "title", "date", "importance",
            "time_density", "qe_score", "final_score")} for r in final],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV companion
    csv_path = out_dir / "final_crawl_urls.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url_norm", "url", "title", "date", "importance", "time_density", "qe_score", "final_score"])
        for r in final:
            w.writerow([r["url_norm"], r.get("url"), r.get("title"), r.get("date"),
                        r["importance"], r["time_density"], r["qe_score"], r["final_score"]])

    print(json.dumps({
        "total_candidates": len(candidates),
        "attempted_excluded": len(attempted),
        "final_count": len(final),
        "json": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
