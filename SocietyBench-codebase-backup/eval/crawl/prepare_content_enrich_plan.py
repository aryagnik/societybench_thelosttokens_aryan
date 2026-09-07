#!/usr/bin/env python3
"""prepare_content_enrich_plan.py — Step 5a (1:1 of skills/web-search-crawl/step5-qe-filter/SKILL.md §5.1).

Pre-Step-5 mechanical prep:
  1. Select records where v2_related == True AND v3_related == True
  2. URL normalization (strip www., lowercase scheme, trailing slash, fragments)
  3. URL de-dup (one URL → one record kept, prefer fullest description)
  4. Emit 5 artifacts per SKILL §5.1:
        v23_intersection_records.csv
        v23_intersection_unique_urls.csv
        v23_url_record_map.csv
        period58_to_round31_map.csv (passthrough from segments)
        website_content_crawler_input_v23.json

Input  : <project>_v3_judged.json + optional <project>_segments.json
Output : <output_dir>/content_enrich/{five files above}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

sys.stdout.reconfigure(encoding="utf-8")


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        u = urlparse(url.strip())
    except Exception:
        return url.strip()
    scheme = (u.scheme or "https").lower()
    netloc = u.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = u.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", u.query, ""))


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step5a: v2∩v3 intersection + URL dedup")
    p.add_argument("--input", required=True, help="{project}_v3_judged.json")
    p.add_argument("--output-dir", required=True, help="will create <output_dir>/content_enrich/")
    p.add_argument("--segments-json", default=None,
                   help="optional {project}_segments.json from Step 2 to emit period58→round31 map")
    args = p.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = raw.get("all_results", []) if isinstance(raw, dict) else raw

    intersected = [r for r in records if r.get("v2_related") and r.get("v3_related")]

    # Assign record_id (stable, ordered)
    for i, r in enumerate(intersected):
        r["record_id"] = i + 1
        r["url_norm"] = normalize_url(r.get("url", ""))

    # URL dedup — keep the record with the longest description
    by_url: Dict[str, Dict[str, Any]] = {}
    for r in intersected:
        u = r["url_norm"]
        if not u:
            continue
        existing = by_url.get(u)
        if existing is None or len(r.get("description", "")) > len(existing.get("description", "")):
            by_url[u] = r

    out_dir = Path(args.output_dir) / "content_enrich"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. v23_intersection_records.csv
    with (out_dir / "v23_intersection_records.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "url", "url_norm", "title", "date", "keyword", "period_start", "period_end"])
        for r in intersected:
            w.writerow([
                r.get("record_id"), r.get("url"), r.get("url_norm"),
                r.get("title"), r.get("date"), r.get("keyword"),
                r.get("period_start"), r.get("period_end"),
            ])

    # 2. v23_intersection_unique_urls.csv
    with (out_dir / "v23_intersection_unique_urls.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url_norm", "url", "title", "date", "source"])
        for u, r in by_url.items():
            w.writerow([u, r.get("url"), r.get("title"), r.get("date"), r.get("source", "")])

    # 3. v23_url_record_map.csv
    with (out_dir / "v23_url_record_map.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["url_norm", "record_ids"])
        url_to_records: Dict[str, List[int]] = {}
        for r in intersected:
            url_to_records.setdefault(r["url_norm"], []).append(r["record_id"])
        for u, rids in url_to_records.items():
            w.writerow([u, ",".join(map(str, rids))])

    # 4. period58_to_round31_map.csv (passthrough from segments)
    if args.segments_json and Path(args.segments_json).exists():
        seg_data = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
        with (out_dir / "period58_to_round31_map.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["period_start", "period_end", "period_days", "period_count", "type"])
            for s in seg_data.get("segments", []):
                w.writerow([s.get("start"), s.get("end"), s.get("period_days"),
                            s.get("period_count"), s.get("type")])

    # 5. website_content_crawler_input_v23.json — Apify crawler input
    crawler_input = {
        "startUrls": [{"url": u} for u in by_url.keys()],
        "crawlerType": "playwright:adaptive",
        "maxCrawlDepth": 0,
        "useSitemaps": False,
        "maxConcurrency": 24,
        "maxRequestRetries": 0,
        "removeCookieWarnings": True,
        "htmlTransformer": "readableText",
        "saveMarkdown": True,
        "saveHtmlAsFile": True,
        "excludeUrlGlobs": ["*://*/search*", "*://*/discover/*"],
    }
    (out_dir / "website_content_crawler_input_v23.json").write_text(
        json.dumps(crawler_input, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "intersection_records": len(intersected),
        "unique_urls": len(by_url),
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
