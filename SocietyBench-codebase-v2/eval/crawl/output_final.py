#!/usr/bin/env python3
"""output_final.py — Step 7 (1:1 of skills/web-search-crawl/step7-output/SKILL.md).

Final consolidation:
  - Reads `{project}_enriched.json` (from Step 6 backfill)
  - Filters to records with `fetch_status == "ok"` AND meaningful content
  - Optionally restricts to text+markdown ≥ 500 chars (SKILL §7.4 quality bar)
  - Emits `<output_dir>/articles.json` with the record schema from SKILL §7.2:
        {period_start, period_end, keyword, title, url, date, description,
         v2_related, v3_related, content_text, content_markdown, fetch_status}
  - Writes a short analytics summary per SKILL §7.3 (date distribution, keyword hit rate, source distribution)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")


SCHEMA_KEYS = [
    "period_start", "period_end", "keyword",
    "title", "url", "date", "description",
    "v2_related", "v3_related",
    "content_text", "content_markdown", "fetch_status",
]


def project_record(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in SCHEMA_KEYS}


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step7: final output + analytics")
    p.add_argument("--input", required=True, help="{project}_enriched.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-content-chars", type=int, default=0,
                   help="filter out records with text+markdown shorter than N chars (default 0 = keep all)")
    p.add_argument("--out-name", default="articles.json",
                   help="final consolidated filename (default: articles.json)")
    args = p.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = raw.get("all_results", []) if isinstance(raw, dict) else raw

    selected: List[Dict[str, Any]] = []
    for r in records:
        if r.get("fetch_status") != "ok":
            continue
        body_len = len((r.get("content_text") or "")) + len((r.get("content_markdown") or ""))
        if body_len < args.min_content_chars:
            continue
        selected.append(project_record(r))

    # Analytics
    by_date = Counter()
    by_source = Counter()
    by_keyword = Counter()
    by_keyword_ok = Counter()
    for r in records:
        kw = r.get("keyword") or ""
        if kw:
            by_keyword[kw] += 1
            if r.get("v3_related"):
                by_keyword_ok[kw] += 1
        if r.get("fetch_status") == "ok":
            d = (r.get("date") or "")[:10]
            if d:
                by_date[d] += 1
            try:
                host = urlparse(r.get("url") or "").netloc.lower()
                if host.startswith("www."):
                    host = host[4:]
                if host:
                    by_source[host] += 1
            except Exception:
                pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    analytics = {
        "total_records_in_enriched": len(records),
        "final_articles_count": len(selected),
        "min_content_chars": args.min_content_chars,
        "date_distribution_top20": dict(by_date.most_common(20)),
        "source_distribution_top20": dict(by_source.most_common(20)),
        "keyword_effectiveness": {
            kw: {"total": by_keyword[kw], "v3_related": by_keyword_ok[kw],
                 "rate": round(100 * by_keyword_ok[kw] / max(1, by_keyword[kw]), 2)}
            for kw in by_keyword
        },
    }
    (out_dir / "analytics_summary.json").write_text(
        json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "enriched_total": len(records),
        "final_articles": len(selected),
        "output_file": str(out_path),
        "analytics_file": str(out_dir / "analytics_summary.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
