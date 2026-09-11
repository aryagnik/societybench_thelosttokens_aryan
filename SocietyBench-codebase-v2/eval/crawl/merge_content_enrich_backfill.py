#!/usr/bin/env python3
"""merge_content_enrich_backfill.py — Step 6 backfill (1:1 of skills/web-search-crawl/step6-crawl/SKILL.md §6.x).

After run_content_crawler_batch.py finishes (one or more rounds), this script:
  - Reads all per-run crawler results from `<output_dir>/content_enrich/`
  - Loads the source v3 records (from {project}_v3_judged.json)
  - Maps each crawled URL → original record_id via v23_url_record_map.csv
  - Backfills `content_text`, `content_markdown`, `fetch_status` into the
    original record
  - Emits `<output_dir>/{project}_enriched.json`

Per SKILL §7.4 measured retention: ~77% after round 1, ~89% after multi-round backfill crawls.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def load_crawler_items(content_enrich_dir: Path) -> List[Dict[str, Any]]:
    """Crawler runs can dump items as .json files in content_enrich/.
    Convention: each run produces one or more {run_id}_items.json files.
    """
    items: List[Dict[str, Any]] = []
    for fp in content_enrich_dir.glob("*items*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict) and isinstance(data.get("items"), list):
            items.extend(data["items"])
    return items


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step6 backfill: merge crawler items into source records")
    p.add_argument("--input-judged", required=True, help="{project}_v3_judged.json")
    p.add_argument("--content-enrich-dir", required=True,
                   help="<output_dir>/content_enrich/ containing crawler items + v23_url_record_map.csv")
    p.add_argument("--output", required=True, help="{project}_enriched.json")
    args = p.parse_args()

    enrich_dir = Path(args.content_enrich_dir)

    # Load record map: url_norm → [record_id, ...]
    url_to_record_ids: Dict[str, List[int]] = {}
    map_csv = enrich_dir / "v23_url_record_map.csv"
    if map_csv.exists():
        with map_csv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                u = row.get("url_norm", "")
                ids = [int(x) for x in (row.get("record_ids") or "").split(",") if x.strip().isdigit()]
                if u and ids:
                    url_to_record_ids[u] = ids

    # Load v3 records
    judged = json.loads(Path(args.input_judged).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = judged.get("all_results", []) if isinstance(judged, dict) else judged
    # Build record_id → record (assuming step5a assigned record_id starting from 1
    # in intersection order); we re-assign here defensively
    intersected_records = [r for r in records if r.get("v2_related") and r.get("v3_related")]
    by_record_id: Dict[int, Dict[str, Any]] = {}
    for i, r in enumerate(intersected_records, start=1):
        by_record_id[i] = r

    # Load crawler items
    items = load_crawler_items(enrich_dir)

    # Group items by url_norm; if multiple, prefer one with text + markdown + ok status
    by_url: Dict[str, Dict[str, Any]] = {}
    for it in items:
        url = it.get("url") or it.get("loadedUrl") or ""
        u = normalize_url(url)
        if not u:
            continue
        cur = by_url.get(u)
        text_len = len((it.get("text") or "") + (it.get("markdown") or ""))
        cur_len = len((cur or {}).get("__rank_text", "")) if cur else 0
        if cur is None or text_len > cur_len:
            by_url[u] = {**it, "__rank_text": (it.get("text") or "") + (it.get("markdown") or "")}

    # Backfill
    enriched = 0
    out_records: List[Dict[str, Any]] = []
    for r in intersected_records:
        url_norm = normalize_url(r.get("url", ""))
        crawled = by_url.get(url_norm)
        new_r = dict(r)
        if crawled:
            new_r["content_text"] = (crawled.get("text") or "")
            new_r["content_markdown"] = (crawled.get("markdown") or "")
            new_r["fetch_status"] = "ok" if (crawled.get("text") or crawled.get("markdown")) else "empty"
            if new_r["fetch_status"] == "ok":
                enriched += 1
        else:
            new_r["content_text"] = ""
            new_r["content_markdown"] = ""
            new_r["fetch_status"] = "missing"
        out_records.append(new_r)

    output = {
        "project": judged.get("project") if isinstance(judged, dict) else None,
        "intersection_total": len(intersected_records),
        "enriched_ok": enriched,
        "enriched_pct": round(100 * enriched / max(1, len(intersected_records)), 2),
        "all_results": out_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "intersection_records": len(intersected_records),
        "crawler_items_loaded": len(items),
        "unique_urls_crawled": len(by_url),
        "enriched_ok": enriched,
        "enriched_pct": output["enriched_pct"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
