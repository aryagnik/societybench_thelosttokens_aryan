#!/usr/bin/env python3
"""search_bulk.py — Step 3 of skills/web-search-crawl/SKILL.md.

Event-generic Google Search execution via Apify. Takes a JSON config that
defines keywords + time segments and runs N×M searches concurrently. Output:
`<output_dir>/<project>_full.json` plus a progress file for resume.

The full 8-step pipeline orchestrator is `web_search_crawl_pipeline.py`. The
other steps live in this directory:
  Step 0: env_precheck.py
  Step 1: keyword_optimize.py
  Step 2: time_periods.py
  Step 3: search_bulk.py                              ← this file
  Step 4a: compile_v2_judge.py
  Step 4b: compile_v3_related.py
  Step 5a: prepare_content_enrich_plan.py
  Step 5b: compile_queryengine_related_cc_v2.py
  Step 5c: finalize_content_crawler_input_under2000.py
  Step 6: run_content_crawler_batch.py
        + merge_content_enrich_backfill.py
  Step 7: output_final.py

Config JSON schema:
  {
    "project": "tesla_mexico",
    "keywords": ["...", "..."],
    "segments": [
      {"start": "2025-04-01", "end": "2025-07-05", "period_days": 30, "period_count": 2},
      ...
    ]
  }
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")

from apify_client import ApifyClient


MAX_PAGES = 100
RESULTS_PER_PAGE = 100
LIMIT = 99999
TIMEOUT = 3600
MAX_RETRIES = 1


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def guess_source(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def expand_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    periods: List[Dict[str, Any]] = []
    for seg in segments:
        seg_start = date.fromisoformat(seg["start"])
        seg_end = date.fromisoformat(seg["end"])
        period_days = int(seg.get("period_days", 30))
        period_count = int(seg.get("period_count", 1))
        cursor = seg_start
        for i in range(period_count):
            p_start = cursor
            if i == period_count - 1:
                p_end = seg_end
            else:
                p_end = cursor + timedelta(days=period_days - 1)
                if p_end > seg_end:
                    p_end = seg_end
            periods.append({"start": p_start, "end": p_end})
            cursor = p_end + timedelta(days=1)
    return periods


def load_token() -> str:
    token = (os.getenv("APIFY_TOKEN") or "").strip()
    if token:
        return token
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("APIFY_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("Missing APIFY_TOKEN (set env or put in crawl/.env)")


def load_progress(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except Exception:
        return set()


def save_progress(path: Path, done: Set[str]) -> None:
    write_json_atomic(path, {"done": sorted(done)})


def run_one_period(
    client: ApifyClient,
    keywords: List[str],
    period: dict, period_idx: int, total_periods: int,
) -> dict:
    start_s = period["start"].isoformat()
    end_s = period["end"].isoformat()
    next_day_s = (period["end"] + timedelta(days=1)).isoformat()
    period_key = f"{start_s}|{end_s}"

    queries = [f"{kw} after:{start_s} before:{next_day_s}" for kw in keywords]
    queries_str = "\n".join(queries)
    print(f"[{period_idx}/{total_periods}] {start_s} ~ {end_s} ({len(keywords)} kws)", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.run_actor_and_fetch_items(
                "apify/google-search-scraper",
                input_data={
                    "queries": queries_str,
                    "resultsPerPage": RESULTS_PER_PAGE,
                    "maxPagesPerQuery": MAX_PAGES,
                },
                item_limit=LIMIT,
                wait_timeout_secs=TIMEOUT,
            )
            items = resp.get("items", []) if isinstance(resp, dict) else []
            results = []
            for serp in items:
                search_query = serp.get("searchQuery", {}) if isinstance(serp, dict) else {}
                raw_query = search_query.get("term") or ""
                kw_part = raw_query
                if " after:" in kw_part:
                    kw_part = kw_part[:kw_part.index(" after:")]
                for r in serp.get("organicResults") or []:
                    url = r.get("url", "")
                    results.append({
                        "period_start": start_s,
                        "period_end": end_s,
                        "keyword": kw_part,
                        "query": raw_query,
                        "title": r.get("title", ""),
                        "url": url,
                        "source": guess_source(url),
                        "date": r.get("date"),
                        "description": (r.get("description") or "")[:500],
                        "position": r.get("position"),
                    })
            print(f"  -> OK, {len(results)} 条", flush=True)
            return {"period_key": period_key, "ok": True, "results": results, "errors": []}
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 5)
            else:
                print(f"  -> 失败: {e}", flush=True)
                return {"period_key": period_key, "ok": False, "results": [], "errors": [str(e)]}
    return {"period_key": period_key, "ok": False, "results": [], "errors": ["unknown"]}


def main() -> None:
    p = argparse.ArgumentParser(description="web-search-crawl: Apify Google Search execution (generic)")
    p.add_argument("--config", required=True, help="JSON with {project, keywords, segments}")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-concurrent", type=int, default=env_int("APIFY_SEARCH_MAX_CONCURRENT", 40))
    p.add_argument("--limit", type=int, default=0, help="only run the first N segments (0 = all)")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project = cfg["project"]
    keywords = cfg["keywords"]
    segments = cfg["segments"]
    if not keywords or not segments:
        raise SystemExit("config must include non-empty keywords and segments")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project}_full.json"
    progress_path = out_dir / f"{project}_progress.json"

    token = load_token()
    client = ApifyClient(token=token, timeout=TIMEOUT)
    periods = expand_segments(segments)
    if args.limit > 0:
        periods = periods[: args.limit]

    print(f"=== web-search-crawl: {project} ===")
    print(f"periods={len(periods)} keywords={len(keywords)} concurrent={args.max_concurrent}")

    done = load_progress(progress_path)
    all_results: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            all_results = existing.get("all_results", [])
        except Exception:
            pass

    todo = []
    for i, p_ in enumerate(periods, start=1):
        key = f"{p_['start'].isoformat()}|{p_['end'].isoformat()}"
        if key not in done:
            todo.append((i, p_))

    errors = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_concurrent)) as ex:
        futs = {ex.submit(run_one_period, client, keywords, period, i, len(periods)): (i, period)
                for i, period in todo}
        save_counter = 0
        for f in concurrent.futures.as_completed(futs):
            try:
                result = f.result()
            except Exception as e:
                errors += 1
                print(f"thread error: {e}", flush=True)
                continue
            if result["ok"]:
                all_results.extend(result["results"])
                done.add(result["period_key"])
                completed += 1
            else:
                errors += 1
            save_counter += 1
            if save_counter % 5 == 0:
                save_progress(progress_path, done)
                write_json_atomic(out_path, {
                    "project": project,
                    "keywords_count": len(keywords),
                    "periods_count": len(periods),
                    "completed_periods": len(done),
                    "errors": errors,
                    "total_results": len(all_results),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "keywords": keywords,
                    "all_results": all_results,
                })

    save_progress(progress_path, done)
    write_json_atomic(out_path, {
        "project": project,
        "keywords_count": len(keywords),
        "periods_count": len(periods),
        "completed_periods": len(done),
        "errors": errors,
        "total_results": len(all_results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "keywords": keywords,
        "all_results": all_results,
    })

    print(f"\n=== Done: {completed} completed this run, errors={errors}, total={len(all_results)}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] progress saved; rerun to resume")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[fatal] {e}")
        raise SystemExit(1)
