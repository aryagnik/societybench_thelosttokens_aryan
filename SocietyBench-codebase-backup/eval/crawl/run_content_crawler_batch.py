#!/usr/bin/env python3
"""run_content_crawler_batch.py — Step 6 (1:1 of skills/web-search-crawl/step6-crawl/SKILL.md).

URL sharding + parallel Apify runs for content enrichment.
SKILL §6.1 config: playwright:adaptive, maxCrawlDepth=0, removeCookieWarnings,
htmlTransformer=readableText, saveMarkdown/HtmlAsFile.
SKILL §6.0: default URL ceiling <2000 (set in Step 5c).
SKILL §6.2 parallelism: parallel_runs=14, memory=8192MB per run when single-event
dedicated, else falls back.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from apify_client import ApifyClient

DEFAULT_QE_INPUT = Path("outputs/content_enrich/final_crawl_urls.json")
DEFAULT_PREPARED_INPUT = Path("outputs/content_enrich/website_content_crawler_input_v23.json")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_apify_token() -> str:
    token = os.getenv("APIFY_TOKEN", "").strip()
    if token:
        return token
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("APIFY_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def build_shards(start_urls: List[Dict[str, Any]], shard_count: int) -> List[List[Dict[str, Any]]]:
    if shard_count <= 1:
        return [start_urls]
    shard_count = min(shard_count, len(start_urls))
    base = len(start_urls) // shard_count
    rem = len(start_urls) % shard_count
    shards: List[List[Dict[str, Any]]] = []
    cursor = 0
    for i in range(shard_count):
        size = base + (1 if i < rem else 0)
        shards.append(start_urls[cursor : cursor + size])
        cursor += size
    return [s for s in shards if s]


def fetch_all_dataset_items(client: ApifyClient, dataset_id: str, page_size: int = 1000) -> List[Dict[str, Any]]:
    offset = 0
    all_items: List[Dict[str, Any]] = []
    while True:
        batch = client.get_dataset_items(dataset_id, limit=page_size, offset=offset, clean=True)
        if not isinstance(batch, list) or not batch:
            break
        all_items.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)
    return all_items


def resolve_input_json(path_str: str) -> Path:
    requested = Path(path_str)
    if requested.exists():
        return requested
    if requested == DEFAULT_QE_INPUT and DEFAULT_PREPARED_INPUT.exists():
        return DEFAULT_PREPARED_INPUT
    raise SystemExit(f"Input not found: {requested}")


def load_existing_run_records(path: Path, resolved_input_path: Path, top_n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    stored_input = payload.get("input_json")
    if stored_input and Path(str(stored_input)) != resolved_input_path:
        return []

    stored_top_n = payload.get("top_n")
    if isinstance(stored_top_n, int) and stored_top_n != top_n:
        return []

    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return runs


def build_run_meta_payload(
    *,
    args: argparse.Namespace,
    top_n: int,
    input_used_path: Path,
    resolved_input_path: Path,
    run_records: List[Dict[str, Any]],
    items_count: int,
) -> Dict[str, Any]:
    return {
        "actor_id": args.actor_id,
        "input_json": str(resolved_input_path),
        "top_n": top_n,
        "parallel_runs": len(run_records),
        "max_concurrency": args.max_concurrency,
        "initial_concurrency": args.initial_concurrency,
        "max_request_retries": args.max_request_retries,
        "memory_mbytes": args.memory_mbytes,
        "input_used": str(input_used_path),
        "runs": run_records,
        "items_count": items_count,
    }


def save_run_snapshot(
    *,
    args: argparse.Namespace,
    top_n: int,
    input_used_path: Path,
    resolved_input_path: Path,
    run_records: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
) -> None:
    write_json_atomic(
        Path(args.run_meta_out),
        build_run_meta_payload(
            args=args,
            top_n=top_n,
            input_used_path=input_used_path,
            resolved_input_path=resolved_input_path,
            run_records=run_records,
            items_count=len(items),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run website-content-crawler with top-N URLs")
    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_QE_INPUT),
        help="Prepared crawler input; defaults to QE-filtered input and falls back to the pre-QE file if needed",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Use first N startUrls for this run (0=all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=env_int("APIFY_CONTENT_MAX_CONCURRENCY", 24),
    )
    parser.add_argument(
        "--initial-concurrency",
        type=int,
        default=env_int("APIFY_CONTENT_INITIAL_CONCURRENCY", 12),
    )
    parser.add_argument("--max-request-retries", type=int, default=0)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=env_int("APIFY_CONTENT_PARALLEL_RUNS", 10),
        help="Start N actor runs in parallel",
    )
    parser.add_argument(
        "--poll-interval-secs",
        type=int,
        default=env_int("APIFY_CONTENT_POLL_INTERVAL_SECS", 5),
        help="Polling interval for run status",
    )
    parser.add_argument(
        "--memory-mbytes",
        type=int,
        default=env_int("APIFY_CONTENT_MEMORY_MBYTES", 4096),
        help="Memory per actor run in MB",
    )
    parser.add_argument(
        "--actor-id",
        default="apify/website-content-crawler",
    )
    parser.add_argument(
        "--run-meta-out",
        default="outputs/content_enrich/crawl_run_meta.json",
    )
    parser.add_argument(
        "--items-out",
        default="outputs/content_enrich/website_content_crawler_items.json",
    )
    parser.add_argument(
        "--strict-start-urls-only",
        action="store_true",
        help="Only crawl startUrls pages (no link following / sitemap / llms.txt expansion)",
    )
    args = parser.parse_args()
    args.max_concurrency = max(1, args.max_concurrency)
    args.initial_concurrency = max(1, min(args.initial_concurrency, args.max_concurrency))
    args.parallel_runs = max(1, args.parallel_runs)
    args.memory_mbytes = max(128, args.memory_mbytes)

    token = load_apify_token()
    if not token:
        raise SystemExit("Missing APIFY_TOKEN (env or .env)")

    in_path = resolve_input_json(args.input_json)
    cfg = json.loads(in_path.read_text(encoding="utf-8"))
    start_urls = cfg.get("startUrls", [])
    if not isinstance(start_urls, list) or not start_urls:
        raise SystemExit("input startUrls is empty")

    top_n = len(start_urls) if args.top_n <= 0 else min(args.top_n, len(start_urls))
    cfg["startUrls"] = start_urls[:top_n]
    cfg["maxCrawlPages"] = top_n
    cfg["maxConcurrency"] = args.max_concurrency
    cfg["initialConcurrency"] = args.initial_concurrency
    cfg["maxRequestRetries"] = args.max_request_retries
    if args.strict_start_urls_only:
        cfg["maxCrawlDepth"] = 0
        cfg["useSitemaps"] = False
        cfg["useLlmsTxt"] = False

    input_used_path = Path(args.run_meta_out).with_name(f"{Path(args.run_meta_out).stem}_input_used.json")
    write_json_atomic(input_used_path, cfg)

    client = ApifyClient(token=token, timeout=120)
    used_start_urls = cfg["startUrls"]
    shards = build_shards(used_start_urls, args.parallel_runs)

    run_records = load_existing_run_records(Path(args.run_meta_out), in_path, top_n)
    items: List[Dict[str, Any]] = []
    try:
        for idx in range(len(run_records), len(shards)):
            shard = shards[idx]
            run_cfg = dict(cfg)
            run_cfg["startUrls"] = shard
            run_cfg["maxCrawlPages"] = len(shard)
            run = client.run_actor(
                args.actor_id,
                input_data=run_cfg,
                memory_mbytes=max(128, args.memory_mbytes),
            )
            run_records.append(
                {
                    "shard_index": idx,
                    "shard_size": len(shard),
                    "run_input": {
                        "maxCrawlPages": run_cfg.get("maxCrawlPages"),
                        "maxConcurrency": run_cfg.get("maxConcurrency"),
                        "initialConcurrency": run_cfg.get("initialConcurrency"),
                        "maxRequestRetries": run_cfg.get("maxRequestRetries"),
                        "maxCrawlDepth": run_cfg.get("maxCrawlDepth"),
                        "useSitemaps": run_cfg.get("useSitemaps"),
                        "useLlmsTxt": run_cfg.get("useLlmsTxt"),
                    },
                    "run": run,
                }
            )
            save_run_snapshot(
                args=args,
                top_n=top_n,
                input_used_path=input_used_path,
                resolved_input_path=in_path,
                run_records=run_records,
                items=items,
            )

        deadline = time.time() + max(1, args.wait_timeout)
        remaining = {r["run"].get("id"): r for r in run_records if r["run"].get("id")}
        while remaining:
            if time.time() >= deadline:
                save_run_snapshot(
                    args=args,
                    top_n=top_n,
                    input_used_path=input_used_path,
                    resolved_input_path=in_path,
                    run_records=run_records,
                    items=items,
                )
                raise TimeoutError(f"Not all runs finished within {args.wait_timeout}s")
            finished_ids: List[str] = []
            status_changed = False
            for run_id, record in list(remaining.items()):
                latest = client.get_run(run_id)
                if latest != record.get("run"):
                    status_changed = True
                record["run"] = latest
                if latest.get("status") in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
                    finished_ids.append(run_id)
            for rid in finished_ids:
                remaining.pop(rid, None)
            if status_changed or finished_ids:
                save_run_snapshot(
                    args=args,
                    top_n=top_n,
                    input_used_path=input_used_path,
                    resolved_input_path=in_path,
                    run_records=run_records,
                    items=items,
                )
            if remaining:
                time.sleep(max(1, args.poll_interval_secs))

        for rec in run_records:
            run = rec.get("run", {})
            if run.get("status") != "SUCCEEDED":
                continue
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                continue
            shard_items = fetch_all_dataset_items(client, dataset_id)
            items.extend(shard_items)
            write_json_atomic(Path(args.items_out), items if isinstance(items, list) else [])
            save_run_snapshot(
                args=args,
                top_n=top_n,
                input_used_path=input_used_path,
                resolved_input_path=in_path,
                run_records=run_records,
                items=items,
            )
    except Exception:
        save_run_snapshot(
            args=args,
            top_n=top_n,
            input_used_path=input_used_path,
            resolved_input_path=in_path,
            run_records=run_records,
            items=items,
        )
        raise

    write_json_atomic(Path(args.items_out), items if isinstance(items, list) else [])
    save_run_snapshot(
        args=args,
        top_n=top_n,
        input_used_path=input_used_path,
        resolved_input_path=in_path,
        run_records=run_records,
        items=items,
    )

    print(
        json.dumps(
            {
                "status": "DONE",
                "input_json": str(in_path),
                "parallel_runs": len(shards),
                "succeeded_runs": sum(1 for r in run_records if r.get("run", {}).get("status") == "SUCCEEDED"),
                "items_count": len(items) if isinstance(items, list) else 0,
                "run_meta_out": args.run_meta_out,
                "items_out": args.items_out,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
