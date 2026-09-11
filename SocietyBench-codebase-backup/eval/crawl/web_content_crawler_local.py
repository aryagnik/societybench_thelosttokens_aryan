#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
import trafilatura
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def soup_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    if soup.title and soup.title.text:
        return soup.title.text.strip()
    return ""


def extract_text(html: str, url: str) -> Dict[str, str]:
    downloaded = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        include_formatting=False,
        include_links=False,
        favor_precision=True,
        with_metadata=True,
    )
    if downloaded:
        try:
            parsed = json.loads(downloaded)
        except Exception:
            parsed = {}
        return {
            "text": (parsed.get("text") or "").strip(),
            "title": (parsed.get("title") or "").strip(),
            "author": (parsed.get("author") or "").strip(),
            "description": (parsed.get("description") or "").strip(),
        }
    return {
        "text": "",
        "title": "",
        "author": "",
        "description": "",
    }


def fetch_one(url: str, timeout: int) -> Dict[str, Any]:
    started = time.time()
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        html = resp.text or ""
        content_type = resp.headers.get("Content-Type", "")
        loaded_url = resp.url
        meta = extract_text(html, loaded_url)
        title = meta["title"] or soup_title(html)
        item = {
            "url": loaded_url,
            "text": meta["text"],
            "markdown": meta["text"],
            "htmlUrl": "",
            "screenshotUrl": "",
            "metadata": {
                "title": title,
                "description": meta["description"],
                "author": meta["author"],
                "canonicalUrl": loaded_url,
                "headers": dict(resp.headers),
            },
            "crawl": {
                "loadedUrl": loaded_url,
                "loadedTime": now_iso(),
                "referrerUrl": url,
                "httpStatusCode": resp.status_code,
                "depth": 0,
                "contentType": content_type,
            },
            "debug": {
                "elapsed_secs": round(time.time() - started, 3),
                "fetch_method": "requests+trafilatura",
                "source_url": url,
            },
        }
        if resp.status_code >= 400:
            item["debug"]["error"] = f"http_{resp.status_code}"
        return item
    except Exception as exc:
        return {
            "url": url,
            "text": "",
            "markdown": "",
            "htmlUrl": "",
            "screenshotUrl": "",
            "metadata": {
                "title": "",
                "description": "",
                "author": "",
                "canonicalUrl": url,
                "headers": {},
            },
            "crawl": {
                "loadedUrl": url,
                "loadedTime": now_iso(),
                "referrerUrl": url,
                "httpStatusCode": 0,
                "depth": 0,
                "contentType": "",
            },
            "debug": {
                "elapsed_secs": round(time.time() - started, 3),
                "fetch_method": "requests+trafilatura",
                "source_url": url,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fallback content crawler for one batch input")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--items-out", required=True)
    parser.add_argument("--run-meta-out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=25)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    start_urls = payload.get("startUrls", [])
    urls = [row.get("url", "").strip() for row in start_urls if isinstance(row, dict) and row.get("url")]
    if not urls:
        raise SystemExit("input startUrls is empty")

    items: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(fetch_one, url, args.timeout): url for url in urls}
        for future in as_completed(future_map):
            item = future.result()
            items.append(item)
            if item.get("text"):
                ok_count += 1
            else:
                error_count += 1

    items.sort(key=lambda row: row.get("url", ""))
    write_json(Path(args.items_out), items)
    write_json(
        Path(args.run_meta_out),
        {
            "input_json": args.input_json,
            "workers": args.workers,
            "timeout": args.timeout,
            "total_urls": len(urls),
            "items_count": len(items),
            "ok_count": ok_count,
            "error_count": error_count,
            "generated_at": now_iso(),
            "mode": "local_fallback",
        },
    )
    print(
        json.dumps(
            {
                "total_urls": len(urls),
                "items_count": len(items),
                "ok_count": ok_count,
                "error_count": error_count,
                "items_out": args.items_out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
