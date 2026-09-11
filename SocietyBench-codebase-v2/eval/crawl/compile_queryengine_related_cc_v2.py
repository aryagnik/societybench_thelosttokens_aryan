#!/usr/bin/env python3
"""compile_queryengine_related_cc_v2.py — Step 5b (1:1 of skills/web-search-crawl/step5-qe-filter/SKILL.md §5.2).

Per-URL QueryEngine classification by domain + path scoring.

Scoring (per SKILL §5.2):
  TOPIC_POSITIVE         +2/each, cap +8
  NEWS_DOMAIN_POS        +1/each, cap +4
  ARTICLE_PATH_POS       +1/each, cap +2
  has 'date' field       +1
  VIDEO_OR_MEDIA_DOMAINS -10 (hard exclude)
  SOCIAL_FORUM_DOMAINS   -3/each
  VIDEO_PATH_HINTS       -10

queryengine_related = (score >= 1) AND (not hard_negative)

Input  : <output_dir>/content_enrich/v23_intersection_records.csv +
         <project>_v3_judged.json (for full record fields)
Output : <output_dir>/content_enrich/v23_intersection_records_qe_labeled.csv (+ qe field)
         <output_dir>/content_enrich/v23_qe_passed_unique_urls.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")


NEWS_DOMAIN_POS = [
    r"news\.", r"finance\.",
    r"sina|sohu|163\.com|qq\.com|ifeng",
    r"guancha|globaltimes|zaobao|hk01|news\.cn|xinhuanet",
    r"bbc|reuters|apnews|nytimes|theguardian|cnn",
    r"thepaper|bjnews|caixin",
]
ARTICLE_PATH_POS = [
    r"/\d{4}[-_/]\d{2}[-_/]\d{2}",
    r"/article", r"/news", r"/detail", r"/p/", r"/a/",
]
VIDEO_OR_MEDIA_DOMAINS = [
    r"youtube\.com", r"youtu\.be",
    r"bilibili\.com", r"douyin\.com", r"tiktok\.com", r"vimeo\.com",
]
SOCIAL_FORUM_DOMAINS = [
    r"facebook\.com", r"instagram\.com", r"x\.com", r"twitter\.com", r"weibo\.com",
    r"reddit\.com", r"zhihu\.com", r"tieba\.baidu\.com",
]
VIDEO_PATH_HINTS = [r"/video/", r"/watch", r"/shorts/", r"/reel/"]


def score_url(url: str, has_date: bool, topic_positive: List[str]) -> tuple[int, bool, str]:
    score = 0
    url_l = url.lower()
    hard_neg = False
    notes: List[str] = []

    for kw in topic_positive[:4]:  # cap +8 = 4 × 2
        if kw and kw.lower() in url_l:
            score += 2
            notes.append(f"+2:topic({kw})")

    hits = 0
    for pat in NEWS_DOMAIN_POS:
        if re.search(pat, url_l):
            hits += 1
            if hits <= 4:
                score += 1
                notes.append(f"+1:newsdom({pat[:15]})")
    hits = 0
    for pat in ARTICLE_PATH_POS:
        if re.search(pat, url_l):
            hits += 1
            if hits <= 2:
                score += 1
                notes.append(f"+1:articlepath")
    if has_date:
        score += 1
        notes.append("+1:hasdate")

    for pat in VIDEO_OR_MEDIA_DOMAINS:
        if re.search(pat, url_l):
            score -= 10
            hard_neg = True
            notes.append(f"-10:video_dom")
            break
    for pat in SOCIAL_FORUM_DOMAINS:
        if re.search(pat, url_l):
            score -= 3
            notes.append(f"-3:social({pat[:15]})")
    for pat in VIDEO_PATH_HINTS:
        if re.search(pat, url_l):
            score -= 10
            hard_neg = True
            notes.append("-10:video_path")
            break

    return score, hard_neg, "; ".join(notes)


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step5b: QueryEngine domain classification")
    p.add_argument("--records-csv", required=True,
                   help="v23_intersection_records.csv from step5a")
    p.add_argument("--input-judged", required=True, help="{project}_v3_judged.json (for has_date)")
    p.add_argument("--keywords-json", default=None, help="{project}_keywords.json for TOPIC_POSITIVE")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    topic_positive: List[str] = []
    if args.keywords_json and Path(args.keywords_json).exists():
        kdata = json.loads(Path(args.keywords_json).read_text(encoding="utf-8"))
        for kw in kdata.get("keywords", []):
            m = re.search(r'"([^"]+)"', kw)
            if m:
                topic_positive.append(m.group(1))

    # Build URL → has_date map from judged json
    judged = json.loads(Path(args.input_judged).read_text(encoding="utf-8"))
    judged_recs = judged.get("all_results", []) if isinstance(judged, dict) else judged
    url_has_date: Dict[str, bool] = {}
    for r in judged_recs:
        url_has_date[r.get("url", "")] = bool(r.get("date"))

    out_dir = Path(args.output_dir) / "content_enrich"
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "v23_intersection_records_qe_labeled.csv"
    passed_url_path = out_dir / "v23_qe_passed_unique_urls.csv"

    total = passed = 0
    seen_urls = set()
    passed_urls: List[Dict[str, Any]] = []
    with Path(args.records_csv).open("r", encoding="utf-8") as fh, \
         labeled_path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(fh)
        writer = csv.writer(out_fh)
        writer.writerow(["record_id", "url", "url_norm", "title", "date", "qe", "qe_score", "qe_notes"])
        for row in reader:
            url = row.get("url", "")
            url_norm = row.get("url_norm", "")
            has_date = url_has_date.get(url, bool(row.get("date")))
            score, hard_neg, notes = score_url(url_norm or url, has_date, topic_positive)
            ok = (score >= 1) and not hard_neg
            writer.writerow([row.get("record_id"), url, url_norm, row.get("title"),
                             row.get("date"), int(ok), score, notes])
            total += 1
            if ok and url_norm and url_norm not in seen_urls:
                passed += 1
                seen_urls.add(url_norm)
                passed_urls.append({"url_norm": url_norm, "url": url,
                                    "title": row.get("title"), "date": row.get("date"),
                                    "qe_score": score})

    with passed_url_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url_norm", "url", "title", "date", "qe_score"])
        for r in passed_urls:
            writer.writerow([r["url_norm"], r["url"], r["title"], r["date"], r["qe_score"]])

    print(json.dumps({
        "total_records": total,
        "qe_passed_records": passed,
        "qe_passed_unique_urls": len(passed_urls),
        "labeled_csv": str(labeled_path),
        "passed_urls_csv": str(passed_url_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
