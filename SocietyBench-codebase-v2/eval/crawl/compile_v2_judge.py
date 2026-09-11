#!/usr/bin/env python3
"""compile_v2_judge.py — Step 4a: rule-based matching (1:1 of skills/web-search-crawl/step4-validation/SKILL.md §4.1).

Rule-based first-layer filter. SKILL filter logic (line 56-60):
  1. title+desc+URL matches any NOT_RELATED pattern → v2_related = False
  2. else, title+desc matches any RELATED_MARKERS → v2_related = True
  3. else, search keyword contains core term AND desc non-empty → v2_related = True
  4. else → v2_related = False

Per SKILL: expected retention ~60-80%.

Input  : <output_dir>/<project>_full.json
Output : <output_dir>/<project>_v2_judged.json (adds v2_related field)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.stdout.reconfigure(encoding="utf-8")


# SKILL §4.1: generic SEO spam sites / porn sites / scholar index / crypto-exchange ads etc.
DEFAULT_NOT_RELATED = [
    r"ahgwyw\.org",
    r"91cg|91天美|曰批免费",
    r"scholar\.google\.",
    r"coinbase|crypto-?\w*\.com",
    r"/search\b|/discover/",
    r"\.pdf$",  # PDF often noise indexes
]


def load_patterns(path: Path | None, defaults: List[str]) -> List[re.Pattern]:
    out = list(defaults)
    if path and path.exists():
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(extra, list):
                out.extend(extra)
        except Exception:
            pass
    return [re.compile(p, re.I) for p in out]


def load_keywords(path: Path | None) -> List[str]:
    """Extract core terms from a *_keywords.json (Step 1 output)."""
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cores: List[str] = []
    # `keywords` field is `intext:"X"` form; strip
    for kw in data.get("keywords", []):
        m = re.search(r'"([^"]+)"', kw)
        if m:
            cores.append(m.group(1))
    return cores


def judge_one(rec: Dict[str, Any], not_related_pats: List[re.Pattern],
              related_markers: List[re.Pattern], core_terms: List[str]) -> bool:
    title = rec.get("title") or ""
    desc = rec.get("description") or ""
    url = rec.get("url") or ""
    kw = rec.get("keyword") or ""
    haystack_full = f"{title} {desc} {url}"

    for pat in not_related_pats:
        if pat.search(haystack_full):
            return False

    for marker in related_markers:
        if marker.search(title) or marker.search(desc):
            return True

    if desc.strip() and any(term in kw for term in core_terms):
        return True

    return False


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step4a: v2 rule-based filter")
    p.add_argument("--input", required=True, help="{project}_full.json from search_bulk.py")
    p.add_argument("--output", required=True, help="{project}_v2_judged.json")
    p.add_argument("--keywords-json", default=None,
                   help="{project}_keywords.json from Step 1 (for core term positive matches)")
    p.add_argument("--not-related-extras", default=None,
                   help="optional JSON array of extra NOT_RELATED regex patterns")
    p.add_argument("--related-markers", default=None,
                   help="optional JSON array of RELATED_MARKERS regex patterns")
    args = p.parse_args()

    not_related_pats = load_patterns(Path(args.not_related_extras) if args.not_related_extras else None,
                                      DEFAULT_NOT_RELATED)
    related_marker_pats = load_patterns(Path(args.related_markers) if args.related_markers else None, [])
    core_terms = load_keywords(Path(args.keywords_json) if args.keywords_json else None)

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = raw.get("all_results", []) if isinstance(raw, dict) else raw

    kept = 0
    out_records: List[Dict[str, Any]] = []
    for rec in records:
        ok = judge_one(rec, not_related_pats, related_marker_pats, core_terms)
        rec2 = {**rec, "v2_related": bool(ok)}
        if ok:
            kept += 1
        out_records.append(rec2)

    output = {
        "project": raw.get("project") if isinstance(raw, dict) else None,
        "total": len(out_records),
        "v2_related_true": kept,
        "v2_retention_pct": round(100 * kept / max(1, len(out_records)), 2),
        "all_results": out_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(out_records),
                      "v2_related_true": kept,
                      "retention_pct": output["v2_retention_pct"],
                      "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
