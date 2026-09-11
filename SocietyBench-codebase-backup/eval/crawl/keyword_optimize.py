#!/usr/bin/env python3
"""web-search-crawl Step 1: keyword preparation (1:1 of skills/web-search-crawl/step1-keywords/SKILL.md).

SKILL §1.1 + §1.2: CC/LLM expands a one-line topic into 40-50 candidate keywords,
then "entity extraction & minimization" reduces to ≤10 (default) or ≤15 (extended) final keywords.

Pragmatic implementation:
  - Round 1: call LLM to generate 40-50 candidate keywords across dimensions
             (core terms / parties involved / nicknames / characterization terms /
             legal terms / English / traditional Chinese / original source)
  - Round 2: extract minimal entity names (substring-containment rule) — if keyword B contains
             keyword A as a substring, B is redundant (intext:B ⊂ intext:A).
             Per language group, keep the shortest covering entity, plus
             entities whose names share no common substring with the root.
  - Output: final keywords list in `intext:"X"` form ready for search_bulk.py.

Input  : topic (one-line description) + optional existing keywords_round1.json
Output : <output_dir>/<project>_keywords.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow importing eval/common.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import call_llm, load_pipeline_config  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


ROUND1_PROMPT = """\
请根据以下事件描述，**生成 40-50 个**适合用于 Google 搜索的中文关键词候选，覆盖以下维度（每个维度 3-8 个）。请基于事件本身的特征推导，**不要照抄示例**：

1. 事件核心词（事件最常见的简称或代号）
2. 当事人真名 / 涉事主体 / 涉事机构
3. 网络昵称 / 别称 / 谐音梗（若存在）
4. 舆情定性词（事件被定性的关键词，如反转/网暴/争议等同类词）
5. 法律 / 司法 / 政策进程词（与该事件相关的程序性术语）
6. 影响 / 延伸领域词（事件波及的领域专用术语）
7. 英文翻译 / 国际报道用词
8. 繁体变体 / 其他常见拼写变体

要求：
- 直接输出 JSON 数组，每条字段：`{{"keyword": "<词>", "dim": "<维度>", "source": "<简短来源说明>"}}`
- 不要 markdown 代码块、不要解释文字
- 每个 keyword 要可直接作为搜索字符串，不要带 `intext:` 前缀
- **不要返回示例里的占位词；返回与下面 topic 实际匹配的关键词**

事件描述：{topic}
"""


CN_RE = re.compile(r"[一-鿿]+")
EN_RE = re.compile(r"[A-Za-z]+")
TRAD_HINT = set("體<event_location_trad>學廣聲處傳發開時間")  # rough hint chars common in traditional only


def classify_lang(text: str) -> str:
    if EN_RE.fullmatch(re.sub(r"[\s\-]+", "", text)):
        return "en"
    if any(c in text for c in TRAD_HINT) and not any(c in text for c in "体<event_location>学广声处传发开时间"):
        return "trad"
    return "zh"


def round1_generate(topic: str, model: str) -> List[Dict[str, Any]]:
    """Call LLM to produce 40-50 candidate keywords. Falls back to empty list on error."""
    try:
        raw = call_llm(
            [{"role": "user", "content": ROUND1_PROMPT.format(topic=topic)}],
            model=model,
            temperature=0.2,
            max_tokens=2000,
            reasoning_effort="none",
        )
    except Exception as e:
        print(f"[round1] LLM call failed: {e}", file=sys.stderr)
        return []
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return []
        items = json.loads(m.group(0))
    cleaned = []
    for it in items:
        if isinstance(it, dict) and it.get("keyword"):
            cleaned.append({
                "keyword": str(it["keyword"]).strip(),
                "dim": str(it.get("dim", "")),
                "source": str(it.get("source", "")),
            })
    return cleaned


def round2_minimize(candidates: List[Dict[str, Any]], max_keywords: int) -> List[Dict[str, Any]]:
    """SKILL §1.2: substring containment = subset relation. Per-language, keep the shortest covering
    entity. Drop any kw whose text is a superstring of another already-kept kw
    in the same language group."""
    by_lang: Dict[str, List[Dict[str, Any]]] = {"zh": [], "en": [], "trad": []}
    for c in candidates:
        lang = classify_lang(c["keyword"])
        by_lang.setdefault(lang, []).append(c)

    final: List[Dict[str, Any]] = []
    for lang, group in by_lang.items():
        # Sort by length asc — shorter first
        group_sorted = sorted(group, key=lambda x: len(x["keyword"]))
        kept_for_lang: List[str] = []
        for c in group_sorted:
            kw = c["keyword"]
            # If kw is a substring of nothing already kept, AND nothing already
            # kept is a substring of kw → keep.
            # If something already kept (shorter) is a substring of kw → drop kw.
            if any(k in kw for k in kept_for_lang):
                continue
            # If kw is a substring of an already kept (longer)? Not possible
            # since we sorted by length.
            kept_for_lang.append(kw)
            final.append({**c, "lang": lang})

    # Cap per SKILL §1.3 (default ≤10, extended ≤15)
    if len(final) > max_keywords:
        # Prefer keeping shorter (more general) keywords
        final = sorted(final, key=lambda x: len(x["keyword"]))[:max_keywords]
    return final


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step1: keyword optimization")
    p.add_argument("--topic", required=True, help="one-line event description")
    p.add_argument("--project", required=True, help="project slug for output filenames")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-keywords", type=int, default=10,
                   help="SKILL default 10 (basic) or 15 (extended)")
    p.add_argument("--input-round1", default=None,
                   help="optional pre-existing round1 candidates JSON to skip LLM call")
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Round 1
    if args.input_round1 and Path(args.input_round1).exists():
        candidates = json.loads(Path(args.input_round1).read_text(encoding="utf-8"))
    else:
        print(f"[round1] generating candidates via LLM for topic: {args.topic[:60]}", file=sys.stderr)
        candidates = round1_generate(args.topic, model)
        (out_dir / f"{args.project}_keywords_round1.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"[round1] {len(candidates)} candidates", file=sys.stderr)

    # Round 2
    final = round2_minimize(candidates, args.max_keywords)
    print(f"[round2] {len(final)} final keywords (max {args.max_keywords})", file=sys.stderr)

    # Output ready for search_bulk.py: list of `intext:"X"` strings
    output = {
        "project": args.project,
        "topic": args.topic,
        "max_keywords": args.max_keywords,
        "candidates_count": len(candidates),
        "keywords": [f'intext:"{c["keyword"]}"' for c in final],
        "keyword_detail": final,
    }
    out_path = out_dir / f"{args.project}_keywords.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidates": len(candidates),
        "final_keywords": len(final),
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
