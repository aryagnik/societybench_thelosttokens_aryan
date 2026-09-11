#!/usr/bin/env python3
"""compile_v3_related.py — Step 4b: CC semantic re-check (1:1 of skills/web-search-crawl/step4-validation/SKILL.md §4.2).

LLM-based second-layer semantic relevance check. Per SKILL §4.0: only
records that already passed v2 (v2_related == True) are sent for v3 check.

Prompt template (§4.2 Path B, since Path A "Codex semantic re-check" requires CC
in the loop and is hard to script — we instead drive the same prompt with
the default LLM). The (Chinese) prompt roughly says:

  You are a Chinese public-opinion data annotator. Judge, from title and
  description only, whether each record relates to the target event.
  Target event: {topic}
  Criteria: 1) explicitly discusses the event itself... -> true 2) same-name /
  tangential / academic index -> false
  Output only a JSON array, each item: {"idx": <int>, "v3_related": <bool>, "reason": "<=30 chars"}

SKILL §4.2 batch size: 500 / batch. We follow that.

Input  : <project>_v2_judged.json
Output : <project>_v3_judged.json (adds v3_related field; carries v2_related through)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import call_llm, load_pipeline_config  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


PROMPT_HEADER = """\
你是中文舆情数据标注员。任务：仅根据每条的 title 和 description 判断是否与目标事件相关。

目标事件：
{topic}

判定标准：
1) 明确讨论该事件本身、当事人、判决进展、舆情争议 → v3_related=true
2) 仅同名、关键词擦边、站内搜索页、广告软文、学术索引、词表/数据文件 → v3_related=false
3) 证据不足时，宁可判 false
4) 只允许依据 title 与 description，不得使用其他字段

请只输出 JSON 数组，每项格式：
{{"idx": <int>, "v3_related": <true|false>, "reason": "<=30字>"}}

不要输出任何额外文字。

待判断数据：
{rows}
"""


def judge_batch(topic: str, batch: List[Dict[str, Any]], model: str) -> Dict[int, Dict[str, Any]]:
    rows_json = json.dumps(
        [{"idx": i, "title": r.get("title", ""), "description": r.get("description", "")}
         for i, r in enumerate(batch)],
        ensure_ascii=False,
    )
    prompt = PROMPT_HEADER.format(topic=topic, rows=rows_json)
    try:
        raw = call_llm(
            [{"role": "system", "content": "你是严格的二分类标注助手，只输出 JSON。"},
             {"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=4000,
            reasoning_effort="none",
        )
    except Exception as e:
        print(f"[v3] LLM call failed: {e}", file=sys.stderr)
        return {}
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return {}
        items = json.loads(m.group(0))
    out: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("idx"), int):
            out[it["idx"]] = {
                "v3_related": bool(it.get("v3_related", False)),
                "v3_reason": str(it.get("reason", ""))[:60],
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step4b: v3 LLM semantic check")
    p.add_argument("--input", required=True, help="{project}_v2_judged.json")
    p.add_argument("--output", required=True, help="{project}_v3_judged.json")
    p.add_argument("--topic", required=True, help="event topic description (for prompt)")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = raw.get("all_results", []) if isinstance(raw, dict) else raw

    # Only v2-passing rows are sent to LLM; others default v3_related=False
    todo_idx = [i for i, r in enumerate(records) if r.get("v2_related")]
    todo_records = [records[i] for i in todo_idx]
    print(f"[v3] {len(todo_records)}/{len(records)} records to judge (v2-passing only)", file=sys.stderr)

    judgments: Dict[int, Dict[str, Any]] = {}
    for b_start in range(0, len(todo_records), args.batch_size):
        batch = todo_records[b_start:b_start + args.batch_size]
        result = judge_batch(args.topic, batch, model)
        # Re-key by global index
        for local_idx, val in result.items():
            global_idx = todo_idx[b_start + local_idx]
            judgments[global_idx] = val
        print(f"[v3] judged batch {b_start}-{b_start+len(batch)}; got {len(result)} verdicts", file=sys.stderr)

    out_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        j = judgments.get(i, {"v3_related": False, "v3_reason": "not in v2-pass set"})
        out_records.append({**rec, **j})

    kept = sum(1 for r in out_records if r.get("v3_related"))
    output = {
        "project": raw.get("project") if isinstance(raw, dict) else None,
        "total": len(out_records),
        "v3_related_true": kept,
        "v2_pass_total": len(todo_records),
        "v3_retention_pct": round(100 * kept / max(1, len(out_records)), 2),
        "all_results": out_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "total": len(out_records),
        "v2_pass": len(todo_records),
        "v3_related_true": kept,
        "v3_retention_pct": output["v3_retention_pct"],
        "output": args.output,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
