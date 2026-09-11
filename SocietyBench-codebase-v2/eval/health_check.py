#!/usr/bin/env python3
"""LLM health check — verify DMXAPI / OpenAI-compatible endpoint works.

Run this before launching any long pipeline to confirm:
  - DMXAPI_KEY env var is set
  - DMXAPI_BASE_URL is reachable
  - The configured default model returns a parseable JSON response

Usage:
    python health_check.py                 # uses pipeline_config.default_llm_model
    python health_check.py --model kimi-k2.5
    python health_check.py --pipeline-config /path/to/pipeline_config.json
"""
from __future__ import annotations

import argparse
import json
import sys

from common import call_llm, load_pipeline_config


def main() -> int:
    p = argparse.ArgumentParser(description="DMXAPI / OpenAI-compatible endpoint health check")
    p.add_argument("--model", default=None,
                   help="Model to test; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")
    if not model:
        print("[health] no model resolved (set --model or pipeline_config.default_llm_model)")
        return 2

    try:
        raw = call_llm(
            [
                {"role": "system", "content": "仅返回严格 JSON，不要任何额外文字或 markdown 标记。"},
                {"role": "user", "content": '返回：{"ping": "pong", "model": "' + model + '"}'},
            ],
            model=model,
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="none",
        )
    except Exception as e:
        print(f"[health] FAIL: LLM call raised {type(e).__name__}: {e}")
        return 1

    raw_stripped = raw.strip().lstrip("`").rstrip("`")
    if raw_stripped.startswith("json"):
        raw_stripped = raw_stripped[4:].strip()
    try:
        parsed = json.loads(raw_stripped)
    except json.JSONDecodeError:
        print(f"[health] PARTIAL: got a response but not valid JSON. content={raw[:200]!r}")
        return 1

    print(f"[health] OK — model={model}")
    print(f"[health] content={parsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
