#!/usr/bin/env python3
"""media-crawl (1:1 of skills/media-crawl/SKILL.md) — thin wrapper.

Per the SKILL, the actual 4-step social-media crawl (platform availability
check, keyword generation, full crawl, validation) is executed by external
tools — primarily MediaCrawlerPro-Python with MediaCrawlerPro-SignSrv. This
script is a thin wrapper that:

  1. Validates that MCP_HOME points to a usable MediaCrawlerPro install
  2. Loads a keywords config (--keywords-config) or accepts a topic and
     defers keyword generation to the external tool
  3. Forwards platforms + keywords to the external CLI

Env vars (set in your shell or `crawl/.env`):
  MCP_HOME              path to MediaCrawlerPro-Python install
  MCP_SIGNSRV_URL       (optional) signsrv endpoint

Default platforms: dy,bili,wb,zhihu,tieba,xhs,ks (7 platforms per SKILL).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_PLATFORMS = "dy,bili,wb,zhihu,tieba,xhs,ks"


def resolve_mcp_home() -> Path:
    home = (os.environ.get("MCP_HOME") or "").strip()
    if not home:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MCP_HOME="):
                    home = line.split("=", 1)[1].strip()
                    break
    if not home:
        sys.exit("MCP_HOME is not set. Point it at your MediaCrawlerPro-Python install (see config.example.env).")
    p = Path(home)
    if not p.exists() or not (p / "main.py").exists():
        sys.exit(f"MCP_HOME={home} does not contain main.py — is the path correct?")
    return p


def keywords_from_config(path: Path) -> List[str]:
    """Optional keywords JSON: {"keywords": ["...", "..."]} (no LLM call here)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("keywords", []))
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def main() -> None:
    p = argparse.ArgumentParser(description="media-crawl: forward to MediaCrawlerPro-Python")
    p.add_argument("--topic", default=None, help="topic string used as keywords source if --keywords-config not given")
    p.add_argument("--keywords-config", default=None, help="JSON with explicit keyword list")
    p.add_argument("--platforms", default=DEFAULT_PLATFORMS,
                   help=f"comma-separated platform codes (default: {DEFAULT_PLATFORMS})")
    p.add_argument("--extra-args", default="",
                   help="extra CLI args to pass through to MediaCrawlerPro main.py")
    args = p.parse_args()

    mcp = resolve_mcp_home()
    if args.keywords_config:
        kws = keywords_from_config(Path(args.keywords_config))
        if not kws:
            sys.exit(f"No keywords in {args.keywords_config}")
        keyword_arg = ",".join(kws)
    elif args.topic:
        keyword_arg = args.topic
    else:
        sys.exit("Provide either --topic or --keywords-config")

    cmd = [sys.executable, "main.py", "--keywords", keyword_arg, "--platforms", args.platforms]
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    print(f"[media-crawl] cwd={mcp} cmd={' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(mcp))


if __name__ == "__main__":
    main()
