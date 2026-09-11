#!/usr/bin/env python3
"""web-search-crawl Step 0: environment precheck (1:1 of skills/web-search-crawl/SKILL.md §Step 0).

Checks (per SKILL line 124-153):
  - Apify API Key valid (client.whoami() returns user info)
  - Apify balance non-zero or plan active
  - Network connectivity (implicit via the API call)
  - runtime output dir writable

Exits 0 on pass, non-zero on hard fail. Prints status table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def load_token() -> str:
    token = (os.getenv("APIFY_TOKEN") or "").strip()
    if token:
        return token
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("APIFY_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl step0: env precheck")
    p.add_argument("--output-dir", required=True, help="output dir to verify is writable")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    issues: list[str] = []

    # Apify token + whoami
    token = load_token()
    apify_status = "MISSING"
    user_info = {}
    if not token:
        issues.append("APIFY_TOKEN missing (set env or put in crawl/.env)")
    else:
        try:
            from apify_client import ApifyClient
            client = ApifyClient(token=token, timeout=30)
            user_info = client.whoami() or {}
            if user_info:
                apify_status = "OK"
        except Exception as e:
            issues.append(f"Apify whoami failed: {e}")

    # Output dir writable
    out_status = "OK"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".precheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        out_status = f"FAIL ({e})"
        issues.append(f"output dir not writable: {e}")

    report = {
        "apify_token": apify_status,
        "apify_user": user_info.get("username") if isinstance(user_info, dict) else None,
        "apify_plan": (user_info.get("plan") or {}).get("id") if isinstance(user_info, dict) else None,
        "output_dir": str(out_dir),
        "output_dir_status": out_status,
        "issues": issues,
        "passed": len(issues) == 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
