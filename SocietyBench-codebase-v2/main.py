#!/usr/bin/env python3
"""SocietyBench — one-line entry point.

Two modes:

1. Reproduce paper main result (one event, no crawling) — uses the released
   HuggingFace dataset ``Social-AI-2026/SocietyBench``::

       python3 main.py --reproduce event3_tiktok /path/to/workspace

   This downloads one event's anonymized timeline + question bank + GT from
   HuggingFace and runs the calibration + temporal evaluation end-to-end.

2. Build a new event from scratch — crawl → process → anonymize → eval::

       python3 main.py "<topic>" /path/to/workspace --replacements-json reps.json

   Thin wrapper around ``eval/total_pipeline.py``; every CLI flag is forwarded
   unchanged. For the full set of flags::

       python3 eval/total_pipeline.py --help

You can also just download the data without running the eval::

    python3 main.py --fetch-data event3_tiktok /path/to/workspace

See also:
    README.md                      — quick start
    docs/case_studies.md           — running on a new event
    docs/pipeline_architecture.md  — data flow, config keys, resume
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ENV_FILE = THIS_DIR / "eval" / ".env"
EXAMPLE_ENV = THIS_DIR / "eval" / "config.example.env"
TOTAL_PIPELINE = THIS_DIR / "eval" / "total_pipeline.py"
PREDICT_PIPELINE = THIS_DIR / "eval" / "predict_pipeline.py"

HF_REPO_ID = "Social-AI-2026/SocietyBench"
KNOWN_EVENTS = [
    "event1_library",
    "event2_trump_tariff",
    "event3_tiktok",
    "event4_us_iran",
    "event5_smci",
]
SUPPORTED_LANGS = ("zh", "en")


def _print_help() -> None:
    print(__doc__)
    print("Common shapes:")
    print("  python3 main.py --reproduce event3_tiktok /path/to/workspace")
    print("  python3 main.py --fetch-data event3_tiktok /path/to/workspace [--lang zh|en]")
    print("  python3 main.py \"<topic>\" /path/to/workspace --replacements-json reps.json")
    print("  python3 main.py \"<topic>\" /path/to/workspace --replacements-json reps.json --start-phase 1")
    print()
    print(f"Known events: {', '.join(KNOWN_EVENTS)}")
    print()


def _preflight() -> None:
    """Friendly warnings before launching a long pipeline."""
    if not ENV_FILE.exists() and not os.environ.get("DMXAPI_KEY"):
        print("[main.py] ⚠ eval/.env not found and DMXAPI_KEY is not set.", file=sys.stderr)
        print("[main.py]   LLM calls will fail. Set up credentials first:", file=sys.stderr)
        print(f"[main.py]     cp {EXAMPLE_ENV.relative_to(THIS_DIR)} {ENV_FILE.relative_to(THIS_DIR)}", file=sys.stderr)
        print(f"[main.py]   then fill in DMXAPI_KEY and DMXAPI_BASE_URL.\n", file=sys.stderr)


def _fetch_data(event: str, lang: str, workspace_root: Path) -> Path:
    """Download one event from HF into <workspace_root>/<event>_workspace/predict_workspace.

    Returns the predict workspace path, populated with everything
    predict_pipeline needs from --start-step 3 onward.
    """
    if event not in KNOWN_EVENTS:
        sys.exit(f"[main.py] unknown event '{event}'. Known: {', '.join(KNOWN_EVENTS)}")
    if lang not in SUPPORTED_LANGS:
        sys.exit(f"[main.py] unknown lang '{lang}'. Use one of: {', '.join(SUPPORTED_LANGS)}")

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.exit(
            "[main.py] huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub\n"
            "Then log in (the dataset is private at submission time):\n"
            "  hf auth login   # paste your HF token"
        )

    target = workspace_root / f"{event}_workspace" / "predict_workspace"
    cache_dir = workspace_root / ".hf_cache"
    target.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main.py] downloading {HF_REPO_ID}:{event}/{lang}/ → {target}")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{event}/{lang}/**"],
            local_dir=str(cache_dir),
        )
    except (GatedRepoError, RepositoryNotFoundError) as e:
        sys.exit(
            f"[main.py] could not access {HF_REPO_ID} ({e.__class__.__name__}).\n"
            f"  The dataset is private. Log in first:\n"
            f"    hf auth login   # paste your HF token with read access\n"
            f"  Or set HF_TOKEN in the environment."
        )

    src = cache_dir / event / lang
    if not src.exists():
        sys.exit(f"[main.py] expected {src} after download, but nothing arrived")

    for item in src.iterdir():
        dst = target / item.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    print(f"[main.py] ✓ data ready at {target}")
    return target


def _parse_event_workspace(argv: list[str], flag: str) -> tuple[str, Path, str]:
    """Pull <event> and <workspace> after `flag`, plus optional --lang."""
    i = argv.index(flag)
    if i + 2 >= len(argv):
        sys.exit(f"[main.py] usage: {flag} <event_name> <workspace> [--lang zh|en]")
    event = argv[i + 1]
    workspace = Path(argv[i + 2]).expanduser().resolve()
    lang = "zh"
    if "--lang" in argv:
        j = argv.index("--lang")
        if j + 1 < len(argv):
            lang = argv[j + 1]
    return event, workspace, lang


def _handle_special_flags(argv: list[str]) -> int | None:
    """Return an exit code if --fetch-data or --reproduce was handled, else None."""
    if "--fetch-data" in argv:
        event, workspace, lang = _parse_event_workspace(argv, "--fetch-data")
        target = _fetch_data(event, lang, workspace)
        print()
        print("[main.py] next step — run the evaluation:")
        print(f"  python3 eval/predict_pipeline.py \\")
        print(f"      {target}/timeline_anon.md \\")
        print(f"      {target} \\")
        print(f"      --event-name {event} \\")
        print(f"      --replacements-json {target}/replacements.json \\")
        print(f"      --start-step 3 --points all")
        return 0

    if "--reproduce" in argv:
        event, workspace, lang = _parse_event_workspace(argv, "--reproduce")
        target = _fetch_data(event, lang, workspace)
        _preflight()
        print(f"\n[main.py] starting evaluation for {event}...")
        cmd = [
            sys.executable,
            str(PREDICT_PIPELINE),
            str(target / "timeline_anon.md"),
            str(target),
            "--event-name", event,
            "--replacements-json", str(target / "replacements.json"),
            "--start-step", "3",
            "--points", "all",
        ]
        return subprocess.run(cmd).returncode

    return None


def main() -> int:
    if len(sys.argv) < 2:
        _print_help()
        return 2
    if sys.argv[1] in ("-h", "--help"):
        _print_help()
        return 0

    rc = _handle_special_flags(sys.argv[1:])
    if rc is not None:
        return rc

    if not TOTAL_PIPELINE.exists():
        print(f"[main.py] FATAL: {TOTAL_PIPELINE} not found.", file=sys.stderr)
        print("[main.py]   Are you running this from the repo root?", file=sys.stderr)
        return 2

    _preflight()
    cmd = [sys.executable, str(TOTAL_PIPELINE), *sys.argv[1:]]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
