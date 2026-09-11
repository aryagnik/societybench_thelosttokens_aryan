#!/usr/bin/env python3
"""web_search_crawl_pipeline.py — orchestrator for skills/web-search-crawl/SKILL.md.

Runs Step 0 → 7 with the breakpoint-recovery contract in SKILL §"breakpoint recovery":

  {project}_enriched.json exists   → Step 6 done → go to Step 7
  {project}_v3_judged.json exists  → resume from Step 5
  {project}_v2_judged.json exists  → resume from Step 4b
  {project}_full.json exists       → resume from Step 4
  {project}_progress.json exists   → Step 3 in progress → resume it
  none of the above                → start from Step 0

Hard-stop circuit breakers (SKILL §"abnormal circuit-breaker rules"):
  - Step 3 raw records = 0            → stop
  - Step 4 combined v2/v3 retention < 30% → stop
  - Step 5 fewer than 10 items after QE   → warn
  - Step 6 enriched_pct < 50%             → stop
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

D = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")


def run(name: str, cmd: List[str]) -> int:
    print(f"\n========== [{name}] {' '.join(cmd)} ==========", flush=True)
    return subprocess.run(cmd).returncode


def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def maybe_load_json(p: Path) -> Dict[str, Any] | None:
    if not file_ok(p):
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="web-search-crawl orchestrator (Step 0→7)")
    p.add_argument("--project", required=True, help="project slug for file naming")
    p.add_argument("--topic", required=True, help="one-line event description")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--event-dates", default=None, help="comma-separated event dates (yyyy-mm-dd)")
    p.add_argument("--event-dates-json", default=None, help="JSON file of event dates")
    p.add_argument("--keywords-config", default=None,
                   help="optional pre-built {project}_keywords.json to skip Step 1 LLM call")
    p.add_argument("--max-keywords", type=int, default=10)
    p.add_argument("--max-final-urls", type=int, default=2000)
    p.add_argument("--min-content-chars", type=int, default=0)
    p.add_argument("--max-concurrent-search", type=int, default=40)
    p.add_argument("--skip-step6", action="store_true",
                   help="skip the Apify content-crawler call (use when crawler already ran out-of-band)")
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    a = p.parse_args()

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enrich_dir = out_dir / "content_enrich"
    py = sys.executable

    common_args = []
    if a.model:
        common_args += ["--model", a.model]
    if a.pipeline_config:
        common_args += ["--pipeline-config", a.pipeline_config]

    full_json = out_dir / f"{a.project}_full.json"
    v2_json = out_dir / f"{a.project}_v2_judged.json"
    v3_json = out_dir / f"{a.project}_v3_judged.json"
    enriched_json = out_dir / f"{a.project}_enriched.json"
    keywords_json = out_dir / f"{a.project}_keywords.json"
    segments_json = out_dir / f"{a.project}_segments.json"

    # ───── Step 0: env precheck ─────
    if run("step0-env", [py, str(D / "env_precheck.py"), "--output-dir", str(out_dir)]) != 0:
        sys.exit("Step 0 env precheck failed — see report above")

    # ───── Step 1: keywords ─────
    if a.keywords_config and Path(a.keywords_config).exists():
        Path(a.keywords_config).resolve()
        # User-supplied config; copy to expected location if different
        src = Path(a.keywords_config)
        if src.resolve() != keywords_json.resolve():
            keywords_json.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[step1] using supplied keywords config: {keywords_json}")
    elif file_ok(keywords_json):
        print(f"[step1] {keywords_json} already exists — skipping")
    else:
        if run("step1-keywords", [py, str(D / "keyword_optimize.py"),
                                   "--topic", a.topic, "--project", a.project,
                                   "--output-dir", str(out_dir),
                                   "--max-keywords", str(a.max_keywords),
                                   *common_args]) != 0:
            sys.exit("Step 1 keyword generation failed")

    # ───── Step 2: segments ─────
    if file_ok(segments_json):
        print(f"[step2] {segments_json} already exists — skipping")
    else:
        if not a.event_dates and not a.event_dates_json:
            print("[step2] WARNING: no --event-dates or --event-dates-json supplied; "
                  "using a single segment 'topic-recent-1y' fallback")
            from datetime import date, timedelta
            today = date.today().isoformat()
            year_ago = (date.today() - timedelta(days=365)).isoformat()
            segments_json.write_text(json.dumps({
                "project": a.project,
                "event_dates": [],
                "segments": [{"start": year_ago, "end": today, "period_days": 30,
                              "period_count": 12, "type": "fallback_yearly"}],
                "summary": {"segment_count": 1, "total_rounds": 12,
                            "earliest": year_ago, "latest": today},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            cmd = [py, str(D / "time_periods.py"),
                   "--project", a.project, "--output-dir", str(out_dir)]
            if a.event_dates:
                cmd += ["--event-dates", a.event_dates]
            if a.event_dates_json:
                cmd += ["--event-dates-json", a.event_dates_json]
            if run("step2-segments", cmd) != 0:
                sys.exit("Step 2 segments failed")

    # Build search config for Step 3
    kdata = maybe_load_json(keywords_json) or {}
    sdata = maybe_load_json(segments_json) or {}
    search_config_path = out_dir / f"{a.project}_search_config.json"
    search_config_path.write_text(json.dumps({
        "project": a.project,
        "keywords": kdata.get("keywords", []),
        "segments": sdata.get("segments", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ───── Step 3: search_bulk ─────
    if file_ok(full_json):
        # circuit breaker: zero results
        fdata = maybe_load_json(full_json) or {}
        if (fdata.get("total_results") or 0) == 0 and not fdata.get("all_results"):
            sys.exit("[circuit-breaker] Step 3 produced 0 results — check keywords / segments")
        print(f"[step3] {full_json} already exists — skipping")
    else:
        if run("step3-search", [py, str(D / "search_bulk.py"),
                                 "--config", str(search_config_path),
                                 "--output-dir", str(out_dir),
                                 "--max-concurrent", str(a.max_concurrent_search)]) != 0:
            sys.exit("Step 3 search_bulk failed")
        fdata = maybe_load_json(full_json) or {}
        if not fdata.get("all_results"):
            sys.exit("[circuit-breaker] Step 3 produced 0 records")

    # ───── Step 4a: v2 ─────
    if file_ok(v2_json):
        print(f"[step4a] {v2_json} already exists — skipping")
    else:
        if run("step4a-v2", [py, str(D / "compile_v2_judge.py"),
                              "--input", str(full_json),
                              "--output", str(v2_json),
                              "--keywords-json", str(keywords_json)]) != 0:
            sys.exit("Step 4a v2 filter failed")

    # ───── Step 4b: v3 ─────
    if file_ok(v3_json):
        print(f"[step4b] {v3_json} already exists — skipping")
    else:
        if run("step4b-v3", [py, str(D / "compile_v3_related.py"),
                              "--input", str(v2_json),
                              "--output", str(v3_json),
                              "--topic", a.topic,
                              *common_args]) != 0:
            sys.exit("Step 4b v3 LLM check failed")

    # SKILL circuit-breaker rule: Step 4 retention <30% → stop
    v3data = maybe_load_json(v3_json) or {}
    v3_pass = v3data.get("v3_related_true", 0)
    if v3data.get("total", 0) > 0 and v3_pass / v3data["total"] < 0.30:
        sys.exit(f"[circuit-breaker] Step 4 combined v2∩v3 retention "
                 f"{round(100 * v3_pass / v3data['total'], 1)}% < 30% — "
                 "停止等待用户决策（SKILL §异常熔断规则）")

    # ───── Step 5a: intersect + dedup ─────
    if not (enrich_dir / "v23_intersection_records.csv").exists():
        if run("step5a-prepare", [py, str(D / "prepare_content_enrich_plan.py"),
                                   "--input", str(v3_json),
                                   "--output-dir", str(out_dir),
                                   "--segments-json", str(segments_json)]) != 0:
            sys.exit("Step 5a failed")
    else:
        print(f"[step5a] {enrich_dir / 'v23_intersection_records.csv'} exists — skipping")

    # ───── Step 5b: QE classification ─────
    if not (enrich_dir / "v23_intersection_records_qe_labeled.csv").exists():
        if run("step5b-qe", [py, str(D / "compile_queryengine_related_cc_v2.py"),
                              "--records-csv", str(enrich_dir / "v23_intersection_records.csv"),
                              "--input-judged", str(v3_json),
                              "--keywords-json", str(keywords_json),
                              "--output-dir", str(out_dir)]) != 0:
            sys.exit("Step 5b QE failed")
    else:
        print(f"[step5b] qe_labeled.csv exists — skipping")

    # SKILL circuit-breaker rule: fewer than 10 items after Step 5 QE filter → stop and wait for user decision
    qe_passed_csv = enrich_dir / "v23_qe_passed_unique_urls.csv"
    qe_passed_count = sum(1 for _ in qe_passed_csv.open(encoding="utf-8")) - 1 if qe_passed_csv.exists() else 0
    if qe_passed_count < 10:
        sys.exit(f"[circuit-breaker] QE passed only {qe_passed_count} URLs (<10) — "
                 "数据可能不足以支撑分析，停止等待用户决策（SKILL §异常熔断规则）")

    # ───── Step 5c: <2000 convergence ─────
    final_urls_json = enrich_dir / "final_crawl_urls.json"
    if not final_urls_json.exists():
        if run("step5c-final", [py, str(D / "finalize_content_crawler_input_under2000.py"),
                                 "--qe-passed-csv", str(qe_passed_csv),
                                 "--output-dir", str(out_dir),
                                 "--segments-json", str(segments_json),
                                 "--keywords-json", str(keywords_json),
                                 "--max-urls", str(a.max_final_urls)]) != 0:
            sys.exit("Step 5c convergence failed")
    else:
        print(f"[step5c] final_crawl_urls.json exists — skipping")

    # ───── Step 6: content crawl ─────
    if file_ok(enriched_json):
        print(f"[step6] {enriched_json} already exists — skipping crawl + backfill")
    else:
        if not a.skip_step6:
            # The existing run_content_crawler_batch.py expects its own argv; we
            # rely on the user to have configured Apify. If unavailable, --skip-step6
            # lets the pipeline skip and use whatever items are already in
            # content_enrich/ for backfill.
            crawl_cmd = [py, str(D / "run_content_crawler_batch.py"),
                         "--input", str(final_urls_json),
                         "--output-dir", str(enrich_dir)]
            rc = subprocess.run(crawl_cmd).returncode
            if rc != 0:
                print(f"[step6] crawler returned {rc}; continuing to backfill with whatever items exist")
        # backfill regardless
        if run("step6-backfill", [py, str(D / "merge_content_enrich_backfill.py"),
                                   "--input-judged", str(v3_json),
                                   "--content-enrich-dir", str(enrich_dir),
                                   "--output", str(enriched_json)]) != 0:
            sys.exit("Step 6 backfill failed")
        # SKILL circuit-breaker rule: Step 6 crawl success rate < 50% → stop
        edata = maybe_load_json(enriched_json) or {}
        if edata.get("enriched_pct", 0) < 50:
            sys.exit(f"[circuit-breaker] enriched_pct {edata.get('enriched_pct')}% < 50% — "
                     "停止等待用户决策（SKILL §异常熔断规则）")

    # ───── Step 7: final output ─────
    if run("step7-output", [py, str(D / "output_final.py"),
                             "--input", str(enriched_json),
                             "--output-dir", str(out_dir),
                             "--min-content-chars", str(a.min_content_chars),
                             "--out-name", "articles.json"]) != 0:
        sys.exit("Step 7 final output failed")

    print(f"\n✅ web-search-crawl 完成 — output: {out_dir / 'articles.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
