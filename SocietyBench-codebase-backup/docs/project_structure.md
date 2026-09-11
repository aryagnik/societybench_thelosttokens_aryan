# Project Structure

A complete file index — what every file does, and where its corresponding spec / output lives.

## Top level

```
code/
├── README.md                          # Main README
├── main.py                            # One-line entry point (wraps eval/total_pipeline.py)
├── requirements.txt                   # Core runtime deps (just `requests`)
├── requirements-crawl.txt             # Optional Phase 0 crawl deps (apify, bs4, trafilatura)
├── eval/                              # Python implementations + helpers
└── docs/                              # Documentation
```

## `eval/` — Python files

```
eval/
├── common.py                          # Shared LLM client (call_llm), config loader, helpers
├── pipeline_config.json               # Runtime parameters (models, thresholds, scoring constants)
├── config.example.env                 # Template — copy to .env and fill in secrets
├── health_check.py                    # Pre-flight: verify LLM endpoint works
│
├── web_pipeline.py                    # Orchestrates web_step1..6
├── web_step1_summarize.py             # Per-article summary (LLM)
├── web_step2_clean.py                 # Strip meta-language (LLM)
├── web_step3_dedup.py                 # Per-day LLM dedup
├── web_step4_extract.py               # "Is there new info?" (LLM)
├── web_step5_short.py                 # Short-form timeline.md
├── web_step6_long.py                  # Long-form (200-300 char) timeline.md
│
├── media_pipeline.py                  # Orchestrates media_step0..5
├── media_step0_aggregate.py           # Aggregate post + top-30 comments
├── media_step1_summarize.py           # Per-post summary (LLM)
├── media_step2_clean.py               # Clean (LLM)
├── media_step3_dedup.py               # Per-day dedup, batch 30 (LLM)
├── media_step4_extract.py             # "Is there new info?" (LLM)
├── media_step5_short.py               # Short-form media timeline
│
├── merge_pipeline.py                  # Orchestrates merge_step5 + 6 + 6.5
├── merge_step5_short.py               # Combine web+media (LLM) with [W]/[M]/[WM] tags
├── merge_step6_long.py                # Expand to 200-300 + attach opinion (LLM)
├── merge_step6_5_gt_refine.py         # 3-round GT refine (LLM)
│
├── predict_pipeline.py                # Orchestrates predict_step0..4
├── predict_step0_prepare.py           # 3-phase anonymization (LLM audit + auto rule derivation)
├── predict_step1_points.py            # Pick ~25 prediction points (LLM quality review)
├── predict_step2_questionbank.py      # A/B/C/D question generation (LLM)
├── predict_step3B_brier.py            # Calibration scoring (calls each evaluation_model)
├── predict_step3F_time.py             # Temporal scoring, 1 run
├── predict_step4_scorecard.py         # Final scorecard JSON per model
├── run_pipeline_parallel.py           # Run step3B + step3F in parallel, then step4
│
├── total_pipeline.py                  # End-to-end: crawl → process → merge → predict
├── pipeline_run.py                    # SKILL-style wrapper around total_pipeline
│
├── crawl/                             # Phase 0 raw collection (optional)
│   ├── web_search_crawl_pipeline.py   # web-search-crawl orchestrator (Step 0→7)
│   ├── web_content_crawler_local.py   # Local crawler (bs4 + trafilatura)
│   ├── media_crawl.py                 # MediaCrawlerPro wrapper
│   ├── apify_client.py                # Thin HTTP client for Apify
│   ├── search_bulk.py                 # Bulk-keyword search driver
│   ├── run_content_crawler_batch.py   # Batch fetcher
│   ├── env_precheck.py                # Verify APIFY_TOKEN
│   ├── keyword_optimize.py            # LLM-driven keyword expansion
│   └── media_scripts/                 # Social-media post validation / export helpers
│
├── agents/                            # Agent baselines (LangGraph / AutoGen / MiroFish)
│   ├── run_agents.py                  # Entry point — same I/O contract as a bare LLM
│   ├── adapter.py                     # Framework adapter protocol
│   ├── fw_langgraph.py / fw_autogen.py / fw_mirofish_oasis.py
│   └── mirofish_sim.py                # Social-simulation forecaster internals
│
└── baseline/                          # Non-LLM baselines
    └── baseline_freq_momentum.py      # Event base-rate + 7-day momentum heuristics
```

Every step script:
- Reads parameters from `pipeline_config.json` (CLI flags override).
- Calls LLM via `common.call_llm()` — no hard-coded model names.
- Writes a `<stem>.log` next to its output.
- Writes a `<stem>_progress.json` snapshot for monitoring.
- Supports resume (skip when output exists, `--force` to override).

## `docs/` — documentation

```
docs/
├── methodology.md            # Framework: anonymization, evaluation axes
├── case_studies.md           # The 5 events used in the paper + how to run a new one
├── pipeline_architecture.md  # Data flow, workspace layout, all config keys
├── scoring.md                # Calibration + temporal score formulas
├── usage.md                  # Extended usage notes
├── faq.md                    # Common questions
└── project_structure.md      # This file
```

## Workspace layout (produced by running the pipeline)

See [`pipeline_architecture.md`](pipeline_architecture.md#workspace-layout) for the full per-event workspace tree (`raw_web/`, `gt_workspace/`, `media_workspace/`, `predict_workspace_v2/`).

## What's NOT in `code/`

- **Benchmark data**: per-event anonymized timelines, question banks, and ground truth ship with this supplement under `../data/`.
- **External tools**: Apify (web crawl) and MediaCrawlerPro-Python (social crawl). Both are external dependencies, not bundled.
- **Real API keys**: every key is loaded from `.env`; only `config.example.env` is included.
