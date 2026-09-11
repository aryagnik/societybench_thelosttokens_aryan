# SocietyBench — Pipeline Architecture

## Data flow

```
raw_web/articles.json ────┐
                           │   ┌──────────────────────────────────┐
                           ├──▶│ Web pipeline (6 steps, LLM)      │──▶ web_workspace/
                           │   │  step1 summarize (300-400 char)  │     timeline_step1..4.jsonl
                           │   │  step2 clean meta-language        │     timeline_short.md
                           │   │  step3 per-day dedup              │     timeline.md
                           │   │  step4 extract "new info"         │
                           │   │  step5 short-form md              │
                           │   │  step6 long-form md (200-300/ev)  │
                           │   └──────────────────────────────────┘
                           │
raw_media/valid_data_llm_verified.csv ──┐
                                         │   ┌──────────────────────────────────┐
                                         ├──▶│ Media pipeline (6 steps, LLM)    │──▶ media_workspace/
                                         │   │  step0 aggregate post + top30 cm │     posts_aggregated.jsonl
                                         │   │  step1 summarize (100-200 char)  │     media_step1..4.jsonl
                                         │   │  step2 clean                      │     media_timeline_short.md
                                         │   │  step3 per-day dedup              │
                                         │   │  step4 extract                    │
                                         │   │  step5 short-form md              │
                                         │   └──────────────────────────────────┘
                                         │
                                         ▼
                            ┌──────────────────────────────────┐
                            │ Merge pipeline                    │──▶ media_workspace/
                            │  step5 web+media merge w/ [W][M] │     timeline_merged_short.md
                            │  step6 expand to 200-300 + opin.  │     timeline_long_part1.md
                            │  step6.5 GT refine (3 rounds)    │     timeline_long_part2.md
                            └──────────────────────────────────┘     timeline_merged_long.md
                                         │                            timeline_merged_long_gt.md
                                         ▼
                            ┌──────────────────────────────────┐
                            │ Predict pipeline                  │──▶ predict_workspace_v2/
                            │  step0 anonymize (3 phases)      │     timeline_anon.md
                            │  step1 pick ~25 prediction points │     prediction_points.json
                            │  step2 question bank             │     questionbank/P*_qb.json
                            │  step3B calibration eval         │     results/run_*/<model>/3b_brier/
                            │  step3F temporal eval (×1 run)  │     results/run_*/<model>/3f_time/
                            │  step4 scorecard                  │     scorecards/scorecard_<model>.json
                            └──────────────────────────────────┘
```

## Workspace layout

```
<workspace_root>/
├── raw_web/                            # raw inputs (Phase 0 / user-supplied)
│   └── articles.json
├── raw_media/
│   └── valid_data_llm_verified.csv
├── _tool_web_runtime/                  # Phase 0 isolated Apify runtime (auto)
├── _tool_media_runtime/                # Phase 0 isolated MCP runtime (auto)
├── gt_workspace/                       # web pipeline outputs (renamed from web_workspace)
│   ├── timeline_step1..4.jsonl
│   ├── timeline_short.md
│   └── timeline.md
├── media_workspace/                    # media pipeline + merge pipeline outputs
│   ├── posts_aggregated.jsonl
│   ├── media_step1..4.jsonl
│   ├── media_timeline_short.md
│   ├── timeline_merged_short.md
│   ├── timeline_long_part{1,2}.md
│   ├── timeline_merged_long.md
│   ├── timeline_merged_long_gt.md
│   ├── timeline_merged_long_gt_report.json
│   ├── timeline_merged_long_gt_iteration_log.md
│   └── _idx_{web,media,opinion}.json
└── predict_workspace_v2/               # predict pipeline outputs
    ├── timeline_anon.md
    ├── replacements_effective.json     # auto-derived rules from Phase 2 audit
    ├── audit_report.json
    ├── consistency_report.json
    ├── prediction_points.json
    ├── timeline_with_points.md
    ├── contexts/P*_context.md          # per-point context (≤ cutoff)
    ├── gt/P*_gt.md                     # per-point GT (> cutoff)
    ├── questionbank/P*_questionbank.json
    ├── results/run_<ts>/<model>/
    │   ├── 3b_brier/P*.json
    │   └── 3f_time/P*.json
    └── scorecards/scorecard_<model>.json
```

## Configuration

| Key in `pipeline_config.json` | What it controls |
|---|---|
| `default_llm_model` | Model used for all processing steps (summarize / clean / dedup / extract / refine / audit) |
| `evaluation_models` | List of models to evaluate (each gets its own scorecard) |
| `media_platforms` | Comma-separated platforms used by media-crawl |
| `predict_step1_points.{interval_threshold_days, target_points, max_points}` | Prediction-point selection thresholds |
| `predict_step2_questionbank.{d_goal_pct, d_blacklist_terms, unfalsifiable_terms}` | D-class fake-question generation policy + reject lists |
| `predict_step3_evaluation.{max_tokens_per_call, thinking_budget_tokens, reasoning_effort_default, batch_max_size, batch_max_same_group, max_parallel_batches, calibration_runs_per_point, temporal_runs_per_point}` | Per-call LLM parameters for evaluation |
| `scoring_calibration.{baseline_uniform_predictor_wmae, time_factor_coefficient, window_factor_offset}` | Calibration formula constants |
| `scoring_temporal.{bucket_days, buckets_midpoint_days}` | Temporal formula constants |
| `anonymization.{date_shift_max_days}` | Date-shift offset range |

## Resume / partial runs

```bash
# Start from any phase if intermediates exist:
python3 eval/total_pipeline.py "<topic>" /path/to/workspace \
    --replacements-json reps.json \
    --start-phase {0,1,2,3}

# 0 = real Apify/MCP crawl
# 1 = per-source processing (needs raw_web/ raw_media/)
# 2 = merge (needs gt_workspace/ media_workspace/)
# 3 = predict only (needs media_workspace/timeline_merged_long_gt.md)
```

```bash
# Within predict pipeline, finer-grained step recovery:
python3 eval/predict_pipeline.py <gt_md> <predict_workspace> \
    --event-name "<event>" \
    --replacements-json reps.json \
    --start-step {0,1,2,3,4}

# 0 = anonymize, 1 = points, 2 = questionbank, 3 = eval, 4 = scorecard
```

All steps support resume — running the same command again will skip any step whose output already exists (use `--force` on sub-pipelines to override).
