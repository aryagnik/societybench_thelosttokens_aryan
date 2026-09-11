# Usage Guide

## Prerequisites

- Python 3.10+
- (Optional) Claude Code — only if you want to run via Skills
- API credentials (see [Configuration](#configuration))

## Configuration

Copy the template and fill in credentials:

```bash
cp eval/config.example.env .env
```

Required variables:

| Variable | Required for |
|---|---|
| `DMXAPI_KEY` | All LLM-driven steps + evaluation |
| `DMXAPI_BASE_URL` | All LLM-driven steps + evaluation |
| `DMXAPI_DEFAULT_MODEL` | Optional; defaults to `pipeline_config.json` value |
| `APIFY_TOKEN` | Web crawl only |
| `MCP_HOME`, `MCP_SIGNSRV_URL` | Media crawl only |

## Path 1 — Skills (via Claude Code)

```
/total-pipeline "<research topic>" /path/to/workspace
```

## Path 2 — Standalone Python (no Claude Code)

### End-to-end from a topic

```bash
python3 eval/total_pipeline.py "<topic>" /path/to/workspace \
    --replacements path/to/replacements.json \
    --event-name "<event name>"
```

### Per-stage execution

```bash
# Stage 0: data collection (requires external Apify + MediaCrawlerPro)
python3 eval/crawl/web_search_crawl_pipeline.py --project <slug> --topic "<topic>" --output-dir /path/to/workspace/raw_web/
python3 eval/crawl/media_crawl.py "<topic>" --output /path/to/workspace/raw_media/

# Stage 1: per-source processing
python3 eval/web_pipeline.py   /path/to/workspace/raw_web/articles.json /path/to/workspace/web/   --event-name "<event>"
python3 eval/media_pipeline.py /path/to/workspace/raw_media/data.csv    /path/to/workspace/media/ --event-name "<event>"

# Stage 2: merge
python3 eval/merge_pipeline.py \
    --web-dir   /path/to/workspace/web/ \
    --media-dir /path/to/workspace/media/ \
    --output-dir /path/to/workspace/merge/ \
    --event-name "<event>"

# Stage 3: predict end-to-end (anonymize → points → QB → eval → score)
python3 eval/predict_pipeline.py \
    /path/to/workspace/merge/timeline_merged_long.md \
    /path/to/workspace/predict/ \
    --event-name "<event>" \
    --replacements path/to/replacements.json
```

## Evaluating only (on already-prepared data)

```bash
python3 eval/run_pipeline_parallel.py \
    --workspace /path/to/workspace/predict/ \
    --event-name "<event>" \
    --models "model1,model2"
```

This skips data collection / processing / merging / question-bank generation, and only runs:
1. `predict_step3B_brier.py` (calibration eval)
2. `predict_step3F_time.py` (temporal eval)
3. `predict_step4_scorecard.py` (aggregation)

## Single-step execution

Every step can also be run on its own:

```bash
python3 eval/web_step1_summarize.py articles.json /out --event-name "<event>"
python3 eval/predict_step0_prepare.py timeline.md /out --replacements replacements.json
python3 eval/predict_step4_scorecard.py --workspace /path/to/workspace
# ...etc.
```

Use `--help` on any script to see its arguments.

## Output layout

After `predict_pipeline.py` completes:

```
<workspace>/
├── timeline_anon.md
├── timeline_with_points.md
├── prediction_points.json
├── contexts/                   P01_context.md, P02_context.md, ...
├── gt/                         P01_gt.md, ...
├── questionbank/               P01_questionbank.json, ...
└── results/
    └── run_<timestamp>/
        ├── brier/<model>/      P01_brier.json, ...
        ├── time/<model>/       P01_time.json, ...
        ├── scorecard.json
        └── scorecard.md
```
