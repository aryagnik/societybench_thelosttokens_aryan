# FAQ

## How do I run on a new event (not one of the 5 in the paper)?

You don't need to change any Python or SKILL file. See [`case_studies.md`](case_studies.md#running-on-a-new-event-not-one-of-the-5) for the 3-step recipe.

## Which LLM does the pipeline use?

Everything is configurable. Three layers, highest precedence first:

1. **CLI flag** — `--model <name>` on any sub-pipeline; `--eval-models a,b,c` on predict/total.
2. **`pipeline_config.json`** — `default_llm_model` (for processing) and `evaluation_models` (for who gets benchmarked).
3. **Environment variable** — `DMXAPI_DEFAULT_MODEL` fallback.

The model name is whatever your aggregator (DMXAPI / OpenRouter / SiliconFlow / OpenAI-direct / Anthropic-direct via gateway) accepts.

## Can I use OpenAI / Anthropic / Gemini directly?

Yes, as long as the endpoint speaks the OpenAI `/v1/chat/completions` protocol. Set `DMXAPI_BASE_URL` to that endpoint and `DMXAPI_KEY` to its key. The variable is named `DMXAPI_*` for historical reasons; it's actually a generic OpenAI-compatible endpoint slot.

## How do I run only a subset of the pipeline?

Use `--start-phase` (on `total_pipeline.py`) or `--start-step` (on `predict_pipeline.py`). All steps auto-skip when their output already exists. See [`pipeline_architecture.md`](pipeline_architecture.md#resume--partial-runs).

## My LLM is returning 429 / 503 / timeout — what now?

The pipeline retries internally. If failures persist:

1. Try a different model: change `pipeline_config.json` → `default_llm_model`, or pass `--model <other_model>`.
2. Reduce concurrency: lower `predict_step3_evaluation.max_parallel_batches` (default 30) or pass `--threads N` to step scripts.
3. Switch endpoint provider: any OpenAI-compatible aggregator works (DMXAPI, OpenRouter, SiliconFlow, etc.).

A persistent 429 burst usually means rate-limit. A persistent 503 usually means the aggregator routed your request to a model that's overloaded — try another model.

## I want to debug a specific step. How?

Every step script is callable standalone. For example:

```bash
python3 eval/web_step3_dedup.py \
    /path/to/timeline_step2.jsonl \
    /path/to/output_dir \
    --event-name "<event>"
```

Each script writes its own log file alongside its output (e.g. `timeline_step3.log`).

For the predict pipeline, you can re-run a single step via `predict_pipeline.py --start-step N`.

## Why does anonymization sometimes succeed even with an empty replacements.json?

That's by design. Phase 2 (LLM audit) auto-derives rules from any leaks it finds and appends them to the replacement table. An empty `{"replacements": []}` is a valid starting point — the audit will populate it. See [`methodology.md`](methodology.md#anonymization-counterfactual-world-construction).

## Why are there two scores per model instead of one?

A model can be well-calibrated (good at probabilities) but date-blind (bad at when), or vice versa. Combining them into one number would lose information. See [`scoring.md`](scoring.md).

## What's in the workspace directories?

See [`pipeline_architecture.md`](pipeline_architecture.md#workspace-layout) for the full tree.

## How big is the data?

Per event, after all pipelines: ≈ 5-20 MB of intermediates and final artefacts. Raw inputs (Phase 0 outputs) can be 50-200 MB.

## How long does a full run take?

Rough orders of magnitude on the 5 paper events (LLM calls dominate):

- Web pipeline: 10-40 minutes (depending on # articles, 1-5k typical)
- Media pipeline: 20-60 minutes (depending on # posts + comments)
- Merge pipeline: 5-15 minutes
- Predict pipeline: 30-90 minutes per evaluated model (depending on # prediction points × question count)

With `default_llm_model` set to a cheap model (e.g. `doubao-seed-1-6-flash`), end-to-end runs at single-digit USD per event.

## Why do some docstrings mention "SKILL" specs?

The project began as a set of interactive agent skill specs. For release, every step was migrated to call a unified LLM API directly, so the same pipeline runs headless; some docstrings still cite the original spec section names they implement.
