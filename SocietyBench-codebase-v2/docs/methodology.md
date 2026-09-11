# SocietyBench Methodology

## Overview

SocietyBench evaluates LLMs and agents on **forecasting the dynamics of real-world social events** — but with a twist. Each evaluated event is rendered into a *counterfactual social world*: structurally identical to what really happened, yet with every named entity replaced and every date shifted. As a result, a model's score reflects **forward reasoning about social dynamics**, not recall of pre-training material.

The framework itself is **fully event-agnostic**. The code in this repository can ingest any topic with a multi-week timeline and produce a calibrated + temporally-scored benchmark for it. The 5 events used in our paper are described in [`case_studies.md`](case_studies.md); they are illustrations, not constraints. To run the benchmark on a new event of your own, see the README quick-start.

## Pipeline (five stages)

```
1. Data collection       Web search + multi-platform social-media crawl
   (web-search-crawl, media-crawl)

2. Per-source processing Summarize → clean → dedup → extract → short/long timeline
   (web-pipeline, media-pipeline; each ~6 sub-steps; all LLM-driven)

3. Merge                 Web + Media → unified chronology, separating
   (merge-pipeline)      "facts" from "public-opinion" layer

4. Anonymize + question  Three-phase entity-and-date anonymization
   bank generation       → prediction points → calibration + temporal questions
   (predict-step0-2)

5. Predict + score       Models answer; scored on two orthogonal axes
   (predict-step3B-brier, predict-step3F-time, predict-step4-scorecard)
```

See [`pipeline_architecture.md`](pipeline_architecture.md) for the full data-flow diagram and per-step input/output schemas.

## Anonymization (counterfactual-world construction)

Each merged timeline is jointly anonymized in three phases:

- **Phase 1 — Substitution.** Replace named entities via a per-event replacement table $\mathcal{R}$ (longest match first, with double-replacement repair), then shift every date by a per-event offset $\delta \sim \mathcal{U}[-180, +180]$ days.

- **Phase 2 — Reverse-identification audit.** The unified LLM API re-reads the anonymized text and reports any residual search-keys (un-replaced real names, searchable headlines, recoverable quotes) and any substitution artefacts (e.g. double-replacements). If leaks are found, replacement rules are **automatically derived** from the audit findings, appended to $\mathcal{R}$, and Phase 1 is re-run from the **raw original text** (not in-place rewriting). The loop terminates when `high == 0 && mid == 0`, or after 5 rounds.

- **Phase 3 — Semantic-consistency pass.** Compare against the original at paragraph, event, and narrative level; flag any content lost or distorted by substitution.

After this, the model sees an event it cannot identify even though that event has already played out — so its score reflects forward social-dynamic reasoning rather than recall.

Note: the user need not hand-craft $\mathcal{R}$. An empty `{"replacements": []}` is a valid starting point — Phase 2 will auto-derive the rules. The user is only the **final approver** of the refined ground truth, via a one-line decision in the iteration log.

## Evaluation axes

Each event yields **two orthogonal** model scores (formulas in [`scoring.md`](scoring.md)):

| Axis | What it measures | Baseline (score 0) | Perfect (score 100) |
|---|---|---|---|
| **Calibration** | Probability quality | Uniform 50 % predictor | Always-correct probability |
| **Temporal accuracy** | When-it-happens prediction | Bucket-midpoint baseline | Exact-day prediction |

A model can be well-calibrated yet date-blind, or vice versa — so the two axes are reported separately, never combined into a single overall score.
