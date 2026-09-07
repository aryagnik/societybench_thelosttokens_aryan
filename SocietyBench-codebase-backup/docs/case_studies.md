# SocietyBench — Case Studies

The framework is event-agnostic. These 5 events are the case studies used in our paper, chosen to span **diverse domains**, **diverse information velocities** (slow policy negotiation vs. fast geopolitical), and **diverse data-modality mixes** (web-heavy vs. social-media-heavy).

| # | Domain | Brief description | Time span | Pipeline mix |
|---|---|---|---|---|
| 1 | Public controversy | A sustained Chinese-language campus controversy with heavy social-media debate | ≈ 5 months | media-heavy |
| 2 | Geopolitical conflict | US–Iran tensions, Strait of Hormuz dynamics | ≈ 6 weeks | web + media balanced |
| 3 | Technology policy | US TikTok divestiture / ban negotiations | ≈ 8 months | web-heavy |
| 4 | Trade policy | US reciprocal-tariff escalation | ≈ 4 months | web-heavy |
| 5 | Financial markets | Short-seller report fallout on a US-listed company | ≈ 3 months | web-heavy |

For each event, the workspace artefacts (anonymized timeline, prediction points, question bank, ground truth, per-model outputs, scorecards) are released **separately on HuggingFace**, not in this code repo.

---

## Why these five

| Constraint | What we wanted to test |
|---|---|
| ≥ 4 weeks per event | Non-trivial post-cutoff prediction horizon |
| ≥ 2 distinct languages across the set | Cross-lingual robustness of the anonymization audit |
| Mix of policy / conflict / market / public-controversy | Generalization across domain priors |
| Mix of slow vs. fast information velocity | Different temporal-prediction difficulty profiles |
| Mix of web-heavy vs. media-heavy | Stress-test both the web pipeline and the media pipeline |

---

## Running on a *new* event (not one of the 5)

The repository code does not contain any event-specific logic. To run the benchmark on any new event:

```bash
# 1) Drop raw data into the expected layout:
mkdir -p /path/to/workspace/raw_web /path/to/workspace/raw_media
# raw_web/articles.json                — JSON array of {url, title, content, date, ...}
# raw_media/valid_data_llm_verified.csv — flattened post + comment CSV

# 2) An empty replacements.json is fine — Phase 2 audit auto-derives the rules.
echo '{"replacements": []}' > replacements.json

# 3) Run end-to-end (skip Phase 0 because raw data is already present).
python3 eval/total_pipeline.py "<one-line topic description>" \
    /path/to/workspace \
    --replacements-json replacements.json \
    --start-phase 1
```

The pipeline produces:

- `media_workspace/timeline_merged_long_gt.md` — the anonymized, refined ground-truth timeline
- `predict_workspace_v2/questionbank/` — generated calibration + temporal questions
- `predict_workspace_v2/scorecards/scorecard_<model>.json` — final per-model scores on both axes

You do not need to touch any Python code or any SKILL file.
