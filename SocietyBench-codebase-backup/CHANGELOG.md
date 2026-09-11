# Changelog

All notable changes to the SocietyBench code are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Because this is a benchmark, one rule is worth stating up front: **any change to the scoring
formulas is a major version bump**, even when the code change is small. Scores produced by
different major versions are not comparable.

## [Unreleased]

Nothing yet.

## [1.0.0] — 2026-08-01

First public release, matching the SocietyBench paper.

### Added

- End-to-end pipeline: crawl → per-source processing → timeline merge → anonymization →
  prediction-point selection → question-bank generation → evaluation → scorecard
- Two scoring axes: probability calibration (`predict_step3B_brier*`) and temporal accuracy
  (`predict_step3F_time*`), each mapped to 0–100 with a trivial anchor at 50
- Three-phase anonymization: entity substitution and date shifting, adversarial
  reverse-identification audit, semantic-consistency repair
- `main.py --reproduce <event>`, which downloads one released event and runs both axes
  without any crawling
- Three agent baselines in `eval/agents/` — LangGraph, AutoGen, MiroFish — and two model-free
  heuristics in `eval/baseline/`
- Seven documents in `docs/`: project structure, methodology, pipeline architecture, scoring,
  case studies, usage, FAQ
- `requirements-agents.txt`, so the agent baselines' dependencies are installable
  independently of the core evaluation

### Notes

- The entity replacement tables, real-name variants, and per-question model responses are
  deliberately not released. See [SECURITY.md](SECURITY.md).
- Data lives separately at
  [`Social-AI-2026/SocietyBench`](https://huggingface.co/datasets/Social-AI-2026/SocietyBench).

[Unreleased]: https://github.com/co-minder/SocietyBench-codebase/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/co-minder/SocietyBench-codebase/releases/tag/v1.0.0
