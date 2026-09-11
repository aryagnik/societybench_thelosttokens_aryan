<div align="center">

# 🔮 SocietyBench

反事实社会世界的演化预测
</br>
Forecasting Counterfactual Social-World Evolution

[![License](https://img.shields.io/badge/License-MIT-2a78d6?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-1baf7a?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SocietyBench-eda100?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)
[![Events](https://img.shields.io/badge/Events-5-eb6834?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)
[![Questions](https://img.shields.io/badge/Questions-25.4k%20%C3%97%202-e87ba4?style=flat-square)](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)

[English](./README.md) | [中文文档](./README-ZH.md)

</div>

## ⚡ Overview

**SocietyBench** measures a capability that task-completion benchmarks do not touch: whether a
model can forecast how a real social event *keeps unfolding*.

The difficulty with evaluating this on real events is that a strong model may simply **recognize**
the event from its pre-training data. SocietyBench removes that shortcut. Before any model sees a
timeline, every named entity is replaced with a placeholder and every date is shifted by a
per-event constant. The causal and temporal structure survives intact; the labels a model could
match against memory do not. What is left is a **counterfactual social world** — an arc that can
only be predicted, never recalled.

> **You provide:** a one-line event topic.</br>
> **The pipeline returns:** an anonymized fact-plus-opinion timeline, an audited question bank at
> every cutoff date, and scores on two orthogonal axes.

This repository is the full pipeline: collecting the raw material, building the timeline,
anonymizing it, generating the questions, and scoring the answers.

## 🎯 Benchmark at a glance

A model is placed at a **cutoff date**. It sees everything before that date and nothing after,
then answers two kinds of question about what happens next:

| Axis | Question | Metric | Trivial anchor |
|------|----------|--------|----------------|
| **Probability calibration** | Does event *E* happen within the next *W* days? | weighted MAE → 0–100 | answering 50% to everything scores exactly **50** |
| **Temporal accuracy** | On what date does event *e* happen? | segment-normalized day error → 0–100 | guessing the bucket midpoint scores about **50** |

| | Scale |
|---|---|
| **Events** | 5 — public controversy, geopolitics, tech policy, financial markets, trade policy |
| **Prediction points** | 25 per event, **125** total |
| **Calibration questions** | **25,364** per language edition |
| **Temporal events** | **3,112** per language edition |
| **Editions** | Chinese and English, one-to-one |
| **Systems evaluated** | 6 frontier LLMs · 3 agent frameworks · 2 model-free baselines |

The strongest model reaches **75.0 / 100** — roughly half the headroom above the trivial anchor.
The benchmark is far from saturated.

## 🔄 How it works

1. **采集 / Collect** — Web news and social-media posts across five platforms, from a one-line topic
2. **蒸馏 / Distill** — an agent chain compresses raw items into a date-indexed timeline that keeps
   *factual* events and a *public-opinion* layer separate
3. **匿名 / Anonymize** — three phases: entity substitution and date shifting, an adversarial
   reverse-identification audit, then a semantic-consistency repair pass
4. **出题 / Generate** — every cutoff date becomes an audited bank of calibration and temporal questions
5. **评分 / Score** — two independent 100-point axes, computed per event and averaged across events

## 🚀 Quick start

### Prerequisites

| Tool | Version | Purpose | Check |
|------|---------|---------|-------|
| **Python** | 3.10+ | everything | `python3 --version` |
| **LLM endpoint** | any OpenAI-compatible | the model under test (DMXAPI, OpenRouter, …) | — |
| **Apify token** | optional | only if you crawl Web news yourself | — |

#### 1. Install

```bash
git clone https://github.com/co-minder/SocietyBench-codebase
cd SocietyBench-codebase

pip install -r requirements.txt          # evaluation
pip install -r requirements-crawl.txt    # only if you also want to crawl
```

#### 2. Configure

```bash
cp eval/config.example.env eval/.env
```

**Required environment variables:**

```env
# Any OpenAI-compatible endpoint works (DMXAPI, OpenRouter, ...)
DMXAPI_KEY=sk-...
DMXAPI_BASE_URL=https://www.dmxapi.com/v1
```

Verify before launching anything long-running:

```bash
python3 eval/health_check.py     # expect: [health] OK — model=...
```

#### 3. Reproduce the paper's numbers

No crawling needed — this pulls one event from the dataset repository and runs both axes
end to end:

```bash
python3 main.py --reproduce event3_tiktok /path/to/workspace
```

Events: `event1_library` · `event2_trump_tariff` · `event3_tiktok` · `event4_us_iran` ·
`event5_smci`. Results land in `<workspace>/results/run_<timestamp>/`.

Already have the data locally? Call the evaluator directly:

```bash
python3 eval/run_pipeline_parallel.py \
    --workspace /path/to/event3_tiktok/zh \
    --event-name event3_tiktok \
    --models "<model-id>"
```

> **English edition.** The scripts switch to English prompts when the workspace path ends in a
> directory named `英文`. To evaluate `<event>/en`, symlink it first — `ln -s en 英文` — then
> pass that path.

#### 4. Build a benchmark from your own event

```bash
# Which real names map to which placeholders
cat > reps.json <<'JSON'
{"Tesla": "Company A", "Elon Musk": "Person A", "Austin": "City A"}
JSON

python3 main.py "Tesla strike in Texas 2025" /path/to/workspace \
    --replacements-json reps.json
```

Add `--start-phase N` to resume partway: `0` from the crawl, `1` from per-source processing,
`2` from the merge, `3` from evaluation only.

## 🏗️ Project structure

| Path | Contents |
|------|----------|
| `main.py` | Entry point — reproduce an event, or build a new one |
| `eval/` | Every pipeline stage, one script each; no event-specific content |
| `eval/predict_step0–2*` | Anonymization, prediction-point selection, question-bank generation |
| `eval/predict_step3B*` | The probability-calibration axis |
| `eval/predict_step3F*` | The temporal-accuracy axis |
| `eval/predict_step4_scorecard*` | Cross-event aggregation |
| `eval/agents/` | Three agent baselines — LangGraph, AutoGen, MiroFish |
| `eval/baseline/` | Two model-free heuristics — event base rate, 7-day momentum |
| `eval/pipeline_config.json` | Runtime parameters: models, thresholds, scoring constants |
| `eval/config.example.env` | Credentials template — copy to `.env` |
| `docs/` | Methodology, scoring, architecture, usage, case studies, FAQ |

> Keep the two configuration files separate: `pipeline_config.json` is generic behaviour and
> belongs in version control; `.env` holds your keys and never does.

## 📚 Documentation

| Document | What it answers |
|----------|-----------------|
| [Project structure](docs/project_structure.md) | Complete file index — what every file does |
| [Methodology](docs/methodology.md) | Framework, anonymization, evaluation axes |
| [Pipeline architecture](docs/pipeline_architecture.md) | Data flow, workspace layout, config keys |
| [Scoring](docs/scoring.md) | The calibration and temporal formulas |
| [Case studies](docs/case_studies.md) | The five events, and how to run on a new one |
| [Usage](docs/usage.md) | Extended usage notes |
| [FAQ](docs/faq.md) | Switching models, debugging, common errors |

## 💾 Data

Anonymized timelines, question banks, and ground truth:
**[🤗 Social-AI-2026/SocietyBench](https://huggingface.co/datasets/Social-AI-2026/SocietyBench)**

The entity replacement tables are deliberately **not** published. Releasing them would
de-anonymize every event and defeat the point of the benchmark.

## 🤝 Contributing

New model adapters, new agent baselines, and new events built with this pipeline are all
welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first — it has two hard rules:
**never commit real-entity material**, and **never commit credentials**.

If what you found is an anonymization leak (a residual real name, a searchable identifying
detail), please do not open a public issue — follow [SECURITY.md](SECURITY.md) instead. A
public report that names the underlying real event contaminates the benchmark for everyone.

## 📄 Citation

```bibtex
@misc{societybench2026,
  title  = {SocietyBench: Forecasting Counterfactual Social-World Evolution},
  author = {Wang, Zhenran and Bian, Zhonghan and Li, Jinsong and Qi, Zhangyang},
  year   = {2026},
  note   = {\url{https://github.com/co-minder/SocietyBench-codebase}}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## 🙏 Acknowledgements

The MiroFish agent baseline is built on **[MiroFish](https://github.com/666ghj/MiroFish)**, whose
simulation engine in turn runs on **[OASIS](https://github.com/camel-ai/oasis)** by CAMEL-AI. The
other two agent baselines use **[LangGraph](https://github.com/langchain-ai/langgraph)** and
**[AutoGen](https://github.com/microsoft/autogen)**. Our thanks to all of these teams for their
open-source work.

## ⚖️ License

Code under the [MIT License](LICENSE). The benchmark data is released separately under CC BY 4.0.
