# Contributing to SocietyBench

Thanks for taking the time. This is a research benchmark, so a few of the rules below are
stricter than in an ordinary software project — they exist to keep published numbers
comparable across papers.

## Ground rules

**1. Never commit real-entity material.** The benchmark only works because the released
timelines are anonymized. Do not add, in code, tests, docs, or issue text:

- entity replacement tables (real name → placeholder), in any form
- un-anonymized timelines, or the true date offset for any event
- anything that would let a reader identify which real event an event ID refers to

If you think you have found such a leak in the repository, follow [SECURITY.md](SECURITY.md)
instead of opening a public issue.

**2. Never commit credentials.** `eval/.env` is gitignored. Keys belong there, never in
`pipeline_config.json`, never in a docstring, never in a test fixture.

**3. Scoring changes are breaking changes.** Anything that touches
`predict_step3B_brier*`, `predict_step3F_time*`, or `predict_step4_scorecard*` changes what
a score means. Such a PR must say so in its title, explain the motivation, and report the
before/after numbers on at least one full event so reviewers can see the size of the shift.

## Getting set up

```bash
git clone https://github.com/co-minder/SocietyBench-codebase
cd SocietyBench-codebase
pip install -r requirements.txt
cp eval/config.example.env eval/.env      # then fill in your key
python3 eval/health_check.py              # expect: [health] OK — model=...
```

## Before you open a PR

```bash
python3 -m compileall -q main.py eval      # must exit 0
```

If you changed the pipeline's behaviour, run one event end to end and paste the resulting
scorecard into the PR:

```bash
python3 main.py --reproduce event1_library /tmp/sb-check
```

`event1_library` is the smallest of the five and the cheapest to use as a smoke test.

## Style

- Python 3.10+, standard library first; add a third-party dependency only if it earns its keep
- Every new dependency goes in the requirements file that matches its scope — core,
  `-crawl`, or `-agents` — never all three
- Comments and docstrings in English
- 4-space indent, 100-column soft limit (see `.editorconfig` and `pyproject.toml`)
- One pipeline stage per file, matching the existing `predict_stepN_*` naming

## What is especially welcome

- New events built with the public pipeline, with the replacement table kept private
- Additional model or agent adapters
- Reproduction reports — if your numbers differ from the paper's, that is a useful issue

## Licence

Contributions are accepted under the [MIT Licence](LICENSE), the same terms as the rest of
the code.
