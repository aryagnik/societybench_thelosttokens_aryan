#!/usr/bin/env python3
"""Main runner for agent evaluation (M3).

Reuses the prompts, parsing, and scoring of predict_step3B_brier / predict_step3F_time,
only replacing the "call model" step with agent_answer (framework x base model). Results land
in the **exact same** output directory as bare models
(results/run_<ts>/{brier,time}/<config_slug>/), so predict_step4_scorecard reads them
like "just another model". config_slug looks like langgraph__doubao-seed-2-0-pro-260215.

Smoke example (event5, single point, truncated questions, very cheap):
  python run_agents.py --workspace <ws> --event-name smci \
      --frameworks langgraph,autogen --bases doubao-seed-2-0-pro-260215 \
      --axes brier,time --points P01 --max-questions 12 --max-events 8 --runs-per-point 1

Full example:
  python run_agents.py --workspace <ws> --event-name smci \
      --frameworks langgraph,autogen,mirofish \
      --bases doubao-seed-2-0-pro-260215,qwen3.5-plus-2026-02-15 --axes brier,time
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock, Semaphore

_HERE = pathlib.Path(__file__).resolve().parent
_EVAL = _HERE.parent
for _p in (str(_HERE), str(_EVAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adapter  # noqa: E402
import predict_step3B_brier as B  # noqa: E402
import predict_step3F_time as F  # noqa: E402
from adapter import agent_answer, make_session  # noqa: E402
from common import load_pipeline_config, parse_date_obj  # noqa: E402

# Batch size / same-event cap follow the project pipeline_config (same source as bare-model 3B
# to keep comparisons consistent; a config change propagates automatically, no code change).
_S3 = load_pipeline_config().get("predict_step3_evaluation", {})
_DEFAULT_BATCH_SIZE = int(_S3.get("batch_max_size", 999999))
_SAME_GROUP = int(_S3.get("batch_max_same_group", 1))
_MAX_REUSABLE_PARSE_FAIL_RATE = 0.02
_AUTOGEN_BASE_FALLBACK = os.environ.get("AGENT_AUTOGEN_BASE_FALLBACK", "0") != "0"
_COVERAGE_ONLY_ENV = (
    os.environ.get("SB_COVERAGE_ONLY", "0") != "0"
    or os.environ.get("SB_AGENT_COVERAGE_ONLY", "0") != "0"
)


def cfg_slug(framework: str, base: str) -> str:
    return f"{framework}__{B.model_slug(base)}"


# ===========================================================================
# Continuous rollout: slicing the real new developments (delta) between two cutoffs
# ===========================================================================
# Each point's context is a strict cumulative prefix: Pn_context starts verbatim with the full
# P(n-1)_context plus appended new date sections.
# So delta(Pn) = Pn_context[len(prev_ctx):], the real new developments between the two cutoffs (no future leakage).
# Fallback: if the prefix property is broken (context was edited), split by `### YYYY-MM-DD`
# date sections and take those with prev_cutoff < date <= cutoff; failing that, use the tail
# difference after the longest common prefix.

_DATE_HDR_RE = re.compile(r"(?m)^###\s+(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")


def _split_date_segments(ctx_text: str):
    """Split into (preamble, [(date_str, segment_text), ...]). A segment includes its `### date` line up to the next date line."""
    matches = list(_DATE_HDR_RE.finditer(ctx_text))
    if not matches:
        return ctx_text, []
    preamble = ctx_text[: matches[0].start()]
    segs = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ctx_text)
        date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        segs.append((date_str, ctx_text[start:end]))
    return preamble, segs


def compute_delta(prev_ctx_text: str, cur_ctx_text: str,
                  prev_cutoff: str, cutoff: str) -> str:
    """Return the new-developments text to feed the session between the previous advanced point and the current one.

    prev_ctx_text is the full context of the previous **advanced point** (which may skip over
    several unselected points, guaranteeing that under a point subset the information equals
    what a bare model sees for that cutoff — no gaps, no overlap).
    """
    cur = cur_ctx_text or ""
    if not (prev_ctx_text or "").strip():
        return cur  # first point: feed the whole context (header + all date sections up to cutoff)
    if cur.startswith(prev_ctx_text):
        return cur[len(prev_ctx_text):]
    # Fallback 1: take the (prev_cutoff, cutoff] date sections
    _preamble, segs = _split_date_segments(cur)
    pc, cc = parse_date_obj(prev_cutoff), parse_date_obj(cutoff)
    keep = []
    for date_str, seg in segs:
        d = parse_date_obj(date_str)
        if d is None:
            continue
        if ((pc is None) or d > pc) and ((cc is None) or d <= cc):
            keep.append(seg)
    if keep:
        return "".join(keep)
    # Fallback 2: tail difference after the longest common prefix (always yields the "new tail", never re-feeds old information)
    n = 0
    for a, b in zip(prev_ctx_text, cur):
        if a != b:
            break
        n += 1
    return cur[n:]


def reusable_brier(path: pathlib.Path, coverage_only: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    n = int(data.get("n") or len(data.get("questions") or []) or len(data.get("details") or []) or 0)
    pf = int(data.get("parse_fail_count") or 0)
    if coverage_only and n > 0:
        return data
    if data.get("agent_fallback"):
        return None
    if n <= 0 or pf / n > _MAX_REUSABLE_PARSE_FAIL_RATE:
        return None
    return data


def reusable_time(path: pathlib.Path, runs_per_point: int, coverage_only: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if coverage_only:
        return data
    if data.get("score_100") is None and data.get("events_total", 1) != 0:
        return None
    if data.get("agent_fallback"):
        return None
    if data.get("events_total", 1) != 0:
        per_run = data.get("per_run") or []
        if len(per_run) < runs_per_point:
            return None
        if any(r.get("weighted_mae_days") is None for r in per_run):
            return None
    return data


def base_fallback_point(
    ws: pathlib.Path,
    run_id: str,
    axis: str,
    framework: str,
    base: str,
    slug: str,
    pid: str,
    runs_per_point: int,
) -> dict | None:
    if framework.lower() != "autogen" or not _AUTOGEN_BASE_FALLBACK:
        return None
    base_slug = B.model_slug(base)
    fp = ws / "results" / f"run_{run_id}" / axis / base_slug / f"{pid}_{axis}.json"
    if axis == "brier":
        data = reusable_brier(fp)
    else:
        data = reusable_time(fp, runs_per_point)
    if data is None:
        return None
    cloned = json.loads(json.dumps(data, ensure_ascii=False))
    cloned["model"] = slug
    cloned["agent_fallback"] = {
        "framework": framework,
        "source_model": base_slug,
        "source_file": str(fp),
        "reason": (
            "AutoGen debate plumbing timed out/hung in formal max-budget runs; "
            "kept the initial draft baseline result, matching fw_autogen fallback policy. "
            "Prompt, context, questionbank, GT, and scoring are unchanged."
        ),
    }
    return cloned


# ===========================================================================
# 3B calibration: reuse B's prompt / parsing / scoring
# ===========================================================================

def brier_point(pid, qb_path, ctx_path, framework, base, cal_cfg,
                max_tokens, thinking_budget, batch_size, max_q=None, max_parallel=1,
                existing_bad=None, answer_fn=None, embed_context=True):
    """Get one prediction point's Brier answers and score them (structure reuses B's parsing/scoring 100%).

    How answers are obtained is decided by answer_fn:
      - None (legacy): each batch calls agent_answer(framework, base, ...), rerunning the full rollout from scratch;
      - callable (session): each batch does raw = answer_fn(prompt), reading off the already-built continuous rollout session.
    With embed_context=False the cumulative context is not stuffed into the prompt (knowledge lives in the session, avoiding double feeding).
    """
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    cutoff = qb.get("cutoff_date") or ""
    questions = qb.get("brier_questions") or []
    if max_q:
        questions = questions[:max_q]
    slug = cfg_slug(framework, base)
    reported_batch = B.reported_batch_size(None, batch_size)
    if not questions:
        return {"point_id": pid, "model": slug, "n": 0, "score_100": None,
                "batch_size": reported_batch,
                "batch_size_note": B.batch_size_note(reported_batch),
                "questions": [], "details": []}
    context = (ctx_path.read_text(encoding="utf-8")
               if (ctx_path.exists() and embed_context) else "")

    indexed = list(enumerate(questions))
    batches = B.build_batches(indexed, pid, batch_size, _SAME_GROUP)  # follows config/CLI (same as bare models)
    batch_by_index = {
        orig_idx: batch_idx
        for batch_idx, batch in enumerate(batches)
        for orig_idx, _q in batch
    }
    probs = [None] * len(questions)
    parse_fails = set()
    fail_kind_by_idx = {}   # orig_idx -> "dmx" (batch/single request error, timeout) / "model" (parse failure), for the v3 three-ratio classification
    parse_fails_lock = Lock()
    batch_records = []
    batch_records_lock = Lock()
    call_gate = Semaphore(max(1, int(max_parallel or 1)))

    def run_batch(batch):
        qblock = "\n".join(f"{i + 1}. {B.render_question(q)}" for i, (_, q) in enumerate(batch))
        prompt = B.ANSWER_PROMPT.format(cutoff=cutoff, context=context, question_list=qblock)
        if answer_fn is not None:
            raw = answer_fn(prompt)  # session: read this batch's answers off the existing continuous rollout state
        else:
            with call_gate:
                raw = agent_answer(framework, base, prompt, "brier", max_tokens, thinking_budget)
        return B.parse_answers(raw)

    if existing_bad:
        existing_probs = B.existing_point_probs(existing_bad, len(questions))
        if existing_probs is not None:
            existing_fail_indices, repair_batch_ids, repair_reason = B.failed_indices_from_existing(
                existing_bad, batches, existing_probs
            )
            probs = [float(p) for p in existing_probs]
            parse_fails = set(existing_fail_indices)
            batches_to_run = {
                bi: batch for bi, batch in enumerate(batches)
                if bi in set(repair_batch_ids)
            }
            for bi, batch in enumerate(batches):
                if bi not in batches_to_run:
                    batch_records.append({
                        "batch_idx": bi,
                        "size": len(batch),
                        "status": "reused",
                        "parse_fail_count": 0,
                        "parse_fail_indices": [],
                    })
            print(
                f"  {slug} {pid} agent batch repair: {len(batches_to_run)}/{len(batches)} "
                f"batches ({repair_reason})"
            )
        else:
            batches_to_run = {bi: batch for bi, batch in enumerate(batches)}
            print(f"  {slug} {pid} agent batch repair fallback: rerun all batches")
    else:
        batches_to_run = {bi: batch for bi, batch in enumerate(batches)}

    def recover_single_questions(items, log_label):
        recovered = {}
        single_errors = {}
        for orig_idx, question in items:
            last = ""
            for attempt in range(1, _SINGLE_REMAKE_ATT + 1):
                try:
                    ans = run_batch([(orig_idx, question)])
                except Exception as e:
                    last = str(e)
                    print(f"  {log_label} q{orig_idx + 1} single {attempt} error: {e}", flush=True)
                    continue
                if 1 in ans:
                    recovered[orig_idx] = ans[1]
                    last = ""
                    break
                last = "parse_fail=1/1"
            if orig_idx not in recovered:
                single_errors[orig_idx] = last
        return recovered, single_errors

    def process(batch_idx, batch):
        last_error = ""
        ans = {}
        status = "exception"
        missing = list(range(1, len(batch) + 1))
        for attempt in range(1, _BATCH_REMAKE_ATT + 1):
            try:
                ans = run_batch(batch)
            except Exception as e:
                last_error = str(e)
                print(
                    f"  {slug} {pid} batch {batch_idx + 1}/{len(batches)} "
                    f"remake {attempt} error: {e}",
                    flush=True,
                )
                continue
            missing = [k for k in range(1, len(batch) + 1) if k not in ans]
            if not missing:
                status = "ok"
                break
            status = "parse_fail"
            last_error = f"parse_fail={len(missing)}/{len(batch)}"
            if attempt < _BATCH_REMAKE_ATT:
                print(
                    f"  {slug} {pid} batch {batch_idx + 1}/{len(batches)} "
                    f"remake {attempt} ({last_error})",
                    flush=True,
                )

        if status == "exception":
            recovered, single_errors = recover_single_questions(
                batch,
                f"{slug} {pid} batch {batch_idx + 1}/{len(batches)}",
            )
            for orig_idx, prob in recovered.items():
                probs[orig_idx] = prob
            failed_orig = [orig for orig, _ in batch if orig not in recovered]
            with parse_fails_lock:
                parse_fails.difference_update(recovered)
                parse_fails.update(failed_orig)
                for _o in recovered:
                    fail_kind_by_idx.pop(_o, None)
                for _o in failed_orig:
                    fail_kind_by_idx[_o] = _fail_kind_from_err(single_errors.get(_o, last_error))
            with batch_records_lock:
                batch_records.append({
                    "batch_idx": batch_idx,
                    "size": len(batch),
                    "status": "single_question_fallback" if recovered else "exception",
                    "error": last_error,
                    "parse_fail_count": len(failed_orig),
                    "parse_fail_indices": failed_orig,
                    "single_question_recovered": sorted(recovered),
                    "single_question_errors": {
                        str(k): v for k, v in single_errors.items() if k not in recovered
                    },
                })
            return

        failed_orig = []
        for k, (orig, _q) in enumerate(batch, start=1):
            if k in ans:
                probs[orig] = ans[k]
                with parse_fails_lock:
                    parse_fails.discard(orig)
            else:
                failed_orig.append(orig)
        recovered = {}
        single_errors = {}
        if failed_orig:
            failed_items = [(orig, q) for k, (orig, q) in enumerate(batch, start=1) if orig in failed_orig]
            recovered, single_errors = recover_single_questions(
                failed_items,
                f"{slug} {pid} batch {batch_idx + 1}/{len(batches)}",
            )
            for orig_idx, prob in recovered.items():
                probs[orig_idx] = prob
            failed_orig = [orig for orig in failed_orig if orig not in recovered]
        with parse_fails_lock:
            parse_fails.difference_update(recovered)
            parse_fails.update(failed_orig)
            for _o in recovered:
                fail_kind_by_idx.pop(_o, None)
            for _o in failed_orig:
                fail_kind_by_idx[_o] = _fail_kind_from_err(single_errors.get(_o, last_error))
        with batch_records_lock:
            batch_records.append({
                "batch_idx": batch_idx,
                "size": len(batch),
                "status": "ok" if not failed_orig else ("single_question_fallback" if recovered else status),
                "parse_fail_count": len(failed_orig),
                "parse_fail_indices": failed_orig,
                **({"single_question_recovered": sorted(recovered)} if recovered else {}),
                **({"single_question_errors": {str(k): v for k, v in single_errors.items() if k not in recovered}}
                   if single_errors else {}),
                **({"error": last_error} if failed_orig and last_error else {}),
            })

    workers = max(1, min(int(max_parallel or 1), len(batches_to_run) or 1))
    if batches_to_run:
        if workers == 1:
            for bi, batch in batches_to_run.items():
                process(bi, batch)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process, bi, batch) for bi, batch in batches_to_run.items()]
                for fut in as_completed(futs):
                    fut.result()

    for i, p in enumerate(probs):
        if p is None:
            parse_fails.add(i)

    # v3 (fixed 2026-07-02): the agent line also distinguishes dmx (upstream/request error/timeout)
    # vs model (parse failure), passing per-question fail_kind through to score_point (unrecorded
    # ones conservatively treated as model);
    # before the fix: all failed agent questions defaulted to model, so DMX-broken questions were
    # wrongly counted into model_fail_frac.
    fail_kinds = [None] * len(questions)
    for i in range(len(questions)):
        if probs[i] is None:
            fail_kinds[i] = fail_kind_by_idx.get(i, "model")

    # Fixed 2026-07-01: agent detail rows must carry answer_attempt, otherwise the aggregate
    # "no-signal gate" (default_frac) treats every row as default fill (answer_attempt=None)
    # -> default_frac=1.0 -> the whole cell's score is wrongly squashed to N/A.
    # Answered questions record 1, parse failures record None (failed questions still correctly
    # count toward the "no-signal" fraction; a healthy run has default_frac~=0).
    answer_attempts = [None if i in parse_fails else 1 for i in range(len(questions))]
    scoring = B.score_point(questions, probs, cal_cfg, parse_fails, batch_by_index,
                            answer_attempts=answer_attempts, fail_kinds=fail_kinds)
    cutoff_d = parse_date_obj(cutoff)
    enriched = []
    for q, p in zip(questions, probs):
        question_index = len(enriched)
        dt = parse_date_obj(q.get("d_target"))
        dd = (dt - cutoff_d).days if (dt and cutoff_d) else 0
        enriched.append({**q, "prob": None if question_index in parse_fails else p, "delta_days": dd,
                         "gt": int((q.get("answer") if q.get("answer") is not None else q.get("gt", 0)) or 0),
                         "question_index": question_index,
                         "batch_idx": batch_by_index.get(question_index),
                         "parse_failed": question_index in parse_fails,
                         "valid_for_scoring": question_index not in parse_fails,
                         "invalid_reason": "parse_failed_or_no_model_answer" if question_index in parse_fails else None})
    return {"point_id": pid, "model": slug, "cutoff_date": cutoff,
            "batch_size": reported_batch,
            "batch_size_note": B.batch_size_note(reported_batch),
            "questions": enriched, **scoring, "parse_fail_count": len(parse_fails),
            "parse_fail_indices": sorted(parse_fails),
            "valid_for_scoring": bool(scoring.get("n_valid", 0)),
            "invalid_reason": "all_questions_invalid" if not scoring.get("n_valid", 0) else None,
            "validity_policy": "parse_failed_or_missing_answers_excluded",
            "fallback_filled_count": 0,
            "fallback_filled_indices": [],
            "batch_records": sorted(batch_records, key=lambda x: int(x.get("batch_idx", 0)))}


# ===========================================================================
# 3F time: reuse F's prompt / parsing / scoring (default average of 2 passes)
# ===========================================================================

def time_point(pid, qb_path, ctx_path, framework, base,
               max_tokens, thinking_budget, runs=1, max_e=None,
               answer_fn=None, embed_context=True, parallel_runs=True):
    """Get one prediction point's time predictions and score them (structure reuses F's parsing/scoring 100%).

    answer_fn / embed_context are as in brier_point. Under session semantics it's "1 rollout,
    readout sampled runs times": answer_fn is called runs times on the same session state
    (reader temperature>0 provides randomness), so parallel_runs=False samples serially
    (sessions are not thread-safe).
    """
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    cutoff = qb.get("cutoff_date") or ""
    ev = qb.get("events", {})
    allev = ev.get("all", []) if isinstance(ev, dict) else ev
    events = [e for e in allev if int(e.get("days_from_cutoff", 0) or 0) <= F.WINDOW_DAYS]
    if max_e:
        events = events[:max_e]
    slug = cfg_slug(framework, base)
    if not events:
        return {"point_id": pid, "model": slug, "events_total": 0, "score_100": None,
                "events": [], "predictions": [], "per_run": []}
    context = (ctx_path.read_text(encoding="utf-8")
               if (ctx_path.exists() and embed_context) else "")

    n_ev = len(events)
    cutoff_d = parse_date_obj(cutoff)

    def _build_prompt(sub_events):
        # Aligned with main-experiment 3F on 2026-07-01: default to the B-style exam template (explains the scoring principle, encourages careful reasoning).
        if os.environ.get("SB_EXAM_PROMPT", "1") != "0":
            return F.EXAM_PROMPT_TEMPLATE.format(
                n=len(sub_events), cutoff=cutoff, context=context,
                event_list=F.render_event_list(sub_events, cutoff),
            )
        return F.PROMPT_TEMPLATE.format(
            n=len(sub_events), context=context,
            event_list=F.render_event_list(sub_events, cutoff),
        )

    # v3 (decided 2026-07-02, aligned with bare-model 3F): time readout also gets "point-level
    # retry" — each retry only re-asks the events still missing a date / with failed requests,
    # at most MAX_ANSWER_ATTEMPTS times (+a 4th if dmx questions remain after the 3rd).
    # Failure classes: readout exception = dmx / no parseable date = model. Fixes the 2026-07-02
    # gap: the agent time path previously asked once and declared parse failures dead without
    # retrying (brier and bare-model time both retried; only this path was missing, so one
    # unparseable Doubao output killed the whole point as model without re-asking). Answering
    # goes through answer_fn (session readout, no simulation rerun).
    _MAX_ATT = int(getattr(F, "MAX_ANSWER_ATTEMPTS", 3))
    _WAIT = int(getattr(F, "RETRY_WAIT_SECONDS", 60))
    merged_dates: Dict[str, str] = {}
    last_fail_kind = [None] * n_ev
    answer_attempt = [None] * n_ev
    attempt_records = []
    for _run in range(max(1, runs)):
        missing = set(i for i in range(n_ev) if str(i + 1) not in merged_dates)
        attempt = 0
        while missing:
            if attempt >= _MAX_ATT + 1:
                break
            if attempt >= _MAX_ATT:
                if not any((last_fail_kind[i] or "").startswith("request_error") for i in missing):
                    break  # MAX_ATT attempts done and no dmx questions -> stop (no extra +1 attempt)
            attempt += 1
            if attempt > 1 and _WAIT > 0:
                print(f"  time {slug} {pid} run{_run + 1} attempt {attempt}: 等待 {_WAIT}s 后重问 "
                      f"{len(missing)} 个未给日期事件 ...", flush=True)
                time.sleep(_WAIT)
            sub_idx = sorted(missing)
            sub_events = [events[i] for i in sub_idx]
            prompt = _build_prompt(sub_events)
            kind = None
            try:
                if answer_fn is not None:
                    raw = answer_fn(prompt)  # session: read off the same rollout state (no simulation rerun)
                else:
                    raw = agent_answer(framework, base, prompt, "time", max_tokens, thinking_budget)
            except Exception as e:  # noqa: BLE001  readout timeout/exception = DMX class
                raw = ""
                kind = f"request_error:{str(e)[:120]}"
                print(f"  time readout {slug} {pid} run{_run + 1} attempt {attempt} 请求失效: {e}", flush=True)
            sub_dates = F.parse_dates(raw)
            n_got = 0
            for pos, orig in enumerate(sub_idx, start=1):
                ds = sub_dates.get(str(pos))
                if ds:
                    merged_dates[str(orig + 1)] = ds
                    answer_attempt[orig] = attempt
                    missing.discard(orig)
                    n_got += 1
                else:
                    last_fail_kind[orig] = kind or "parse_fail"
            attempt_records.append({"run": _run + 1, "attempt": attempt, "asked": len(sub_idx),
                                    "answered": n_got, "still_missing": len(missing),
                                    "request_error": 1 if kind else 0})
    # v3: failed-event classification (request_error/quota=dmx / parse_fail=model); main score excludes them, secondary score fills midpoints
    fail_kinds = [None] * n_ev
    for i in range(n_ev):
        if str(i + 1) not in merged_dates:
            lk = last_fail_kind[i] or ""
            fail_kinds[i] = "dmx" if lk.startswith(("request_error", "quota")) else "model"
    n_dmx = sum(1 for fk in fail_kinds if fk == "dmx")
    n_model = sum(1 for fk in fail_kinds if fk == "model")
    _att_hist = {}
    for a in answer_attempt:
        if a is not None:
            _att_hist[str(a)] = _att_hist.get(str(a), 0) + 1
    scoring_ans = F.score_events(events, merged_dates)               # main score: answered only
    withdef = dict(merged_dates)                                     # secondary score: failures filled with segment midpoints
    for i in range(n_ev):
        if str(i + 1) not in withdef:
            d = int(events[i].get("days_from_cutoff", 0) or 0)
            mid = int(round(F.segment_midpoint(d)))
            if cutoff_d:
                withdef[str(i + 1)] = (cutoff_d + timedelta(days=mid)).isoformat()
    scoring_def = F.score_events(events, withdef)

    ans_details = scoring_ans.get("details", [])
    def_details = scoring_def.get("details", [])
    legacy = []
    for i, e in enumerate(events):
        gt_d = parse_date_obj(e.get("date") or e.get("gt_date"))
        delta = (gt_d - cutoff_d).days if (gt_d and cutoff_d) else 0
        det = ans_details[i] if i < len(ans_details) else {}
        det_def = def_details[i] if i < len(def_details) else {}
        answered = str(i + 1) in merged_dates
        legacy.append({
            "eid": str(e.get("id") or e.get("eid") or (i + 1)),
            "event_desc": e.get("event") or e.get("event_desc", ""),
            "gt_date": (e.get("date") or e.get("gt_date") or ""),
            "pred_date": det.get("pred_date") or "",
            "error_days": det.get("error_days") if det.get("error_days") is not None else 999,
            "error_days_with_default": det_def.get("error_days"),
            "delta_days": delta,
            "answered": answered,
            "fail_kind": None if answered else fail_kinds[i],
            "answer_attempt": answer_attempt[i],
        })
    dmx_frac = n_dmx / n_ev if n_ev else 0.0
    model_frac = n_model / n_ev if n_ev else 0.0
    return {"point_id": pid, "model": slug, "cutoff_date": cutoff,
            "events_total": n_ev,
            "weighted_mae_days": scoring_ans.get("weighted_mae_days"),
            "baseline_wmae": scoring_ans.get("baseline_wmae"),
            # main = score_answered (answered only); secondary = score_with_default (failures filled with midpoints)
            "score_100": scoring_ans.get("score_100"),
            "score_answered": scoring_ans.get("score_100"),
            "score_with_default": scoring_def.get("score_100"),
            "weighted_mae_with_default": scoring_def.get("weighted_mae_days"),
            "predictions": [le["pred_date"] for le in legacy],
            # The three invalid-fraction entries (decided 2026-07-02)
            "dmx_error_frac": round(dmx_frac, 4),
            "model_fail_frac": round(model_frac, 4),
            "invalid_frac": round(dmx_frac + model_frac, 4),
            # Accounting of the readout point-level retries (added 2026-07-02; fields aligned with bare-model 3F)
            "max_answer_attempts": _MAX_ATT,
            "retry_wait_seconds": _WAIT,
            "attempt_records": attempt_records,
            "answer_attempt_histogram": _att_hist,
            "n_runs_completed": max(1, runs),
            "events": legacy, "per_run": [scoring_ans]}


def _all_dmx_brier_record(pid, slug, cutoff, qb_path, cal_cfg, max_q=None):
    """advance rollout failed -> all of this point's Brier questions marked dmx-invalid.
    Reuses B.score_point (all None probs, fail_kinds all 'dmx') to produce **complete details**
    so downstream aggregate_across_points correctly counts dmx_error_frac/invalid_frac and the
    secondary score (fixes 2026-07-02 bug: the old version wrote empty events/questions/details
    -> aggregation silently swallowed this point's failure)."""
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    questions = qb.get("brier_questions") or []
    if max_q:
        questions = questions[:max_q]
    n = len(questions)
    r = B.score_point(questions, [None] * n, cal_cfg,
                      fail_kinds=["dmx"] * n, answer_attempts=[None] * n)
    r.update({"point_id": pid, "model": slug, "cutoff_date": cutoff,
              "valid_for_scoring": False, "invalid_reason": "advance_dmx_failed",
              "questions": questions})
    return r


def _all_dmx_time_record(pid, slug, cutoff, qb_path, max_e=None):
    """advance rollout failed -> all of this point's time events marked dmx-invalid, producing
    **complete legacy events** (error_days=999, error_days_with_default=midpoint error,
    fail_kind='dmx') so the pooled pooled_score correctly counts dmx_error_frac/invalid_frac
    and the secondary score. Same bug fix as above."""
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    ev = qb.get("events", {})
    allev = ev.get("all", []) if isinstance(ev, dict) else ev
    events = [e for e in allev if int(e.get("days_from_cutoff", 0) or 0) <= F.WINDOW_DAYS]
    if max_e:
        events = events[:max_e]
    n_ev = len(events)
    cutoff_d = parse_date_obj(cutoff)
    withdef: Dict[str, str] = {}                    # nothing answered -> secondary score fills segment midpoints
    for i in range(n_ev):
        d = int(events[i].get("days_from_cutoff", 0) or 0)
        mid = int(round(F.segment_midpoint(d)))
        if cutoff_d:
            withdef[str(i + 1)] = (cutoff_d + timedelta(days=mid)).isoformat()
    scoring_ans = F.score_events(events, {})         # main score: nothing answered
    scoring_def = F.score_events(events, withdef)    # secondary score: midpoint fill
    def_details = scoring_def.get("details", [])
    legacy = []
    for i, e in enumerate(events):
        gt_d = parse_date_obj(e.get("date") or e.get("gt_date"))
        delta = (gt_d - cutoff_d).days if (gt_d and cutoff_d) else 0
        det_def = def_details[i] if i < len(def_details) else {}
        legacy.append({
            "eid": str(e.get("id") or e.get("eid") or (i + 1)),
            "event_desc": e.get("event") or e.get("event_desc", ""),
            "gt_date": (e.get("date") or e.get("gt_date") or ""),
            "pred_date": "",
            "error_days": 999,
            "error_days_with_default": det_def.get("error_days"),
            "delta_days": delta,
            "answered": False,
            "fail_kind": "dmx",
        })
    return {"point_id": pid, "model": slug, "cutoff_date": cutoff,
            "events_total": n_ev,
            "weighted_mae_days": scoring_ans.get("weighted_mae_days"),
            "baseline_wmae": scoring_ans.get("baseline_wmae"),
            "score_100": None, "score_answered": None,
            "score_with_default": scoring_def.get("score_100"),
            "weighted_mae_with_default": scoring_def.get("weighted_mae_days"),
            "predictions": ["" for _ in legacy],
            "dmx_error_frac": 1.0 if n_ev else 0.0,
            "model_fail_frac": 0.0,
            "invalid_frac": 1.0 if n_ev else 0.0,
            "valid_for_scoring": False, "invalid_reason": "advance_dmx_failed",
            "events": legacy, "per_run": [scoring_ans]}


# ===========================================================================
# Continuous rollout: one session per (framework, base model), iterating points serially in ascending pid order
# ===========================================================================

def _fail_kind_from_err(err):
    """v3 per-question failure classification: parse failure (parse_fail)=model; everything else (batch/single request error, timeout)=dmx."""
    return "model" if (err and "parse_fail" in str(err)) else "dmx"


def _write_incident(ws, slug, first_pid, invalid_pids, err):
    """advance chain-break incident: write a prominent manifest at the workspace root so the
    config can be manually redone after the run. One file per config (slug unique), avoiding parallel write races."""
    try:
        p = pathlib.Path(ws) / f"_INCIDENT_{slug}.json"
        p.write_text(json.dumps({
            "type": "advance_failed_incident",
            "config": slug,
            "first_failed_point": first_pid,
            "invalidated_points": invalid_pids,
            "error": err,
            "action_required": "推演链断裂,该 config 需人工重做(全部跑完后手动补)",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [session] 事故清单已写: {p}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [session] 写事故清单失败: {e}", flush=True)


# advance-failure retry (decided 2026-07-01): same as the bare-model "3 attempts" logic — a
# point's rollout fails -> cool down SB_AGENT_ADVANCE_COOLDOWN seconds (default 10s) and retry,
# at most 3 times; still failing after 3 = rollout chain broken, session cannot continue.
ADVANCE_MAX_ATTEMPTS = int(os.environ.get("SB_AGENT_ADVANCE_MAX_ATTEMPTS", "4"))  # 2026-07-02; env-tunable (policy B "persist at the breakpoint" sets it very large -> almost never declares an incident)
ADVANCE_COOLDOWN_SECONDS = int(os.environ.get("SB_AGENT_ADVANCE_COOLDOWN", "10"))
# brier answer-parse retry caps (env-tunable; defaults follow the B module values / 2). Under policy B set very large -> keep re-asking in place instead of declaring dead
_SINGLE_REMAKE_ATT = int(os.environ.get("SB_AGENT_SINGLE_REMAKE_ATTEMPTS", str(getattr(B, "SINGLE_QUESTION_REMAKE_ATTEMPTS", 2))))
_BATCH_REMAKE_ATT = int(os.environ.get("SB_AGENT_BATCH_REMAKE_ATTEMPTS", str(getattr(B, "BATCH_REMAKE_ATTEMPTS", 2))))


def run_event_session(fw, base, qb_files_sorted, ctx_dir, axes, run_id, ws,
                      cal_cfg, args, batch_size, coverage_only):
    """For one (fw, base): start a continuous rollout session, advance along the timeline, reading out each point's answers along the way.

    Returns (out, sess_stats):
      out        = [(axis, slug, pid, result_dict), ...] (same shape as legacy work(), for aggregation)
      sess_stats = {advance, answer, llm, sim} (this session's call counts, evidencing the rollout-count reduction)
    """
    slug = cfg_slug(fw, base)
    out = []
    # Information-parity switch (**default 1=on**, team-lead decision): the readout also embeds
    # the full context for that cutoff (the same "known information" as bare models, with belief
    # layered on top as the ongoing analysis), guaranteeing agent information = bare model and
    # never starved by belief compression; still satisfies "1 rollout per event" (belief is one
    # continuous chain; the readout seeing the source text is just answering reference, not a
    # repeated rollout). Knowledge lives in the session (belief) + this point's real cutoff context.
    # Set SB_AGENT_SESSION_CARRY_CONTEXT=0 to revert to pure belief.
    carry_ctx = os.environ.get("SB_AGENT_SESSION_CARRY_CONTEXT", "1") != "0"

    # Pre-scan: per point per axis, decide "must compute" vs "reusable". If everything is reusable, skip the whole session (not even paying for advance).
    plan = []
    any_need = False
    for qbf in qb_files_sorted:
        pid = qbf.stem.replace("_questionbank", "")
        rec = {"pid": pid, "qbf": qbf, "ctxf": ctx_dir / f"{pid}_context.md"}
        try:
            rec["cutoff"] = (json.loads(qbf.read_text(encoding="utf-8")).get("cutoff_date") or "")
        except Exception:
            rec["cutoff"] = ""
        if "brier" in axes:
            fp = ws / "results" / f"run_{run_id}" / "brier" / slug / f"{pid}_brier.json"
            rec["brier_fp"] = fp
            rec["brier_reuse"] = reusable_brier(fp, coverage_only=coverage_only)
            if rec["brier_reuse"] is None:
                any_need = True
        if "time" in axes:
            fp = ws / "results" / f"run_{run_id}" / "time" / slug / f"{pid}_time.json"
            rec["time_fp"] = fp
            rec["time_reuse"] = reusable_time(fp, args.runs_per_point, coverage_only=coverage_only)
            if rec["time_reuse"] is None:
                any_need = True
        plan.append(rec)

    sess = None
    if any_need:
        sim_params = None
        if fw.lower() == "mirofish":
            sim_params = {"n_agents": int(os.environ.get("MIROFISH_AGENTS", "8")),
                          "n_steps": int(os.environ.get("MIROFISH_STEPS", "3"))}
        sess = make_session(fw, base, max_tokens=args.max_tokens,
                            thinking_budget=args.thinking_budget, sim_params=sim_params)
        print(f"  [session] start {slug}: {len(plan)} points (need-compute), one continuous reasoning")
    else:
        print(f"  [session] {slug}: all points reusable, skip reasoning entirely")

    prev_ctx_text, prev_cutoff = "", ""
    sess_stats = {"advance": 0, "answer": 0, "llm": 0, "sim": 0}
    try:
        for _pi, rec in enumerate(plan):
            pid, cutoff, qbf, ctxf = rec["pid"], rec["cutoff"], rec["qbf"], rec["ctxf"]
            cur_ctx_text = ctxf.read_text(encoding="utf-8") if ctxf.exists() else ""
            if sess is not None:
                delta = compute_delta(prev_ctx_text, cur_ctx_text, prev_cutoff, cutoff)
                # Advance to this point: on failure -> cool down 10s and retry, at most 3 times
                # (same as the bare-model 3-attempt logic); still failing after 3 = rollout chain
                # broken, session cannot continue (raise; the caller records this config as failed).
                _adv_err = None
                for _att in range(1, ADVANCE_MAX_ATTEMPTS + 1):
                    try:
                        sess.advance(delta, cutoff)
                        break
                    except Exception as e:  # noqa: BLE001
                        _adv_err = e
                        print(f"  [session] {slug} 推演到 {pid} 失败 (第 {_att}/{ADVANCE_MAX_ATTEMPTS} 次): {e}", flush=True)
                        if _att < ADVANCE_MAX_ATTEMPTS:
                            time.sleep(ADVANCE_COOLDOWN_SECONDS)
                else:
                    # Repeated advance crashes = rollout chain broken (classified 2026-07-02 as an
                    # "incident", very rare): all later points in this session would build on a
                    # defective understanding missing this point's knowledge -> this point + **all
                    # later points** marked invalid (incident), stop the session (no more tokens
                    # burned on defective rollout), write a prominent incident manifest, and
                    # manually redo the config after everything finishes. Points answered before
                    # the break stay valid.
                    remaining = plan[_pi:]
                    print(f"  [session] ⚠️事故 {slug} 在 {pid} 推演 {ADVANCE_MAX_ATTEMPTS} 次失败 → "
                          f"本点及之后 {len(remaining)} 点全判无效、停会话、记事故待人工重做: {_adv_err}", flush=True)
                    _write_incident(ws, slug, pid, [r["pid"] for r in remaining], str(_adv_err))
                    for _rr in remaining:
                        for ax, fpk in (("brier", "brier_fp"), ("time", "time_fp")):
                            if ax in axes and _rr.get(fpk):
                                if ax == "brier":
                                    inv = _all_dmx_brier_record(_rr["pid"], slug, _rr["cutoff"], _rr["qbf"],
                                                                cal_cfg, args.max_questions)
                                else:
                                    inv = _all_dmx_time_record(_rr["pid"], slug, _rr["cutoff"], _rr["qbf"],
                                                               args.max_events)
                                inv["invalid_reason"] = "advance_failed_incident"
                                inv["incident"] = True
                                _rr[fpk].parent.mkdir(parents=True, exist_ok=True)
                                _rr[fpk].write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
                                out.append((ax, slug, _rr["pid"], inv))
                    break
            prev_ctx_text, prev_cutoff = cur_ctx_text, cutoff

            # Note: the session path does **not** use base_fallback_point (that writes the
            # agent_fallback marker, which the gate rejects outright). Mid-run failures are handled
            # inside each framework's Session (folding to single calls etc., producing genuine
            # agent answers without fallback markers).
            if "brier" in axes:
                fp, r = rec["brier_fp"], rec["brier_reuse"]
                if r is None:
                    existing_bad = None
                    if fp.exists():
                        try:
                            existing_bad = json.loads(fp.read_text(encoding="utf-8"))
                        except Exception:
                            existing_bad = None
                    r = brier_point(pid, qbf, ctxf, fw, base, cal_cfg,
                                    args.max_tokens, args.thinking_budget, batch_size,
                                    args.max_questions, max_parallel=1,
                                    existing_bad=existing_bad,
                                    answer_fn=(lambda p: sess.answer(p, "brier")),
                                    embed_context=carry_ctx)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(f"  reuse brier {slug} {pid}")
                out.append(("brier", slug, pid, r))

            if "time" in axes:
                fp, r = rec["time_fp"], rec["time_reuse"]
                if r is None:
                    try:
                        r = time_point(pid, qbf, ctxf, fw, base,
                                       args.max_tokens, args.thinking_budget,
                                       args.runs_per_point, args.max_events,
                                       answer_fn=(lambda p: sess.answer(p, "time")),
                                       embed_context=carry_ctx, parallel_runs=False)
                    except Exception as e:  # noqa: BLE001  point-level isolation: time timeout/exception doesn't drag down the whole session
                        print(f"  [session] {slug} {pid} time 点级失败/超时 → 该点判无效排除、续跑后续点: {e}", flush=True)
                        r = {"point_id": pid, "model": slug, "cutoff_date": cutoff,
                             "events_total": 0, "score_100": None, "events": [], "per_run": [],
                             "valid_for_scoring": False, "invalid_reason": "timeout_or_error"}
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(f"  reuse time {slug} {pid}")
                out.append(("time", slug, pid, r))

        if sess is not None:
            sess_stats = sess.stats()
    finally:
        if sess is not None:
            try:
                sess.close()
            except Exception as e:  # noqa: BLE001
                print(f"  [session] {slug} close warn: {e}")
    print(f"  [session] done {slug}: stats={sess_stats}")
    return out, sess_stats


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Agent evaluation (M3): framework x base model, reusing 3B/3F scoring")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--event-name", required=True)
    ap.add_argument("--frameworks", required=True, help="comma-separated: langgraph,autogen,mirofish")
    ap.add_argument("--bases", required=True, help="comma-separated base model ids")
    ap.add_argument("--axes", default="brier,time")
    ap.add_argument("--points", default="all")
    ap.add_argument("--runs-per-point", type=int, default=1, help="How many 3F passes per point (decided 2026-07-01: 1; the 1-vs-2 mean difference < between-run noise)")
    ap.add_argument("--max-questions", type=int, default=None, help="For smoke tests: truncate calibration questions per point")
    ap.add_argument("--max-events", type=int, default=None, help="For smoke tests: truncate time events per point")
    ap.add_argument("--batch-size", type=int, default=None, help="3B batch size per point; defaults to pipeline_config")
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--thinking-budget", type=int, default=24000)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--coverage-only", action="store_true",
                    help="Reuse existing agent point files even if they are fallback/bad; only fill missing coverage.")
    ap.add_argument("--legacy-per-point", action="store_true",
                    help="Use the old stateless per-point path (each point reruns the full rollout from scratch); for old-vs-new comparison / rollback. Default is the new single-continuous-rollout session path.")
    args = ap.parse_args()
    coverage_only = args.coverage_only or _COVERAGE_ONLY_ENV

    cfg = load_pipeline_config()
    cal_cfg = cfg.get("scoring_calibration", {})
    ws = pathlib.Path(args.workspace)
    # Aligned with the main experiment on 2026-07-01: English workspace -> agents also switch to
    # English prompts (B's ANSWER_PROMPT / F's PROMPT_TEMPLATE and EXAM_PROMPT_TEMPLATE).
    # Otherwise English M3 would use Chinese instructions, a different setup from English M1, not comparable.
    if "英文" in ws.parts:
        B.LANG = "en"
        B.ANSWER_PROMPT, B.RETRY_PROMPT = B.ANSWER_PROMPT_EN, B.RETRY_PROMPT_EN
        F.PROMPT_TEMPLATE, F.EXAM_PROMPT_TEMPLATE = F.PROMPT_TEMPLATE_EN, F.EXAM_PROMPT_TEMPLATE_EN
        # 2026-07-02: the three frameworks' internal rollout scaffolding (belief/roles/persona/
        # event bulletin/background extraction) also switches to English, passed via env to each
        # framework and the mirofish subprocess, so in English workspaces the model's full prompt is all English.
        os.environ["SB_AGENT_LANG"] = "en"
        print("[run_agents] 英文工作区 → 已切换 B/F 英文 prompt + SB_AGENT_LANG=en(三框架脚手架全英文)")
    qb_dir, ctx_dir = ws / "questionbank", ws / "contexts"
    if not qb_dir.exists():
        raise SystemExit(f"No questionbank/ under {ws}")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = args.batch_size or _DEFAULT_BATCH_SIZE
    batch_size_for_metadata = None if batch_size >= 999999 else batch_size

    qb_files = sorted(qb_dir.glob("P*_questionbank.json"))
    if args.points != "all":
        want = set(args.points.split(","))
        qb_files = [f for f in qb_files if f.stem.replace("_questionbank", "") in want]

    frameworks = [x.strip() for x in args.frameworks.split(",") if x.strip()]
    bases = [x.strip() for x in args.bases.split(",") if x.strip()]
    axes = [x.strip() for x in args.axes.split(",") if x.strip()]

    # Collect: results_by[(slug, axis)][pid] = result
    results_by: dict = {}
    adapter.STATS.reset()
    mode = "legacy-per-point" if args.legacy_per_point else "session(连续推演)"
    configs = [cfg_slug(fw, b) for fw in frameworks for b in bases]
    print(
        f"[run_agents] run_id={run_id} mode={mode} configs={configs} points={len(qb_files)} "
        f"axes={axes} batch_size={batch_size} coverage_only={coverage_only}"
    )

    if args.legacy_per_point:
        # ---- Old path: (framework x base x point) Cartesian product; each point statelessly calls agent_answer, rerunning the full rollout from scratch ----
        tasks = []
        for fw in frameworks:
            for base in bases:
                for qbf in qb_files:
                    pid = qbf.stem.replace("_questionbank", "")
                    tasks.append((fw, base, pid, qbf, ctx_dir / f"{pid}_context.md"))

        def work(fw, base, pid, qbf, ctxf):
            slug = cfg_slug(fw, base)
            out = []
            if "brier" in axes:
                fp = ws / "results" / f"run_{run_id}" / "brier" / slug / f"{pid}_brier.json"
                r = reusable_brier(fp, coverage_only=coverage_only)
                if r is None:
                    r = base_fallback_point(ws, run_id, "brier", fw, base, slug, pid, args.runs_per_point)
                    if r is None:
                        existing_bad = None
                        if fp.exists():
                            try:
                                existing_bad = json.loads(fp.read_text(encoding="utf-8"))
                            except Exception:
                                existing_bad = None
                        r = brier_point(pid, qbf, ctxf, fw, base, cal_cfg,
                                        args.max_tokens, args.thinking_budget, batch_size,
                                        args.max_questions, args.max_parallel,
                                        existing_bad=existing_bad)
                    else:
                        print(f"  fallback brier {slug} {pid} from base {B.model_slug(base)}")
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(f"  reuse brier {slug} {pid}")
                out.append(("brier", slug, pid, r))
            if "time" in axes:
                fp = ws / "results" / f"run_{run_id}" / "time" / slug / f"{pid}_time.json"
                r = reusable_time(fp, args.runs_per_point, coverage_only=coverage_only)
                if r is None:
                    r = base_fallback_point(ws, run_id, "time", fw, base, slug, pid, args.runs_per_point)
                    if r is None:
                        r = time_point(pid, qbf, ctxf, fw, base,
                                       args.max_tokens, args.thinking_budget,
                                       args.runs_per_point, args.max_events)
                    else:
                        print(f"  fallback time {slug} {pid} from base {B.model_slug(base)}")
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(f"  reuse time {slug} {pid}")
                out.append(("time", slug, pid, r))
            return out

        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = [ex.submit(work, *t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    for axis, slug, pid, r in fut.result():
                        results_by.setdefault((slug, axis), {})[pid] = r
                        print(f"  ✓ {axis} {slug} {pid} score={r.get('score_100')}")
                except Exception as e:
                    print(f"  ✗ task error: {e}")
                    traceback.print_exc()
    else:
        # ---- New path: one continuous rollout session per (framework x base); concurrency across sessions, serial points within a session ----
        # Decided 2026-07-01 ("split B/F"): each "framework x base x axis" gets its own independent
        # session (each reads once), preventing F questions from "spoiling real events" (the prompt
        # explicitly says "these events will happen") and polluting B via the shared session context.
        # Concurrency across sessions, serial points within a session (continuous rollout).
        # Amended 2026-07-03: **mirofish exception — B/F share the same social simulation**.
        # Its ANSWER is read-only (the reader is stateless: assembles the prompt independently each
        # time, never writes the transcript, never steps the simulation — verified at code level),
        # so F questions have no session context through which to pollute B, consistent with the
        # "split B/F" motivation; sharing cuts simulations per combo 2->1 (memory/CPU/netizen calls
        # halved). langgraph/autogen keep the per-axis split.
        session_units = []
        for fw in frameworks:
            for base in bases:
                if fw.lower() == "mirofish" and len(axes) > 1:
                    session_units.append((fw, base, list(axes)))
                else:
                    for axis in axes:
                        session_units.append((fw, base, [axis]))
        sess_stats_by = {}
        with ThreadPoolExecutor(max_workers=max(1, min(args.max_parallel, len(session_units)))) as ex:
            futs = {
                ex.submit(run_event_session, fw, base, qb_files, ctx_dir, ax_list,
                          run_id, ws, cal_cfg, args, batch_size, coverage_only): f"{cfg_slug(fw, base)}[{'+'.join(ax_list)}]"
                for fw, base, ax_list in session_units
            }
            for fut in as_completed(futs):
                try:
                    out, sstats = fut.result()
                    for axis, slug, pid, r in out:
                        results_by.setdefault((slug, axis), {})[pid] = r
                        print(f"  ✓ {axis} {slug} {pid} score={r.get('score_100')}")
                    sess_stats_by[futs[fut]] = sstats
                except Exception as e:
                    print(f"  ✗ session error ({futs[fut]}): {e}")
                    traceback.print_exc()
        # Measured rollout counts: each session only advances (=continuous reasoning) some times + answers (readout) some times
        total = {"advance": 0, "answer": 0, "llm": 0, "sim": 0}
        for slug, s in sorted(sess_stats_by.items()):
            for k in total:
                total[k] += int(s.get(k, 0) or 0)
            print(f"  [stats] {slug}: advance={s.get('advance')} answer={s.get('answer')} "
                  f"llm={s.get('llm')} sim={s.get('sim')}")
        print(f"  [stats] TOTAL session reasoning: {total}  "
              f"(每事件×config 仅 1 条连续推演;对比 legacy 每点独立 episode)")

    # Write aggregated.json per config (reusing B/F aggregation), for the step4 scorecard
    if coverage_only:
        for slug in configs:
            if "brier" in axes:
                outdir = ws / "results" / f"run_{run_id}" / "brier" / slug
                for path in sorted(outdir.glob("P*_brier.json")):
                    pid = path.stem.replace("_brier", "")
                    if pid in results_by.get((slug, "brier"), {}):
                        continue
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if int(data.get("n") or len(data.get("questions") or []) or len(data.get("details") or []) or 0) > 0:
                        results_by.setdefault((slug, "brier"), {})[pid] = data
            if "time" in axes:
                outdir = ws / "results" / f"run_{run_id}" / "time" / slug
                for path in sorted(outdir.glob("P*_time.json")):
                    pid = path.stem.replace("_time", "")
                    if pid in results_by.get((slug, "time"), {}):
                        continue
                    try:
                        results_by.setdefault((slug, "time"), {})[pid] = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
    for (slug, axis), per_point in results_by.items():
        outdir = ws / "results" / f"run_{run_id}" / axis / slug
        if axis == "brier":
            agg = B.aggregate_across_points(per_point, cal_cfg)
            agg_batch_size, batch_size_by_point, batch_size_source = B.summarize_point_batch_size(
                per_point,
                batch_size_for_metadata,
            )
            payload = {"model": slug, "n_points": len(per_point),
                       "batch_size": agg_batch_size,
                       "batch_size_source": batch_size_source,
                       "batch_size_by_point": batch_size_by_point,
                       "validity_policy": "parse_failed_or_missing_answers_excluded",
                       "fallback_filled_count": 0,
                       "aggregated": agg,
                       "per_point": {p: {k: v for k, v in r.items() if k not in ("questions", "details")}
                                     for p, r in per_point.items()}}
        else:
            agg = F.pooled_score(per_point)
            payload = {"model": slug, "n_points": len(per_point),
                       "batch_size": None,
                       "pooled": agg,
                       "per_point": {p: {k: v for k, v in r.items() if k not in ("events", "per_run")}
                                     for p, r in per_point.items()}}
        (outdir / "aggregated.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Σ {axis} {slug}: {agg}")

    # Measured rollout-count accounting (for old-vs-new comparison):
    #   legacy mode  -> episode = agent_answer call count (= independent "full rollout" count per point)
    #   session mode -> advance (continuous rollout steps) + answer (readouts) + llm (underlying requests) + sim (social simulation launches)
    print(json.dumps({"run_id": run_id, "run_dir": str(ws / 'results' / f'run_{run_id}'),
                      "configs": configs, "mode": mode,
                      "reasoning_stats": adapter.STATS.as_dict()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
