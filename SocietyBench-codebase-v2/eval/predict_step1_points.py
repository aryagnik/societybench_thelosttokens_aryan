#!/usr/bin/env python3
"""predict-step1-points (1:1 of skills/predict-step1-points/SKILL.md).

Mostly pure algorithm. Parses date nodes from a refined timeline, applies the
interval rule (≤threshold days → candidate), trims by min-context /
min-future, performs stage-aware quota convergence. Optional opt-in: LLM
quality review (SKILL §Step 2.1) — pass --quality-review-model to score each
candidate target node 1-5 and demote weak ones before quota allocation.
Outputs three files:

    <output_dir>/prediction_points.json   — meta + nodes + prediction_points
    <output_dir>/prediction_points_summary.md — human-readable summary
    <output_dir>/timeline_with_points.md   — input timeline + [Pxx] markers

The input timeline is treated as **immutable**: the only allowed modification
in `timeline_with_points.md` is appending `[Pxx]` to date headers.

Usage:
    python predict_step1_points.py <timeline_md> <output_dir> \\
        [--threshold-days 14] [--min-context 1] [--min-future 0] \\
        [--max-points 30] [--target-points 25] [--pipeline-config <path>]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import call_llm, load_pipeline_config


STAGE_RE = re.compile(r"^##\s+(阶段.+|Stage.+|Phase.+)$")
# Allow:  ### 2025-07-26
#         ### 2024-02-04 ~ 02-05
#         ### 2024-02-04 ~ 2024-02-05
DATE_RE = re.compile(
    r"^###\s+(\d{4}-\d{2}-\d{2}(?:\s*~\s*(?:\d{4}-)?\d{2}(?:-\d{2})?)?)\s*$"
)


# ===========================================================================
# Parsing
# ===========================================================================

def parse_date(date_str: str) -> Tuple[datetime, str]:
    """Return (end-date as datetime, original_raw_text).

    Single:  `2025-07-26`              → end = 2025-07-26
    Range:   `2024-02-04 ~ 02-05`      → end = 2024-02-05
             `2024-02-04 ~ 2024-02-05` → end = 2024-02-05
             `2024-12-30 ~ 01-02`      → end = 2024-01-02 (caller responsible
                                          if year wraps; we use start's year)
    """
    raw = date_str.strip()
    if "~" in raw:
        start, end = [s.strip() for s in raw.split("~", 1)]
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        if len(end) <= 2:  # "05"
            end_full = f"{start_dt.year}-{start_dt.month:02d}-{end}"
        elif len(end) <= 5:  # "02-05"
            end_full = f"{start_dt.year}-{end}"
        else:  # "2024-02-05"
            end_full = end
        return datetime.strptime(end_full, "%Y-%m-%d"), raw
    return datetime.strptime(raw, "%Y-%m-%d"), raw


def parse_timeline(md_path: Path) -> List[Dict[str, Any]]:
    """Extract `### YYYY-MM-DD` headers, attaching the current stage and line range."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    nodes: List[Dict[str, Any]] = []
    current_stage: Optional[str] = None
    for i, line in enumerate(lines):
        line_stripped = line.rstrip()
        m_stage = STAGE_RE.match(line_stripped)
        if m_stage:
            current_stage = m_stage.group(1)
            continue
        m_date = DATE_RE.match(line_stripped)
        if m_date:
            date_raw = m_date.group(1)
            date_obj, _ = parse_date(date_raw)
            nodes.append({
                "date_raw": date_raw,
                "date": date_obj.strftime("%Y-%m-%d"),
                "_date_obj": date_obj,
                "stage": current_stage or "未知阶段",
                "line_start": i + 1,
                "line_end": None,  # filled later
                "original_line": line,
            })

    for k in range(len(nodes) - 1):
        nodes[k]["line_end"] = nodes[k + 1]["line_start"] - 1
    if nodes:
        nodes[-1]["line_end"] = len(lines)
    return nodes


# ===========================================================================
# Selection
# ===========================================================================

def _stage_quota(
    candidates_by_stage: Dict[str, List[int]],
    target_points: int,
) -> Dict[str, int]:
    """Allocate per-stage quota: each populated stage keeps at least 1, then
    distribute the remainder proportionally to candidate density."""
    stages = list(candidates_by_stage.keys())
    if not stages:
        return {}
    total_candidates = sum(len(v) for v in candidates_by_stage.values())
    if total_candidates <= target_points:
        return {s: len(candidates_by_stage[s]) for s in stages}

    base = {s: 1 for s in stages}
    remaining = target_points - len(stages)
    if remaining < 0:
        # too many stages, force fairness: top N stages by density get 1 each
        ranked = sorted(stages, key=lambda s: -len(candidates_by_stage[s]))
        return {s: (1 if s in ranked[:target_points] else 0) for s in stages}

    # Proportional allocation of the remainder
    weights = {s: len(candidates_by_stage[s]) for s in stages}
    total_w = sum(weights.values()) or 1
    allocs = {s: weights[s] * remaining / total_w for s in stages}
    # Largest-remainder rounding
    floored = {s: int(allocs[s]) for s in stages}
    leftover = remaining - sum(floored.values())
    fractions = sorted(
        ((s, allocs[s] - floored[s]) for s in stages),
        key=lambda t: -t[1],
    )
    for s, _ in fractions[:leftover]:
        floored[s] += 1

    # Add the base 1
    return {s: floored[s] + base[s] for s in stages}


def _uniform_sample(items: List[int], k: int) -> List[int]:
    """Pick `k` items from `items` with roughly uniform spacing (preserves order)."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    n = len(items)
    return [items[int(round(j * (n - 1) / (k - 1) if k > 1 else 0))] for j in range(k)]


QUALITY_REVIEW_PROMPT = """你是预测点质量审核员。事件「{event}」。
下面是一个候选预测点的目标节点正文（截止 cutoff={cutoff}）。请按 SKILL §Step 2.1
"目标节点质量审查"给该节点打分（1-5），并简短说明理由。

评分参考：
- 5: 离散、明确、可验证的事实推进（政策动作、官方表态、调查/裁决、谈判/协议、执行/暂停等）
- 4: 较明确的事实推进，但部分细节模糊
- 3: 可入选但边缘（轻量补充信息）
- 2: 偏向 explainer / 评论 / 综述 / 风险升温描写，缺少明确动作落点
- 1: portal / search 聚合 / FAQ / hashtag 长帖 / 评论残片 / 异题前溯

================ 目标节点正文 ================
{body}
================

只输出严格 JSON（不要 markdown 代码块）：
{{"score": <1-5>, "reason": "<一句话>"}}
"""


def llm_quality_score(event_name: str, cutoff: str, body: str, model: str) -> Tuple[int, str]:
    """Return (score 1-5, reason). On failure return (3, 'unknown')."""
    if not body.strip():
        return 1, "empty"
    try:
        raw = call_llm(
            [{"role": "user", "content": QUALITY_REVIEW_PROMPT.format(
                event=event_name, cutoff=cutoff, body=body[:2000])}],
            model=model,
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="none",
        )
    except Exception:
        return 3, "llm_error"
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    # Parse JSON and verify it's a dict — LLMs occasionally return a top-level
    # list/scalar even when prompted for an object. Fall through to regex
    # fallback in that case.
    data: Any = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        try:
            return int(data.get("score", 3)), str(data.get("reason", ""))[:200]
        except (ValueError, TypeError):
            pass
    m_s = re.search(r'"score"\s*:\s*(\d)', raw)
    return (int(m_s.group(1)) if m_s else 3), raw[:120]


def build_points(
    nodes: List[Dict[str, Any]],
    *,
    threshold_days: int,
    min_context: int,
    min_future: int,
    target_points: int,
    max_points: int,
    quality_scores: Optional[Dict[int, int]] = None,
    quality_min: int = 3,
) -> Dict[str, Any]:
    """Build the full prediction_points structure described in the SKILL.

    If `quality_scores` is provided (mapping node-index → 1-5), candidates with
    score < quality_min are demoted to `eligible_quality_rejected` instead of
    competing for the stage quota. Within each stage the remaining candidates
    are sorted by quality (high → low) before uniform-time sampling, so the
    final `selected` set prefers high-quality nodes.
    """
    # Strict monotonic date check (SKILL quality check #1)
    for k in range(len(nodes) - 1):
        if nodes[k]["_date_obj"] >= nodes[k + 1]["_date_obj"]:
            raise ValueError(
                f"Dates not strictly increasing: {nodes[k]['date']} >= {nodes[k + 1]['date']}"
            )

    # Annotate nodes with node_id
    nodes_out: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes):
        nodes_out.append({
            "node_id": f"N{i + 1:02d}",
            "date": n["date"],
            "date_raw": n["date_raw"],
            "stage": n["stage"],
            "line_start": n["line_start"],
            "line_end": n["line_end"],
        })

    # Pass 1: emit a transition entry for every i ≥ 1
    points: List[Dict[str, Any]] = []
    for i in range(1, len(nodes)):
        gap = (nodes[i]["_date_obj"] - nodes[i - 1]["_date_obj"]).days
        context_depth = i
        remaining = len(nodes) - 1 - i
        eligible_by_gap = (gap <= threshold_days)
        ineligible_context = context_depth < min_context
        ineligible_future = remaining < min_future

        if not eligible_by_gap:
            status = "ineligible_gap"
        elif ineligible_context or ineligible_future:
            status = "ineligible_window"
        else:
            status = "eligible"  # candidate; may be promoted to "selected" later

        points.append({
            "point_id": None,
            "status": status,
            "input_nodes": [f"N{j + 1:02d}" for j in range(i)],
            "target_node": f"N{i + 1:02d}",
            "target_date": nodes[i]["date"],
            "target_stage": nodes[i]["stage"],
            "gap_days": gap,
            "predictable": False,
            "context_depth": context_depth,
            "remaining_nodes": remaining,
        })

    # SKILL §Step 2.1: quality review (optional, opt-in via --quality-review-model).
    # When quality_scores is supplied, demote sub-threshold candidates before
    # quota allocation; within each stage, sort by quality desc so the uniform-
    # time sampler prefers high-quality nodes.
    n_quality_rejected = 0
    if quality_scores:
        for k, p in enumerate(points):
            if p["status"] != "eligible":
                continue
            # Map "transition index" k → "target node index in nodes list"
            target_node_idx = k + 1
            sc = quality_scores.get(target_node_idx, 3)
            p["quality_score"] = sc
            if sc < quality_min:
                p["status"] = "eligible_quality_rejected"
                n_quality_rejected += 1

    # Pass 2: per-stage quota over `eligible`
    eligible_indices = [k for k, p in enumerate(points) if p["status"] == "eligible"]
    by_stage: Dict[str, List[int]] = {}
    for k in eligible_indices:
        by_stage.setdefault(points[k]["target_stage"], []).append(k)

    desired = min(target_points, max_points, len(eligible_indices))
    quota = _stage_quota(by_stage, desired)

    selected_idx: List[int] = []
    for stage, k_list in by_stage.items():
        pick = quota.get(stage, 0)
        if quality_scores:
            # Sort by quality desc, then take uniform-time sample of top tier
            k_list = sorted(k_list, key=lambda kk: -int(points[kk].get("quality_score", 3)))
        selected_idx.extend(_uniform_sample(k_list, pick))
    selected_idx.sort()

    # Enforce hard cap (max_points). If quota slightly overshoots, trim by uniform sample.
    if len(selected_idx) > max_points:
        selected_idx = _uniform_sample(selected_idx, max_points)

    # Pass 3: mark statuses + assign Pxx
    selected_set = set(selected_idx)
    p_counter = 0
    for k, p in enumerate(points):
        if p["status"] == "eligible":
            if k in selected_set:
                p_counter += 1
                p["point_id"] = f"P{p_counter:02d}"
                p["status"] = "selected"
                p["predictable"] = True
            else:
                p["status"] = "eligible_not_selected"

    meta = {
        "total_nodes": len(nodes),
        "total_transitions": len(nodes) - 1,
        "threshold_days": threshold_days,
        "min_context": min_context,
        "min_future": min_future,
        "target_points": target_points,
        "max_points": max_points,
        "n_eligible": len(eligible_indices),
        "n_selected": p_counter,
        "n_ineligible_gap": sum(1 for p in points if p["status"] == "ineligible_gap"),
        "n_ineligible_window": sum(1 for p in points if p["status"] == "ineligible_window"),
        "n_eligible_not_selected": sum(1 for p in points if p["status"] == "eligible_not_selected"),
        "n_eligible_quality_rejected": n_quality_rejected,
        "stage_quota": quota,
        "quality_review_enabled": bool(quality_scores),
        "quality_min": quality_min if quality_scores else None,
    }
    return {"meta": meta, "nodes": nodes_out, "prediction_points": points}


# ===========================================================================
# Outputs
# ===========================================================================

def annotate_timeline(md_path: Path, points: List[Dict[str, Any]]) -> str:
    """Append [Pxx] to the date header line whose date matches the selected point."""
    by_date_to_id: Dict[str, str] = {
        p["target_date"]: p["point_id"]
        for p in points
        if p["point_id"]
    }
    out_lines: List[str] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = DATE_RE.match(line.strip())
        if m:
            date_raw = m.group(1)
            end_date_obj, _ = parse_date(date_raw)
            end_iso = end_date_obj.strftime("%Y-%m-%d")
            if end_iso in by_date_to_id:
                line = f"{line.rstrip()}  [{by_date_to_id[end_iso]}]"
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def render_summary(result: Dict[str, Any], source_name: str) -> str:
    meta = result["meta"]
    nodes = result["nodes"]
    points = result["prediction_points"]

    lines: List[str] = []
    lines.append("# 预测点筛选结果\n")
    lines.append(f"> 来源：{source_name}")
    lines.append(
        f"> 阈值：{meta['threshold_days']}天 | min_context={meta['min_context']} | "
        f"min_future={meta['min_future']} | target≈{meta['target_points']} | "
        f"cap={meta['max_points']}\n"
    )

    lines.append("## 统计\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总节点数 | {meta['total_nodes']} |")
    lines.append(f"| 总过渡数 | {meta['total_transitions']} |")
    lines.append(f"| 候选(eligible) | {meta['n_eligible']} |")
    lines.append(f"| 入选(selected) | {meta['n_selected']} |")
    lines.append(f"| 因配额未入选 | {meta['n_eligible_not_selected']} |")
    lines.append(f"| 间隔超阈值 | {meta['n_ineligible_gap']} |")
    lines.append(f"| context/未来不足 | {meta['n_ineligible_window']} |")

    lines.append("\n## 节点总览\n")
    lines.append("| ID | 日期 | 阶段 | 行号 |")
    lines.append("|----|------|------|------|")
    for n in nodes:
        lines.append(
            f"| {n['node_id']} | {n['date_raw']} | {n['stage']} | "
            f"{n['line_start']}-{n['line_end']} |"
        )

    lines.append(f"\n## 入选预测点（{meta['n_selected']}个）\n")
    lines.append("| 编号 | context | 后续GT | 目标日期 | 间隔 | 阶段 |")
    lines.append("|------|---------|--------|---------|------|------|")
    for p in points:
        if p["status"] == "selected":
            lines.append(
                f"| {p['point_id']} | {p['context_depth']} | "
                f"{p['remaining_nodes']} | {p['target_date']} | "
                f"{p['gap_days']}天 | {p['target_stage']} |"
            )

    lines.append("\n## 候选但未入选的点\n")
    lines.append("| 目标日期 | 间隔 | 阶段 | 状态 |")
    lines.append("|---------|------|------|------|")
    for p in points:
        if p["status"] != "selected":
            lines.append(
                f"| {p['target_date']} | {p['gap_days']}天 | "
                f"{p['target_stage']} | {p['status']} |"
            )

    lines.append("\n## 阶段配额\n")
    for stage, q in meta["stage_quota"].items():
        lines.append(f"- {stage}: {q}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step1-points: pick predictable target nodes")
    p.add_argument("timeline_md", help="refined timeline markdown")
    p.add_argument("output_dir")
    p.add_argument("--threshold-days", type=int, default=None)
    p.add_argument("--min-context", type=int, default=None)
    p.add_argument("--min-future", type=int, default=None)
    p.add_argument("--target-points", type=int, default=None,
                   help="Target count after stage-aware quota (default 25 from pipeline_config)")
    p.add_argument("--max-points", type=int, default=None,
                   help="Hard cap for selected points (default 30 from pipeline_config)")
    # SKILL §Step 2.1 quality review (default ON per SKILL hard requirement —
    # uses pipeline_config.default_llm_model unless overridden).
    p.add_argument("--quality-review-model", default=None,
                   help="LLM used for SKILL §Step 2.1 quality scoring. Defaults to "
                        "pipeline_config.default_llm_model. Set to '' or use --skip-quality-review "
                        "to disable.")
    p.add_argument("--skip-quality-review", action="store_true",
                   help="Emergency switch: disable SKILL §Step 2.1 quality review. "
                        "Not recommended — SKILL requires this as a hard constraint.")
    p.add_argument("--quality-min", type=int, default=3,
                   help="Quality score threshold; nodes below this are demoted to "
                        "eligible_quality_rejected. Default 3 (SKILL §Step 2.1).")
    p.add_argument("--event-name", default="<event_name>",
                   help="Event label used inside the quality-review prompt (generic placeholder ok).")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    s1 = cfg.get("predict_step1_points", {})
    threshold = args.threshold_days if args.threshold_days is not None else int(s1.get("interval_threshold_days", 14))
    min_ctx = args.min_context if args.min_context is not None else int(s1.get("min_context_nodes", 1))
    min_fut = args.min_future if args.min_future is not None else int(s1.get("min_future_nodes", 0))
    target = args.target_points if args.target_points is not None else int(s1.get("target_points", 25))
    cap = args.max_points if args.max_points is not None else int(s1.get("max_points", 30))

    in_path = Path(args.timeline_md)
    nodes = parse_timeline(in_path)
    if not nodes:
        raise SystemExit(f"No `### YYYY-MM-DD` headers found in {in_path}")

    # SKILL §Step 2.1 LLM quality review — default ON. Uses
    # --quality-review-model if set, else pipeline_config.default_llm_model.
    # --skip-quality-review disables it (emergency only).
    quality_scores: Optional[Dict[int, int]] = None
    qr_model = args.quality_review_model or cfg.get("default_llm_model")
    if not args.skip_quality_review and qr_model:
        args.quality_review_model = qr_model  # propagate resolved value
    elif args.skip_quality_review:
        args.quality_review_model = None
    if args.quality_review_model:
        lines_all = in_path.read_text(encoding="utf-8").splitlines()
        quality_scores = {}
        # Compute the cutoff for each candidate: the previous node's date.
        for i, node in enumerate(nodes):
            if i == 0:
                continue  # no prediction transition for the very first node
            body = "\n".join(
                lines_all[node["line_start"] - 1: (node["line_end"] or node["line_start"])]
            )
            cutoff = nodes[i - 1]["date"]
            score, _reason = llm_quality_score(args.event_name, cutoff, body, args.quality_review_model)
            quality_scores[i] = score
        print(f"[quality-review] scored {len(quality_scores)} target nodes "
              f"(model={args.quality_review_model})")

    result = build_points(
        nodes,
        threshold_days=threshold,
        min_context=min_ctx,
        min_future=min_fut,
        target_points=target,
        max_points=cap,
        quality_scores=quality_scores,
        quality_min=args.quality_min,
    )
    result["meta"]["source_file"] = in_path.name

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prediction_points.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "prediction_points_summary.md").write_text(
        render_summary(result, in_path.name), encoding="utf-8"
    )
    (out_dir / "timeline_with_points.md").write_text(
        annotate_timeline(in_path, result["prediction_points"]), encoding="utf-8"
    )

    meta = result["meta"]
    print(json.dumps({
        "total_nodes": meta["total_nodes"],
        "selected": meta["n_selected"],
        "eligible_not_selected": meta["n_eligible_not_selected"],
        "ineligible_gap": meta["n_ineligible_gap"],
        "ineligible_window": meta["n_ineligible_window"],
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
