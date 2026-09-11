#!/usr/bin/env python3
"""Phase 2: standalone answering (strict two-phase design, 2026-07-05):
Read the opinion snapshots saved by phase 1, answer and score each point independently, and
write the result. **No simulation, no rollout** — pure offline batch processing; like m12 it
can be parallelized, rerun, and the answering sample size tuned (MIROFISH_READ_MAX_ITEMS/CHARS).

Usage: python miro_phase2_answer.py --workspace WS --base doubao-... --run-id RID [--max-parallel 4]
"""
import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def build_rich_materials(snapshot_texts, db_path):
    """Reconstruct each opinion item from the existing social.db into a rich record
    (author/type/as-of-cutoff likes-dislikes/reply relations) plus a netizen roster.
    Strictly as-of-cutoff: likes/dislikes are filtered by the like table's created_at <= this
    point's cutoff moment, no future leakage.
    Cutoff moment = the max created_at across posts/comments in this point's snapshot
    (i.e. the time of the last opinion item appearing at this point).
    If the DB is missing or unmatched, fall back to plain text as-is (no error). Returns (roster_text, [rich_records])."""
    if not db_path or not os.path.exists(db_path):
        return None, list(snapshot_texts)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        users = {u: (n, (b or "").strip()) for u, n, b in
                 cur.execute("SELECT user_id,name,bio FROM user")}
        posts = {c: (pid, uid, ts) for pid, uid, c, ts in
                 cur.execute("SELECT post_id,user_id,content,created_at FROM post")}
        comments = {c: (cid, ppid, uid, ts) for cid, ppid, uid, c, ts in
                    cur.execute("SELECT comment_id,post_id,user_id,content,created_at FROM comment")}
        # This point's cutoff moment (as-of-cutoff filter baseline)
        ts_pool = [posts[t][2] for t in snapshot_texts if t in posts]
        ts_pool += [comments[t][3] for t in snapshot_texts if t in comments]
        moment = max(ts_pool) if ts_pool else "9999"

        def _cnt(table, col, key):
            return cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col}=? AND created_at<=?",  # noqa: S608
                               (key, moment)).fetchone()[0]

        # Language-adaptive labels: English answering (SB_AGENT_LANG=en) uses English labels, otherwise Chinese. Does not affect already-finished Chinese runs.
        _en = os.environ.get("SB_AGENT_LANG", "zh").strip().lower() == "en"
        _sep = ": " if _en else "："
        _post, _cmt = ("post", "comment") if _en else ("帖", "评论")
        roster = "\n".join(f"- {n}{_sep}{b}" for _u, (n, b) in sorted(users.items())) or None
        rich = []
        for t in snapshot_texts:
            if t in posts:
                pid, uid, _ts = posts[t]
                name = users.get(uid, ("?", ""))[0]
                rich.append(f"[{name}·{_post}·👍{_cnt('like', 'post_id', pid)} "
                            f"👎{_cnt('dislike', 'post_id', pid)}] {t}")
            elif t in comments:
                cid, ppid, uid, _ts = comments[t]
                name = users.get(uid, ("?", ""))[0]
                prow = cur.execute("SELECT user_id FROM post WHERE post_id=?", (ppid,)).fetchone()
                if prow:
                    _pn = users.get(prow[0], ("?", ""))[0]
                    pauth = (f" ↳replying to {_pn}'s post" if _en else f" ↳回应{_pn}的帖")
                else:
                    pauth = ""
                rich.append(f"[{name}·{_cmt}·👍{_cnt('comment_like', 'comment_id', cid)} "
                            f"👎{_cnt('comment_dislike', 'comment_id', cid)}{pauth}] {t}")
            else:
                rich.append(t)  # unmatched (rare): keep as-is, don't lose information
        return roster, rich
    finally:
        con.close()

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):  # agents/ + parent eval/ (predict_step3B/F live there)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import predict_step3B_brier as B          # noqa: E402
import predict_step3F_time as F           # noqa: E402
import mirofish_sim                        # noqa: E402  reader/call_llm (answering uses sync urllib, no simulation)
from run_agents import brier_point, time_point  # noqa: E402  reuse scoring, same accounting as bare models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--base", default="doubao-seed-2-0-pro-260215")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--thinking-budget", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=999999)
    ap.add_argument("--runs-per-point", type=int, default=1)
    ap.add_argument("--max-parallel", type=int, default=4)  # cross-point parallelism (like m12)
    ap.add_argument("--axes", default="brier,time")
    args = ap.parse_args()

    ws = Path(args.workspace)
    # English workspace switches to en prompts (aligned with main experiment / bare models)
    if "英文" in ws.parts:
        B.LANG = "en"
        B.ANSWER_PROMPT, B.RETRY_PROMPT = B.ANSWER_PROMPT_EN, B.RETRY_PROMPT_EN
        F.PROMPT_TEMPLATE, F.EXAM_PROMPT_TEMPLATE = F.PROMPT_TEMPLATE_EN, F.EXAM_PROMPT_TEMPLATE_EN
        os.environ["SB_AGENT_LANG"] = "en"

    qb_dir, ctx_dir = ws / "questionbank", ws / "contexts"
    snap_dir = Path(os.environ.get("MIROFISH_SNAPSHOT_DIR") or str(ws / "mirofish_snapshots"))
    slug = f"mirofish__{args.base}"
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    cal_cfg = {}
    carry_ctx = os.environ.get("SB_AGENT_SESSION_CARRY_CONTEXT", "1") != "0"

    qb_files = sorted(qb_dir.glob("P*_questionbank.json"))
    tasks = []
    for qbf in qb_files:
        pid = qbf.stem.replace("_questionbank", "")
        try:
            cutoff = json.loads(qbf.read_text(encoding="utf-8")).get("cutoff_date") or ""
        except Exception:
            cutoff = ""
        safe = cutoff.replace("/", "-").replace(" ", "_")
        snapf = snap_dir / f"{safe}.json"
        if not snapf.exists():
            print(f"[phase2] 跳过 {pid}: 无快照 {snapf.name}(阶段1未推到此点)", flush=True)
            continue
        tasks.append({"pid": pid, "cutoff": cutoff, "qbf": qbf,
                      "ctxf": ctx_dir / f"{pid}_context.md", "snapf": snapf})

    print(f"[phase2] {len(tasks)} 个点有快照,开始并行答题(并行度{args.max_parallel})", flush=True)

    db_path = str(ws / "results" / f"run_{args.run_id}" / "mirofish_process" / "social.db")

    def answer_point(t):
        pid, qbf, ctxf, snapf = t["pid"], t["qbf"], t["ctxf"], t["snapf"]
        transcript = json.loads(snapf.read_text(encoding="utf-8")).get("transcript", [])
        # Rich materials: author/type/as-of-cutoff likes-dislikes/reply relations + netizen roster (as-of-cutoff, no future leakage)
        roster, rich = build_rich_materials(transcript, db_path)
        # answer_fn: read this point's rich opinions -> reader assembles signals+roster -> call_llm (single path, no racing; no cap, controlled by env)
        def _afn_brier(prompt):
            return mirofish_sim.reader(prompt, rich, args.base, args.max_tokens,
                                       args.thinking_budget, roster=roster)
        _afn_time = _afn_brier
        done = []
        if "brier" in axes:
            fp = ws / "results" / f"run_{args.run_id}" / "brier" / slug / f"{pid}_brier.json"
            r = brier_point(pid, qbf, ctxf, "mirofish", args.base, cal_cfg,
                            args.max_tokens, args.thinking_budget, args.batch_size,
                            None, max_parallel=1, existing_bad=None,
                            answer_fn=_afn_brier, embed_context=carry_ctx)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            done.append(f"brier={r.get('score_100')}")
        if "time" in axes:
            fp = ws / "results" / f"run_{args.run_id}" / "time" / slug / f"{pid}_time.json"
            try:
                r = time_point(pid, qbf, ctxf, "mirofish", args.base,
                               args.max_tokens, args.thinking_budget,
                               args.runs_per_point, None,
                               answer_fn=_afn_time, embed_context=carry_ctx, parallel_runs=False)
            except Exception as e:  # noqa: BLE001
                r = {"point_id": pid, "model": slug, "cutoff_date": t["cutoff"],
                     "events_total": 0, "score_100": None, "events": [], "per_run": [],
                     "valid_for_scoring": False, "invalid_reason": f"error: {e}"}
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            done.append(f"time={r.get('score_100')}")
        return pid, done

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        futs = {ex.submit(answer_point, t): t["pid"] for t in tasks}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                _pid, done = fut.result()
                ok += 1
                print(f"[phase2] ✓ {_pid}: {' '.join(done)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[phase2] ✗ {pid} 答题失败: {e}", flush=True)
    print(f"[phase2] ALL_DONE {ok}/{len(tasks)} 点已答", flush=True)


if __name__ == "__main__":
    main()
