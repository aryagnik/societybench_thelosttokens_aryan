# -*- coding: utf-8 -*-
"""export_llm_validated_v2.py — 社媒有效性验证 v2（kimi 一步语义版，通用可复用）

背景：v1（各事件的 export_llm_validated.py）用关键词规则判有效，方法参差且
对 SMCI 这类英文事件过严（实体∧主体∧动作 三重 AND → 174,779 砍到 57）。

v2 改为**一步语义判断**，对所有事件通用：
  - 逐帖让 LLM(kimi) 判「这条帖是不是在讲 {topic} 这件事」→ 留相关帖
  - 相关帖下面的非空评论一并保留（评论随父帖）
不依赖任何事件写死的关键词；事件只通过 --topic 传入 → 可复用。

用法：
  python3 export_llm_validated_v2.py --db media_crawler.db --topic "<一句话事件>" \
      --output data/valid_data_llm_verified_v2.csv [--model kimi-k2.5] [--batch-size 80] [--workers 8]
"""
from __future__ import annotations
import argparse, csv, json, os, re, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

# 复用项目统一 LLM 客户端（DMXAPI / 默认 kimi-k2.5）。
# 稳健定位含 common.py 的 code/eval 目录：支持脚本被复制到任意运行目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.path.abspath(os.path.join(_HERE, "..", "..")),  # 原位: code/eval/crawl/media_scripts -> code/eval
              os.environ.get("SB_EVAL_DIR", "")]:
    if _cand and os.path.exists(os.path.join(_cand, "common.py")):
        sys.path.insert(0, _cand); break
from common import call_llm  # noqa: E402

TZ = timezone(timedelta(hours=8))

# 帖子表 schema: platform -> (table, id, title, content, time, like, comment, share, ip, sk, url, name, uid)
POST_SCHEMA = [
    ("douyin",   "douyin_aweme",   "aweme_id",  "title", "desc",         "create_time", "liked_count", "comment_count", "share_count", "ip_location", "source_keyword", "aweme_url",   "nickname",      "user_id"),
    ("bilibili", "bilibili_video", "video_id",  "title", "desc",         "create_time", "liked_count", "video_comment", None,          None,          "source_keyword", "video_url",   "nickname",      "user_id"),
    ("weibo",    "weibo_note",     "note_id",   None,    "content",      "create_time", "liked_count", "comments_count","shared_count","ip_location", "source_keyword", "note_url",    "nickname",      "user_id"),
    ("zhihu",    "zhihu_content",  "content_id","title", "content_text", "created_time","voteup_count","comment_count", None,          None,          "source_keyword", "content_url", "user_nickname", "user_id"),
    ("tieba",    "tieba_note",     "note_id",   "title", "desc",         "publish_time",None,          "total_replay_num",None,        "ip_location", "source_keyword", "note_url",    "user_nickname", "user_id"),
    ("xhs",      "xhs_note",       "note_id",   "title", "desc",         "time",        "liked_count", "comment_count", "share_count", "ip_location", "source_keyword", "note_url",    "nickname",      "user_id"),
    ("kuaishou", "kuaishou_video", "video_id",  "title", "desc",         "create_time", "liked_count", None,            None,          None,          "source_keyword", "video_url",   "nickname",      "user_id"),
]
# 评论表 schema: platform -> (table, id, parent, pcid, content, time, like, sub, ip, name, uid)
COMMENT_SCHEMA = [
    ("douyin",   "douyin_aweme_comment",   "comment_id", "aweme_id",   "parent_comment_id", "content", "create_time", "like_count", "sub_comment_count", "ip_location", "nickname",      "user_id"),
    ("bilibili", "bilibili_video_comment", "comment_id", "video_id",   "parent_comment_id", "content", "create_time", "like_count", "sub_comment_count", None,          "nickname",      "user_id"),
    ("weibo",    "weibo_note_comment",     "comment_id", "note_id",    "parent_comment_id", "content", "create_time", "like_count", "sub_comment_count", "ip_location", "nickname",      "user_id"),
    ("zhihu",    "zhihu_comment",          "comment_id", "content_id", "parent_comment_id", "content", "publish_time","like_count", "sub_comment_count", "ip_location", "user_nickname", "user_id"),
    ("tieba",    "tieba_comment",          "comment_id", "note_id",    "parent_comment_id", "content", "publish_time",None,         "sub_comment_count", "ip_location", "user_nickname", "comment_id"),
    ("xhs",      "xhs_note_comment",       "comment_id", "note_id",    "parent_comment_id", "content", "create_time", "like_count", "sub_comment_count", "ip_location", "nickname",      "user_id"),
    ("kuaishou", "kuaishou_video_comment", "comment_id", "video_id",   "parent_comment_id", "content", "create_time", "like_count", "sub_comment_count", None,          "nickname",      "user_id"),
]
FIELDS = ['platform','type','id','parent_id','parent_comment_id','user_id','user_name','title','content',
          'create_time','create_timestamp','liked_count','comment_count','share_count','ip_location','source_keyword','url']

PROMPT = """你是中文舆情数据标注员。判断每条社媒帖子是否在讨论「目标事件」（与其直接相关）。
目标事件：
{topic}

判定标准：
- 帖子在讲该事件本身、涉事公司/机构/当事人、相关进展，或针对该事件的讨论、吐槽、情绪、追问 → related=true
- 同名或谐音噪声、与该事件无关的其它话题、纯泛泛内容（与本事件无具体关联）→ related=false
- 证据不足时判 false
只允许依据给出的 text 判断。只输出 JSON 数组，每项 {{"idx": <int>, "related": <true|false>}}，不要任何额外文字。

帖子列表：
{rows}
"""

def ts_to_dt(ts):
    try:
        if isinstance(ts, str) and '-' in ts: return ts
        return datetime.fromtimestamp(int(float(ts)), tz=TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts or "")

def table_exists(c, t):
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

def col(r, name, keys):
    return r[name] if name and name in keys else ''

def judge_batch(topic, batch, model):
    """batch: list of (global_idx, text). 返回 {global_idx: bool}"""
    rows = json.dumps([{"idx": gi, "text": (t or "")[:160]} for gi, t in batch], ensure_ascii=False)
    try:
        raw = call_llm([{"role": "system", "content": "你是严格的二分类标注助手，只输出 JSON。"},
                        {"role": "user", "content": PROMPT.format(topic=topic, rows=rows)}],
                       model=model, temperature=0.0, max_tokens=4000, reasoning_effort="none")
    except Exception as e:
        print(f"[v2] batch LLM failed (idx {batch[0][0]}..): {e}", file=sys.stderr)
        return {}
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"): raw = raw[4:].strip()
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m: return {}
        try: items = json.loads(m.group(0))
        except Exception: return {}
    out = {}
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("idx"), int):
            out[it["idx"]] = bool(it.get("related", False))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="media_crawler.db")
    p.add_argument("--topic", required=True, help="一句话目标事件（可复用：换事件只换这个）")
    p.add_argument("--output", default="data/valid_data_llm_verified_v2.csv")
    p.add_argument("--model", default=None, help="判定所用 LLM；默认走 common 解析的默认 LLM（不写死具体模型）")
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    conn = sqlite3.connect(a.db); conn.row_factory = sqlite3.Row; c = conn.cursor()

    # ── 收集全部帖子 ──
    posts = []  # (global_idx, platform, record_dict, text)
    for plat, tbl, idc, tc, cc, timc, likc, comc, shac, ipc, skc, urlc, namc, uidc in POST_SCHEMA:
        if not table_exists(c, tbl): continue
        for r in c.execute(f"SELECT * FROM {tbl}").fetchall():
            keys = set(r.keys())
            title = col(r, tc, keys); content = col(r, cc, keys); sk = col(r, skc, keys)
            pt = col(r, timc, keys)
            text = ((title or "") + " " + (content or "")).strip()
            rec = {'platform': plat, 'type': 'post', 'id': r[idc], 'parent_id': '', 'parent_comment_id': '',
                   'user_id': col(r, uidc, keys) or r[idc], 'user_name': col(r, namc, keys),
                   'title': title or '', 'content': content or '',
                   'create_time': ts_to_dt(pt), 'create_timestamp': pt,
                   'liked_count': str(col(r, likc, keys) or 0), 'comment_count': str(col(r, comc, keys) or 0),
                   'share_count': str(col(r, shac, keys) or 0), 'ip_location': col(r, ipc, keys) or '',
                   'source_keyword': sk or '', 'url': col(r, urlc, keys) or ''}
            posts.append((len(posts), plat, rec, text))
    total_posts = len(posts)
    print(f"[v2] 总帖子 {total_posts}，开始 kimi 逐帖判定（batch={a.batch_size}, workers={a.workers}）...", flush=True)

    # ── 并行 kimi 判定 ──
    batches = [posts[i:i+a.batch_size] for i in range(0, total_posts, a.batch_size)]
    verdicts = {}
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge_batch, a.topic, [(gi, txt) for gi, _, _, txt in b], a.model): bi
                for bi, b in enumerate(batches)}
        for f in as_completed(futs):
            verdicts.update(f.result()); done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"[v2] 批次 {done}/{len(batches)} 完成", flush=True)

    valid_post_ids = set(); records = []
    for gi, plat, rec, _ in posts:
        if verdicts.get(gi, False):
            valid_post_ids.add((plat, rec['id'])); records.append(rec)
    post_kept = len(records)
    print(f"[v2] kimi 判定相关帖：{post_kept}/{total_posts}", flush=True)

    # ── 评论：父帖有效 + 内容非空 ──
    total_comments = 0
    for plat, tbl, idc, parc, pcidc, cc, timc, likc, subc, ipc, namc, uidc in COMMENT_SCHEMA:
        if not table_exists(c, tbl): continue
        for r in c.execute(f"SELECT * FROM {tbl}").fetchall():
            total_comments += 1
            keys = set(r.keys())
            parent = col(r, parc, keys)
            content = col(r, cc, keys)
            if (plat, parent) in valid_post_ids and content:
                pt = col(r, timc, keys)
                records.append({'platform': plat, 'type': 'comment', 'id': r[idc], 'parent_id': parent,
                    'parent_comment_id': col(r, pcidc, keys) or '', 'user_id': col(r, uidc, keys) or r[idc],
                    'user_name': col(r, namc, keys), 'title': '', 'content': content or '',
                    'create_time': ts_to_dt(pt), 'create_timestamp': pt,
                    'liked_count': str(col(r, likc, keys) or 0), 'comment_count': str(col(r, subc, keys) or 0),
                    'share_count': '0', 'ip_location': col(r, ipc, keys) or '', 'source_keyword': '', 'url': ''})
    conn.close()
    comment_kept = len(records) - post_kept

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(records)

    from collections import Counter
    by = Counter((r['platform'], r['type']) for r in records)
    print("="*56)
    print(f"[v2 kimi] 帖子 {total_posts} → 有效 {post_kept}")
    print(f"[v2 kimi] 评论 {total_comments} → 有效 {comment_kept}（父帖有效且非空）")
    print(f"[v2 kimi] 合计有效 = {len(records)} 条  → {a.output}")
    for (pl, ty), n in sorted(by.items()): print(f"    {pl} {ty}: {n}")
    print("="*56)

if __name__ == "__main__":
    main()
