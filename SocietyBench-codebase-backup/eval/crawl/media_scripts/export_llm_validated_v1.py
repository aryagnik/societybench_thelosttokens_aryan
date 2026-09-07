# -*- coding: utf-8 -*-
"""export_llm_validated_v1.py — 社媒有效性验证 v1（LLM 生成分层词表 + 规则匹配，全事件通用模板）

由 LLM(kimi) 从一句话事件**自动生成分层词表**，再用规则秒级过滤：
  判定 = 不命中 noise 且 ( 命中任一 anchor(强专名,单独即可)  或  (entity ∧ event 共现) )
- anchor_terms：本事件独有、单独命中即判相关（做空机构名/当事人全名/"实体+事件"固定搭配）。
- entity_terms ∧ event_terms：实体名与泛事件词**共现**才算——避免"超微/AI服务器/退市"等宽词单独误召回。
- noise_patterns：误命中源（支持负向前瞻正则）。

与 v2 的区别：v1 只调 1 次 LLM（生成词表）+ 规则飞快、宽召回基线；v2 逐帖调 kimi 语义判、精度更高（media 标准）。
两者并存：v1 做快速基线/对照，v2 做标准产出。
（注意：SMCI 旧版 export_llm_validated.smci.py 是 实体∧主体∧动作 三重 AND，过严、异类，仅留反例。）

事件只通过 **--topic** 传入（LLM 生成词表）→ 全事件可复用、零写死关键词。
也可 --keywords-config 手动给分层词表 {anchor_terms,entity_terms,event_terms,noise_patterns}（兼容旧扁平 {"keywords":[...]}，视作 anchor）。

用法：
  PYTHONPATH=<repo>/code/eval python3 export_llm_validated_v1.py \
      --db media_crawler.db --topic "<一句话事件>" --output data/valid_data_llm_verified_v1.csv
"""
from __future__ import annotations
import argparse, csv, json, os, re, sqlite3, sys
from collections import Counter
from datetime import datetime, timezone, timedelta

# 复用项目统一 LLM 客户端（DMXAPI / kimi-k2.5），用于「由 LLM 生成关键词」。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.path.abspath(os.path.join(_HERE, "..", "..")),
              os.environ.get("SB_EVAL_DIR", "")]:
    if _cand and os.path.exists(os.path.join(_cand, "common.py")):
        sys.path.insert(0, _cand); break
try:
    from common import call_llm
except Exception:
    call_llm = None  # 仅 --keywords-config / --keywords 手动模式时不需要

TZ = timezone(timedelta(hours=8))

KW_GEN_PROMPT = """你是社媒舆情检索专家。给定目标事件，生成「分层检索词表」，供规则召回用。
核心目标：**命中即应高概率确属本事件**——既要召回，又要避免把"只提到该公司/泛泛话题"的无关帖也召回。

目标事件：
{topic}

请输出 4 类词，分层是关键：

1. anchor_terms（强专名，单独命中即判相关）：本事件**独有**、几乎不可能出现在无关语境的专名——做空机构名、当事人全名、事件独有代号/梗、"实体+事件"的固定搭配。例：做空机构名、关键人物全名。
2. entity_terms（实体名，需与 event 共现才算）：涉事公司/机构的**具体且无歧义**全名与官方简称。
   - 必须无歧义：**禁止**会在无关语境高频出现的短词/常用词（如中文 2-3 字常用词，会误命中其它含义）。
   - **禁止**公司的业务/产品泛称（如"某类服务器/某类芯片/某行业"——这是公司业务，不是本事件）。
3. event_terms（事件词，需与 entity 共现才算）：本事件特有的动作/进展/术语（如 具体的处分/诉讼/财报/监管动作 等）。这些词较泛，单独不足为凭，必须与某个 entity 同现。
4. noise_patterns（噪声正则，命中即排除）：会误命中的同名不同义、相近无关领域、纯泛词。

判定规则（你据此挑词）：valid = 不命中noise 且 ( 命中任一 anchor  或  (命中任一 entity 且 命中任一 event) )。
所以：anchor 要足够"独一无二"；entity 要"准且专"；泛词放 event（靠共现兜底），**不要把泛词或业务词放进 anchor/entity**。

只输出 JSON：{{"anchor_terms":[...],"entity_terms":[...],"event_terms":[...],"noise_patterns":[...]}}，不要任何额外文字。
"""

def generate_keywords(topic: str, model: str) -> dict:
    """由 LLM(kimi) 从一句话事件生成分层词表 {anchor_terms, entity_terms, event_terms, noise_patterns}。"""
    if call_llm is None:
        sys.exit("[v1] 需要 LLM 生成关键词，但未能加载 common.call_llm（设置 SB_EVAL_DIR 指向 code/eval）")
    raw = call_llm([{"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": KW_GEN_PROMPT.format(topic=topic)}],
                   model=model, temperature=0.0, max_tokens=2000)
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"): raw = raw[4:].strip()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        cfg = json.loads(m.group(0)) if m else {}
    for k in ("anchor_terms", "entity_terms", "event_terms", "noise_patterns"):
        cfg.setdefault(k, [])
    return cfg

# 帖子表 schema（与 v2 一致）: platform -> (table, id, title, content, time, like, comment, share, ip, sk, url, name, uid)
POST_SCHEMA = [
    ("douyin",   "douyin_aweme",   "aweme_id",  "title", "desc",         "create_time", "liked_count", "comment_count", "share_count", "ip_location", "source_keyword", "aweme_url",   "nickname",      "user_id"),
    ("bilibili", "bilibili_video", "video_id",  "title", "desc",         "create_time", "liked_count", "video_comment", None,          None,          "source_keyword", "video_url",   "nickname",      "user_id"),
    ("weibo",    "weibo_note",     "note_id",   None,    "content",      "create_time", "liked_count", "comments_count","shared_count","ip_location", "source_keyword", "note_url",    "nickname",      "user_id"),
    ("zhihu",    "zhihu_content",  "content_id","title", "content_text", "created_time","voteup_count","comment_count", None,          None,          "source_keyword", "content_url", "user_nickname", "user_id"),
    ("tieba",    "tieba_note",     "note_id",   "title", "desc",         "publish_time",None,          "total_replay_num",None,        "ip_location", "source_keyword", "note_url",    "user_nickname", "user_id"),
    ("xhs",      "xhs_note",       "note_id",   "title", "desc",         "time",        "liked_count", "comment_count", "share_count", "ip_location", "source_keyword", "note_url",    "nickname",      "user_id"),
    ("kuaishou", "kuaishou_video", "video_id",  "title", "desc",         "create_time", "liked_count", None,            None,          None,          "source_keyword", "video_url",   "nickname",      "user_id"),
]
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

class Rule:
    """分层判定：valid = 不命中 noise 且 ( 命中任一 anchor  或  (命中 entity 且 命中 event) )。
    - anchor_terms：强专名，单独命中即有效（做空机构名/当事人全名/事件独有代号）。
    - entity_terms ∧ event_terms：实体与事件词**共现**才算（泛事件词靠共现兜底，避免单独误召回）。
    - 兼容旧配置：扁平 keywords 视作 anchor（单独命中即有效）。"""
    def __init__(self, cfg):
        self.noise   = [re.compile(p, re.I) for p in cfg.get("noise_patterns", [])]
        self.anchor  = [k.lower() for k in cfg.get("anchor_terms", []) if k]
        self.entity  = [k.lower() for k in cfg.get("entity_terms", []) if k]
        self.event   = [k.lower() for k in cfg.get("event_terms", []) if k]
        self.anchor += [k.lower() for k in cfg.get("keywords", []) if k]   # 向后兼容旧扁平词表
        self.require_by_source = cfg.get("require_by_source", {}) or {}
        if not (self.anchor or (self.entity and self.event)):
            sys.exit("[v1] 词表为空：需要 anchor_terms 或 (entity_terms + event_terms)")

    def is_valid(self, text, sk):
        if not text:
            return False
        low = text.lower()
        for pat in self.noise:                                   # 噪声 → 删
            if pat.search(low):
                return False
        if sk in self.require_by_source and not any(k in text for k in self.require_by_source[sk]):
            return False
        if any(a in low for a in self.anchor):                   # ① 强专名单独命中
            return True
        if self.entity and self.event \
           and any(e in low for e in self.entity) \
           and any(v in low for v in self.event):                # ② 实体 ∧ 事件 共现
            return True
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="media_crawler.db")
    p.add_argument("--topic", default=None, help="一句话事件 → 由 LLM 自动生成关键词（推荐默认用法）")
    p.add_argument("--model", default=None, help="生成关键词所用 LLM；默认走 common 解析的默认 LLM（不写死具体模型）")
    p.add_argument("--keywords-config", default=None, help="手动关键词配置 JSON（覆盖 LLM 生成）")
    p.add_argument("--keywords", default=None, help="手动简单模式：逗号分隔关键词")
    p.add_argument("--output", default="data/valid_data_llm_verified_v1.csv")
    p.add_argument("--save-keywords", default=None, help="生成/使用的关键词另存 JSON（默认放 output 同目录）")
    a = p.parse_args()

    if a.keywords_config:                       # 优先级1：手动配置
        cfg = json.loads(open(a.keywords_config, encoding="utf-8").read())
        print(f"[v1] 用手动关键词配置: {a.keywords_config}")
    elif a.keywords:                            # 优先级2：手动简单词表
        cfg = {"keywords": [k.strip() for k in a.keywords.split(",") if k.strip()]}
    elif a.topic:                               # 优先级3（推荐）：LLM 从 topic 生成
        print(f"[v1] 由 LLM({a.model}) 从 topic 自动生成关键词 ...", flush=True)
        cfg = generate_keywords(a.topic, a.model)
        kwpath = a.save_keywords or os.path.join(os.path.dirname(a.output) or ".", "generated_keywords.json")
        os.makedirs(os.path.dirname(kwpath) or ".", exist_ok=True)
        json.dump({"topic": a.topic, **cfg}, open(kwpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[v1] LLM 生成分层词表 → {kwpath}")
        print(f"[v1]   anchor(单独命中) {len(cfg.get('anchor_terms',[]))}: {cfg.get('anchor_terms')}")
        print(f"[v1]   entity(需共现)   {len(cfg.get('entity_terms',[]))}: {cfg.get('entity_terms')}")
        print(f"[v1]   event (需共现)   {len(cfg.get('event_terms',[]))}: {cfg.get('event_terms')}")
        print(f"[v1]   noise            {len(cfg.get('noise_patterns',[]))}: {cfg.get('noise_patterns')}")
    else:
        sys.exit("[v1] 需要 --topic（LLM 生成关键词，推荐）/ --keywords-config / --keywords 之一")
    rule = Rule(cfg)

    conn = sqlite3.connect(a.db); conn.row_factory = sqlite3.Row; c = conn.cursor()
    total_posts = 0; valid_post_ids = set(); records = []; per_plat = Counter(); per_plat_valid = Counter()
    for plat, tbl, idc, tc, cc, timc, likc, comc, shac, ipc, skc, urlc, namc, uidc in POST_SCHEMA:
        if not table_exists(c, tbl): continue
        for r in c.execute(f"SELECT * FROM {tbl}").fetchall():
            total_posts += 1; per_plat[plat] += 1
            keys = set(r.keys())
            title = col(r, tc, keys); content = col(r, cc, keys); sk = col(r, skc, keys)
            text = ((title or "") + " " + (content or "")).strip()
            if rule.is_valid(text, sk):
                pt = col(r, timc, keys); per_plat_valid[plat] += 1
                valid_post_ids.add((plat, r[idc]))
                records.append({'platform': plat, 'type': 'post', 'id': r[idc], 'parent_id': '', 'parent_comment_id': '',
                    'user_id': col(r, uidc, keys) or r[idc], 'user_name': col(r, namc, keys),
                    'title': title or '', 'content': content or '',
                    'create_time': ts_to_dt(pt), 'create_timestamp': pt,
                    'liked_count': str(col(r, likc, keys) or 0), 'comment_count': str(col(r, comc, keys) or 0),
                    'share_count': str(col(r, shac, keys) or 0), 'ip_location': col(r, ipc, keys) or '',
                    'source_keyword': sk or '', 'url': col(r, urlc, keys) or ''})
    post_kept = len(records)

    total_comments = 0
    for plat, tbl, idc, parc, pcidc, cc, timc, likc, subc, ipc, namc, uidc in COMMENT_SCHEMA:
        if not table_exists(c, tbl): continue
        for r in c.execute(f"SELECT * FROM {tbl}").fetchall():
            total_comments += 1
            keys = set(r.keys()); parent = col(r, parc, keys); content = col(r, cc, keys)
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

    print("="*56)
    print(f"[v1 关键词OR] 帖子 {total_posts} → 有效 {post_kept}")
    print(f"[v1 关键词OR] 评论 {total_comments} → 有效 {comment_kept}（父帖有效且非空）")
    print(f"[v1 关键词OR] 合计 = {len(records)} 条  → {a.output}")
    for pl in per_plat:
        print(f"    {pl}: {per_plat[pl]} → {per_plat_valid[pl]}")
    print("="*56)

if __name__ == "__main__":
    main()
