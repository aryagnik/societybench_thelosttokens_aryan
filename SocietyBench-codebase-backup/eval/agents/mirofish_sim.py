#!/usr/bin/env python3
"""MiroFish social simulator (runs inside the OASIS / py3.11 venv, invoked as a subprocess by fw_mirofish_oasis.py).

Reads job JSON: {base_model, context, prompt, kind, n_agents, n_steps, db_path}
Flow (based on open-source OASIS):
  1. From context, use an LLM to generate N virtual netizen personas with stances;
  2. OASIS Reddit simulation: account 0 posts the event bulletin -> all LLM agents interact freely for K steps (opinion evolves forward);
  3. Read the discussion out of the simulation database (crowd-belief signal);
  4. reader: given "crowd discussion + original questions", output answer lines in the standard format.
The answer is printed to stdout wrapped in <<<ANSWER>>> ... <<<END>>>.

Design intent: OASIS actually runs the social simulation (faithful to the paper's "built on
OASIS"); the crowd discussion emerging from the simulation serves as extra evidence for
prediction; the reader keeps the bare-model prompt so the output format stays compatible with scoring.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
import urllib.request

# Suppress the OASIS/camel INFO flood (each step prints every post each agent observes).
import logging as _logging  # noqa: E402
for _n in ("social.agent", "social.twitter", "oasis.env", "oasis", "camel", "httpx", "root"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)

DMX_KEY = os.environ.get("DMXAPI_KEY", "").strip()
DMX_BASE = os.environ.get("DMXAPI_BASE_URL", "https://www.dmxapi.com/v1").rstrip("/")
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "32000"))
DEFAULT_THINKING_BUDGET = int(os.environ.get("AGENT_THINKING_BUDGET", "24000"))

# Decided 2026-07-03: netizen calls (in-simulation actions) disable thinking — the official
# doubao param thinking=disabled. Measured on the same ~80-char post: 23.7s -> 3.6s (each post
# previously wrote ~1.2k chars of hidden reasoning; the standard OASIS/MiroFish form is
# lightweight netizen chat generation anyway — thinking was a side effect of DMXAPI defaulting
# to auto after switching to the seed-2.0-pro base).
# Implementation: camel's ChatGPTConfig doesn't accept a thinking field, so inside this
# simulation subprocess we monkey-patch the openai client (passed through via the official
# extra_body). Only affects netizen action calls made through the openai client;
# make_personas and reader (readout) use this file's call_llm (urllib), unaffected, still full thinking.
# Set MIROFISH_NETIZEN_NOTHINK=0 to restore the old behavior.
if os.environ.get("MIROFISH_NETIZEN_NOTHINK", "1") != "0":
    try:
        from openai.resources.chat import completions as _oai_comp

        _NTZ_TIMEOUT = float(os.environ.get("MIROFISH_NETIZEN_TIMEOUT", "120"))
        _NTZ_LANES = max(1, int(os.environ.get("MIROFISH_NETIZEN_HEDGE_LANES", "3")))
        _NTZ_STAGGER = float(os.environ.get("MIROFISH_NETIZEN_HEDGE_STAGGER", "30"))

        def _prep_kw(kw):
            # Disable thinking (decided 2026-07-03) + hang-detection cut 600s -> 120s (approved; a hung call yields nothing anyway, semantically equivalent)
            eb = dict(kw.get("extra_body") or {})
            eb.setdefault("thinking", {"type": "disabled"})
            kw["extra_body"] = eb
            kw.setdefault("timeout", _NTZ_TIMEOUT)
            return kw

        def _nothink_wrap(orig):
            def _w(self, *a, **kw):
                return orig(self, *a, **_prep_kw(kw))
            return _w

        def _nothink_hedge_async(orig):
            # Hedged netizen calls (2026-07-03: "high-latency calls during rollout also get hedging"):
            # normal second-scale calls cost nothing extra; if no return after STAGGER seconds,
            # fire one more lane (up to LANES), first result wins, later ones cancelled.
            async def _w(self, *a, **kw):
                kw = _prep_kw(kw)
                import asyncio as _aio
                pending = {_aio.ensure_future(orig(self, *a, **kw))}
                fired, last = 1, None
                try:
                    while pending or fired < _NTZ_LANES:
                        if not pending:
                            pending = {_aio.ensure_future(orig(self, *a, **kw))}
                            fired += 1
                            continue
                        tmo = _NTZ_STAGGER if fired < _NTZ_LANES else None
                        done, pending = await _aio.wait(pending, timeout=tmo,
                                                        return_when=_aio.FIRST_COMPLETED)
                        for d in done:
                            try:
                                return d.result()
                            except BaseException as e:  # noqa: BLE001
                                last = e
                        if not done and fired < _NTZ_LANES:
                            sys.stderr.write("[ntz-hedge] slow netizen call, backup lane fired\n")
                            pending = set(pending)
                            pending.add(_aio.ensure_future(orig(self, *a, **kw)))
                            fired += 1
                    raise last if last else RuntimeError("ntz hedge: no result")
                finally:
                    for t in pending:
                        t.cancel()
            return _w

        _oai_comp.Completions.create = _nothink_wrap(_oai_comp.Completions.create)
        _oai_comp.AsyncCompletions.create = _nothink_hedge_async(_oai_comp.AsyncCompletions.create)
        sys.stderr.write("[mirofish_sim] netizen thinking=disabled patch active\n")
    except Exception as _e:  # noqa: BLE001  silently skip when openai isn't installed (doesn't affect non-simulation uses)
        sys.stderr.write(f"[mirofish_sim] nothink patch skipped: {_e}\n")


def call_llm(
    prompt: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    thinking: bool = True,
) -> str:
    url = DMX_BASE if DMX_BASE.endswith("/chat/completions") else DMX_BASE + "/chat/completions"
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if thinking:
        payload.update({
            "reasoning": {"effort": "high"},
            "include_reasoning": True,
            "thinking": {"type": "enabled", "budget_tokens": int(thinking_budget)},
        })
    else:
        payload["thinking"] = {"type": "disabled"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {DMX_KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=int(os.environ.get("MIROFISH_SIM_TIMEOUT", "1200"))) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
                _ledger_write(payload, content)  # L2 ledger (answering + personas): record request/response for later offline replay; failures silent
                return content
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"call_llm failed: {last}")


def _ledger_write(payload: dict, response: str) -> None:
    """L2 ledger recorder: append one call_llm request+response to a jsonl (path from MIROFISH_LEDGER).
    Records only call_llm (answering/personas), not netizen async calls (huge volume; recording could disturb the rollout). Fully wrapped in try, never affects the main flow."""
    path = os.environ.get("MIROFISH_LEDGER", "").strip()
    if not path:
        return
    try:
        rec = {"ts": time.time(), "model": payload.get("model"),
               "messages": payload.get("messages"),
               "temperature": payload.get("temperature"),
               "thinking": payload.get("thinking"), "response": response}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _is_en() -> bool:
    return os.environ.get("SB_AGENT_LANG", "zh").strip().lower() == "en"


def make_personas(context: str, n: int, model: str, max_tokens: int, thinking_budget: int) -> list:
    if _is_en():
        p = (
            f"Based on the event background below, design {n} virtual social-media users who would "
            "discuss this event. Make their stances and identities diverse (covering different "
            "opinion camps, levels of involvement, and professional backgrounds).\n"
            'One JSON object per user, fields: {"username","bio","persona","mbti","gender","age","country"}.\n'
            "Output only a single JSON array, no explanation.\n\nBackground:\n" + (context[:4000])
        )
    else:
        p = (
            f"根据下面的事件背景,设计 {n} 个会在社交媒体上讨论此事的虚拟网民,"
            "立场与身份要多样(覆盖不同观点阵营、不同卷入度、不同专业背景)。\n"
            '每人一个 JSON 对象,字段:{"username","bio","persona","mbti","gender","age","country"}。\n'
            "只输出一个 JSON 数组,不要解释。\n\n背景:\n" + (context[:4000])
        )
    raw = call_llm(p, model, max_tokens=max_tokens, thinking_budget=thinking_budget).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list):
            arr = []
    except Exception:  # noqa: BLE001
        arr = []
    out = []
    for i in range(n):
        a = arr[i] if i < len(arr) and isinstance(arr[i], dict) else {}
        out.append({
            "username": str(a.get("username") or f"user{i}"),
            "bio": str(a.get("bio") or ("a netizen following this event" if _is_en() else "一名关注该事件的网民")),
            "persona": str(a.get("persona") or ("holds a neutral, observing stance on this event and comments rationally" if _is_en() else "对该事件持中立观察态度,会理性发表看法")),
            "mbti": str(a.get("mbti") or "INTP"),
            "gender": str(a.get("gender") or "unknown"),
            "age": str(a.get("age") or 30),
            "country": str(a.get("country") or "Unknown"),
        })
    return out


# ====================== v7 (decided 2026-07-03, "align everything"): reference-style population + sparse activation ======================
# Switch: MIROFISH_SPARSE=1 enables. Parameter provenance: 666ghj/MiroFish:
#   simulation_config_generator.py:602-603 (activation formula) / run_reddit_simulation.py:478-520 (activity gate + selection)
#   / ontology_generator.py (hard ontology rules) / oasis_profile_generator.py (per-entity 2000-char personas,
#   temperature 0.7 decreasing on retry, individual/org dual templates, parallelism 5).
# Population = full entity set from the material (the reference has no clamp, no invented padding);
# _POP_MAX is only a cost fuse.
_SPARSE = os.environ.get("MIROFISH_SPARSE", "0") != "0"
_POP_MAX = int(os.environ.get("MIROFISH_POP_MAX", "60"))  # pure fuse; no such concept in the reference
_ACTIVITY_LEVEL = float(os.environ.get("MIROFISH_ACTIVITY_LEVEL", "0.5"))  # reference activity_level default 0.5
_PERSONA_PAR = 5  # reference parallel_count=5 (parallelism of per-entity persona generation)
# Individual/org type lists from reference oasis_profile_generator.py:170-179 (unknown custom types map to the org template, per reference logic)
_IND_TYPES = {"student", "alumni", "professor", "person", "publicfigure",
              "expert", "faculty", "official", "journalist", "activist"}


def _parse_json_array(raw: str) -> list:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else []
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                return arr if isinstance(arr, list) else []
            except Exception:  # noqa: BLE001
                return []
        return []


def extract_entities(context: str, model: str, max_tokens: int, thinking_budget: int) -> list:
    """Extract entities from the material following the reference ontology rules (six fields per
    entity). Population = full entity set from the material: no target count, no padding, no
    invention (reference population = all ontology entities in the graph, count decided by the
    material); _POP_MAX is only a cost fuse. Temperature 0.3 matches the reference ontology generation."""
    if _is_en():
        p = (
            "You are the entity extractor for a public-opinion simulation. From the event material "
            "below, extract the REAL-WORLD SUBJECTS that would speak about (or be discussed in) this "
            "event on social media, to be turned into virtual netizens.\n"
            "Hard rules (from the reference ontology):\n"
            "1. Entity types: design at most 8 event-specific types (in the style of Official/"
            "Journalist/Company/GovernmentAgency/MediaOutlet/Expert/Activist/AffectedGroupRep), plus "
            "the two fallback types Person and Organization — at most 10 types total.\n"
            "2. Every entity MUST be a real-world subject able to speak on social media (person/"
            "company/institution/government body/media outlet/representative of an affected group). "
            "FORBIDDEN: abstract concepts (public opinion, sentiment), topics, or stance camps "
            "('supporters'); a *representative netizen* of a group is allowed.\n"
            "3. Count: extract ALL such subjects that actually appear in the material, covering every "
            "side — err on the side of completeness. ALSO: for every distinct stance camp or affected "
            "group that appears in the material (e.g. 'skeptical netizens', 'supporters'), extract ONE "
            "representative netizen (type=AffectedGroupRep or Person, with a descriptive name like "
            "'skeptic-camp netizen rep') — these groups genuinely appear in the material, so this is "
            "grounded, not invention. Beyond that, do NOT pad or invent subjects absent from the "
            "material.\n"
            'Each entity: {"name","type","summary"(<=50 words),"attributes"(object, may be empty),'
            '"facts"(3-8 natural-language facts linking it to the event),"related"(list of related '
            "entity names, may be empty)}\n"
            "Output ONLY one JSON array.\n\nMaterial:\n" + context[:120000]
        )
    else:
        p = (
            "你是舆情仿真的实体抽取器。从下面的事件材料中,抽取会在社交媒体上就此事发声/被讨论的"
            "**现实真实主体**,用于生成虚拟网民。\n"
            "硬规则(参照基准本体):\n"
            "1. 实体类型:先为本事件定制不超过8个专属类型(风格如 Official/Journalist/Company/"
            "GovernmentAgency/MediaOutlet/Expert/Activist/AffectedGroupRep),另加兜底类型 Person 与 "
            "Organization,共不超过10类;\n"
            "2. 实体必须是现实中真实存在、能在社媒发声或被讨论的主体(个人/公司/机构/政府部门/媒体/"
            "受影响群体的代表网民);**禁止**抽象概念(舆论、情绪)、话题、立场阵营(如'支持方')——"
            "但'某群体的一位代表网民'允许;\n"
            "3. 数量:把材料中**实际出现**的这类主体全部抽出、覆盖各方,宁全勿漏;此外,材料中出现的"
            "每个明显立场阵营/受影响群体(如'质疑的网友''支持者'这类群体表述),各抽1名**代表网民**"
            "(type=AffectedGroupRep或Person,名字用描述性代称如'质疑派网民代表')——这些群体在材料中"
            "真实出现,属于材料落地,不算虚构;除此之外不要凑数、不要虚构材料里没有的主体。\n"
            '每个实体输出:{"name","type","summary"(50字内),"attributes"(对象,可空),'
            '"facts"(3-8条与事件关联的自然语言事实),"related"(相关实体名列表,可空)}\n'
            "只输出一个JSON数组,不要解释。\n\n材料:\n" + context[:120000]
        )
    arr = _parse_json_array(call_llm(p, model, max_tokens=max_tokens, temperature=0.3,
                                     thinking_budget=thinking_budget))
    ents = [e for e in arr if isinstance(e, dict) and e.get("name")]
    if len(ents) > _POP_MAX:  # pure cost fuse (no clamp in the reference); truncation always leaves a trace
        sys.stderr.write(f"[mirofish_sim] population fuse: {len(ents)} entities truncated to {_POP_MAX}\n")
        ents = ents[:_POP_MAX]
    return ents


def _parse_json_obj(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                o = json.loads(m.group(0))
                return o if isinstance(o, dict) else {}
            except Exception:  # noqa: BLE001
                return {}
        return {}


def _persona_prompt(e: dict, context: str) -> str:
    """Reference per-entity persona prompt (word-for-word match of the dual templates in oasis_profile_generator.py:690-775; English events use its English mirror)."""
    ind = str(e.get("type") or "person").lower() in _IND_TYPES
    attrs = json.dumps(e.get("attributes") or {}, ensure_ascii=False)
    facts = "; ".join(map(str, e.get("facts") or [])) or ("无" if not _is_en() else "none")
    head = (f"实体名称: {e.get('name')}\n实体类型: {e.get('type')}\n实体摘要: {e.get('summary')}\n"
            f"实体属性: {attrs}\n实体相关事实: {facts}\n\n上下文信息:\n{context[:3000]}"
            if not _is_en() else
            f"Entity name: {e.get('name')}\nEntity type: {e.get('type')}\nEntity summary: {e.get('summary')}\n"
            f"Entity attributes: {attrs}\nEntity facts: {facts}\n\nContext:\n{context[:3000]}")
    if not _is_en():
        if ind:
            return ("为实体生成详细的社交媒体用户人设,最大程度还原已有现实情况。\n\n" + head + "\n\n"
                    "请生成JSON,包含以下字段:\n"
                    "1. bio: 社交媒体简介,200字\n"
                    "2. persona: 详细人设描述(2000字的纯文本),需包含: 基本信息(年龄、职业、教育背景、所在地)、"
                    "人物背景(重要经历、与事件的关联、社会关系)、性格特征(MBTI类型、核心性格、情绪表达方式)、"
                    "社交媒体行为(发帖频率、内容偏好、互动风格、语言特点)、立场观点(对话题的态度、可能被激怒/感动的内容)、"
                    "独特特征(口头禅、特殊经历、个人爱好)、个人记忆(人设的重要部分,要介绍这个个体与事件的关联,"
                    "以及这个个体在事件中的已有动作与反应)\n"
                    "3. age: 年龄数字(必须是整数)\n4. gender: 性别,必须是英文\"male\"或\"female\"\n"
                    "5. mbti: MBTI类型(如INTJ、ENFP等)\n6. country: 国家\n7. profession: 职业\n"
                    "8. interested_topics: 感兴趣话题数组\n\n"
                    "重要: persona必须是一段连贯的文字描述,不要使用换行符;内容要与实体信息保持一致;只输出JSON对象。")
        return ("为机构/群体实体生成详细的社交媒体账号设定,最大程度还原已有现实情况。\n\n" + head + "\n\n"
                "请生成JSON,包含以下字段:\n"
                "1. bio: 官方账号简介,200字,专业得体\n"
                "2. persona: 详细账号设定描述(2000字的纯文本),需包含: 机构基本信息(正式名称、机构性质、成立背景、主要职能)、"
                "账号定位(账号类型、目标受众、核心功能)、发言风格(语言特点、常用表达、禁忌话题)、"
                "发布内容特点(内容类型、发布频率、活跃时间段)、立场态度(对核心话题的官方立场、面对争议的处理方式)、"
                "特殊说明(代表的群体画像、运营习惯)、机构记忆(机构人设的重要部分,要介绍这个机构与事件的关联,"
                "以及这个机构在事件中的已有动作与反应)\n"
                "3. age: 固定填30(机构账号的虚拟年龄)\n4. gender: 固定填\"other\"(机构账号)\n"
                "5. mbti: MBTI类型,用于描述账号风格,如ISTJ代表严谨保守\n6. country: 国家\n"
                "7. profession: 机构职能描述\n8. interested_topics: 关注领域数组\n\n"
                "重要: persona必须是一段连贯的文字描述,不要使用换行符;机构账号发言要符合其身份定位;只输出JSON对象。")
    if ind:
        return ("Create a detailed social-media user persona for the entity, staying maximally faithful "
                "to the known reality.\n\n" + head + "\n\n"
                "Generate JSON with fields:\n"
                "1. bio: social-media bio, ~60 words\n"
                "2. persona: detailed persona description (~1000 words plain text) covering: basic info "
                "(age, occupation, education, location); background (key experiences, link to the event, "
                "social relations); personality (MBTI, core traits, emotional style); social-media "
                "behavior (posting frequency, content preferences, interaction style, language habits); "
                "stance (attitude toward the topic, what would anger/move them); distinctive traits "
                "(catchphrases, special experiences, hobbies); personal memory (KEY part: the entity's "
                "OWN involvement in the event and its actions/reactions so far)\n"
                "3. age: integer\n4. gender: \"male\" or \"female\"\n5. mbti: e.g. INTJ\n6. country\n"
                "7. profession\n8. interested_topics: array\n\n"
                "Important: persona must be one coherent paragraph without newlines; stay consistent "
                "with the entity dossier; output ONLY the JSON object.")
    return ("Create a detailed social-media OFFICIAL-account profile for the organization/group entity, "
            "staying maximally faithful to the known reality.\n\n" + head + "\n\n"
            "Generate JSON with fields:\n"
            "1. bio: official account bio, ~60 words, professional\n"
            "2. persona: detailed account profile (~1000 words plain text) covering: basic info (formal "
            "name, nature, background, functions); account positioning (type, audience, core function); "
            "speaking style (language traits, common phrasings, taboo topics); content characteristics "
            "(types, frequency, active hours); stance (official position on the core topic, how it "
            "handles controversy); special notes (represented group profile, operating habits); "
            "institutional memory (KEY part: the org's involvement in the event and its actions/"
            "reactions so far)\n"
            "3. age: fixed 30\n4. gender: fixed \"other\"\n5. mbti: style descriptor, e.g. ISTJ\n"
            "6. country\n7. profession: institutional function\n8. interested_topics: array\n\n"
            "Important: persona must be one coherent paragraph without newlines; official tone; "
            "output ONLY the JSON object.")


def _persona_fallback(e: dict) -> dict:
    """In the spirit of the reference rule-based fallback (_generate_profile_rule_based): build from the entity's existing fields, no invention."""
    ind = str(e.get("type") or "person").lower() in _IND_TYPES
    summ = str(e.get("summary") or "")
    facts = " ".join(map(str, e.get("facts") or []))
    return {
        "bio": summ[:200] or f"{e.get('type')}: {e.get('name')}",
        "persona": (summ + " " + facts).strip() or f"{e.get('name')} — {e.get('type')}",
        "age": 30 if not ind else 35,
        "gender": "other" if not ind else "male",
        "mbti": "ISTJ" if not ind else "INTP",
        "country": "Unknown",
        "profession": str(e.get("type") or ""),
        "interested_topics": ["General"],
    }


def _gen_one_persona(e: dict, context: str, model: str) -> dict:
    """Reference per-entity generation: 3 retries, temperature 0.7-0.1*attempt, non-reasoning; falls back to the rule-based profile on failure (oasis_profile_generator.py:524-560)."""
    ind = str(e.get("type") or "person").lower() in _IND_TYPES
    a = {}
    for attempt in range(3):
        try:
            raw = call_llm(_persona_prompt(e, context), model, max_tokens=16000,
                           temperature=0.7 - attempt * 0.1, thinking=False)
            a = _parse_json_obj(raw)
            if a.get("persona"):
                break
        except Exception:  # noqa: BLE001
            a = {}
    if not a:
        a = _persona_fallback(e)
    if not a.get("bio"):
        a["bio"] = str(e.get("summary") or "")[:200] or str(e.get("name"))
    if not a.get("persona"):
        a["persona"] = str(e.get("summary") or e.get("name"))
    try:
        age = int(a.get("age") or 30)
    except Exception:  # noqa: BLE001
        age = 30
    if not ind:  # reference fixed fields for org accounts
        age, a["gender"] = 30, "other"
    g = str(a.get("gender") or "").lower()
    return {
        "username": str(e.get("name") or a.get("username") or "user")[:40],
        "bio": str(a.get("bio")),
        "persona": str(a.get("persona")),
        "mbti": str(a.get("mbti") or ("ISTJ" if not ind else "INTP")),
        "gender": g if g in ("male", "female", "other") else ("other" if not ind else "male"),
        "age": age,
        "country": str(a.get("country") or "Unknown"),
        "profession": str(a.get("profession") or e.get("type") or ""),
        "interested_topics": a.get("interested_topics") if isinstance(a.get("interested_topics"), list) else ["General"],
    }


def personas_from_entities(entities: list, context: str, model: str,
                           max_tokens: int, thinking_budget: int) -> list:
    """Reference form: one independent LLM call per entity generating a ~2000-char persona, thread pool parallelism 5 (parallel_count=5)."""
    import concurrent.futures
    out = [None] * len(entities)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PERSONA_PAR) as ex:
        futs = {ex.submit(_gen_one_persona, e, context, model): i for i, e in enumerate(entities)}
        for f in concurrent.futures.as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception:  # noqa: BLE001
                out[i] = _persona_fallback(entities[i]) | {"username": str(entities[i].get("name"))[:40]}
    return [o for o in out if o]


def make_population(context: str, model: str, max_tokens: int, thinking_budget: int) -> list:
    ents = extract_entities(context, model, max_tokens, thinking_budget)
    profs = personas_from_entities(ents, context, model, max_tokens, thinking_budget)
    n = len(profs)
    sys.stderr.write(f"[mirofish_sim] v7 population: {n} agents (entities={len(ents)}, "
                     f"activation=[{max(1, n // 15)},{max(5, n // 5)}]×gate{_ACTIVITY_LEVEL})\n")
    return profs


def _step_actors(env) -> list:
    """Per-step actors — word-for-word adaptation of the reference _get_active_agents_for_round
    (run_reddit_simulation.py:478-520, minus the day/night multiplier: our time points have no clock):
      target = int(uniform(max(1,N//15), max(5,N//5)))   # reference formula band (config_generator:602-603)
      candidates = each netizen passes an activity_level (default 0.5) coin flip   # reference activity gate
      selected = sample(candidates, min(target, len(candidates)))"""
    agents = [ag for _, ag in env.agent_graph.get_agents()]
    if not _SPARSE or not agents:
        return agents
    n = len(agents)
    lo = max(2, n // 15)  # lower bound 2 = calibrated 2026-07-03: the reference's bound 1 targets large populations; with small populations single-actor steps fire unreasonably often
    hi = max(5, n // 5)
    target = int(random.uniform(lo, min(n, hi)))
    target = max(lo, min(target, n))
    candidates = [ag for ag in agents if random.random() < _ACTIVITY_LEVEL]
    if len(candidates) < min(target, n):  # fallback when the activity gate filters out too many (low probability); guarantees the per-step target count
        candidates = agents
    return random.sample(candidates, min(target, len(candidates)))


async def run_sim(personas: list, context: str, base_model: str, n_steps: int, db_path: str) -> None:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=base_model,
        api_key=DMX_KEY,
        url=DMX_BASE,
        # The reference passes no model_config_dict (camel treats max_tokens as the whole context budget -> chunking flood; temperature is left to the server default)
    )
    prof = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(personas, prof, ensure_ascii=False)
    prof.close()

    # The reference's original 13-action Reddit list (run_reddit_simulation.py:389-403; REPOST is Twitter-side, absent on Reddit)
    actions = [ActionType.LIKE_POST, ActionType.DISLIKE_POST, ActionType.CREATE_POST,
               ActionType.CREATE_COMMENT, ActionType.LIKE_COMMENT, ActionType.DISLIKE_COMMENT,
               ActionType.SEARCH_POSTS, ActionType.SEARCH_USER, ActionType.TREND,
               ActionType.REFRESH, ActionType.DO_NOTHING, ActionType.FOLLOW, ActionType.MUTE]
    graph = await generate_reddit_agent_graph(profile_path=prof.name, model=model, available_actions=actions)

    if os.path.exists(db_path):
        os.remove(db_path)
    env = oasis.make(agent_graph=graph, platform=oasis.DefaultPlatformType.REDDIT,
                     database_path=db_path, semaphore=30)  # semaphore=30 as in the reference (throttles against API overload)
    await env.reset()
    seed = ("[Event Bulletin] " if _is_en() else "【事件速报】") + context[:8000]  # 8000 matches the reference event-config context length; no more silent truncation
    a0 = env.agent_graph.get_agent(0)
    await env.step({a0: ManualAction(action_type=ActionType.CREATE_POST, action_args={"content": seed})})
    for _ in range(max(1, n_steps)):
        await env.step({ag: LLMAction() for ag in _step_actors(env)})
    await env.close()
    try:
        os.unlink(prof.name)
    except OSError:
        pass


def read_transcript(db_path: str, limit: int = 100) -> list:
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    texts = []
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [c[1] for c in cur.fetchall()]
            for tc in [c for c in cols if c.lower() in ("content", "text", "body", "message", "comment")]:
                cur.execute(f"SELECT {tc} FROM {t} ORDER BY rowid")  # noqa: S608 (trusted col name)
                texts += [r[0] for r in cur.fetchall() if isinstance(r[0], str) and len(r[0]) > 5]
        except Exception:  # noqa: BLE001
            pass
    con.close()
    seen, out = set(), []
    for x in texts:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[-limit:]  # v7: keep the newest when over limit (previously kept the oldest, so late answering only saw stale opinion)


def _hedged_llm(prompt: str, model: str, max_tokens: int, thinking_budget: int) -> str:
    """Hedged answering (decided twice on 2026-07-03: enabled after the 39-min trigger breach;
    later upgraded to **three lanes fired together**).
    MIROFISH_HEDGE_LANES (default 3) concurrent lanes on the same question, first result wins
    (taken by arrival, unbiased); MIROFISH_HEDGE_DELAY>0 reverts to hedge mode (backup fires
    only after the first lane times out)."""
    import concurrent.futures as _f
    lanes = max(1, int(os.environ.get("MIROFISH_HEDGE_LANES", "3")))
    delay = float(os.environ.get("MIROFISH_HEDGE_DELAY", "0"))  # 0 = fire all at once
    ex = _f.ThreadPoolExecutor(max_workers=lanes)
    try:
        futs = [ex.submit(call_llm, prompt, model, max_tokens, 0.0, thinking_budget)]
        if delay > 0 and lanes > 1:
            try:
                return futs[0].result(timeout=delay)
            except _f.TimeoutError:
                sys.stderr.write(f"[hedge] first call >{delay:.0f}s, firing backups\n")
            except Exception:  # noqa: BLE001
                sys.stderr.write("[hedge] first call failed, firing backups\n")
        while len(futs) < lanes:
            futs.append(ex.submit(call_llm, prompt, model, max_tokens, 0.0, thinking_budget))
        pending, last_err = set(futs), None
        while pending:
            done, pending = _f.wait(pending, return_when=_f.FIRST_COMPLETED)
            for d in done:
                try:
                    return d.result()
                except Exception as e:  # noqa: BLE001
                    last_err = e
        raise RuntimeError(f"hedged call: all {lanes} lanes failed: {last_err}")
    finally:
        ex.shutdown(wait=False)


def reader(prompt: str, transcript: list, model: str, max_tokens: int,
           thinking_budget: int, roster: str = None) -> str:
    # v7: fill from newest backwards (previously filled from oldest + head truncation, doubly losing the newest opinion); display stays chronological.
    # Answering opinion sample size is tunable (after rollout/answer decoupling this is a pure answering-side param; enlarging doesn't affect or rerun the rollout):
    # unlimited mode: set MIROFISH_READ_MAX_ITEMS/CHARS very large to feed everything.
    # transcript may be rich-record strings ([author·type·👍👎] body); roster is the netizen roster (optional).
    _max_items = int(os.environ.get("MIROFISH_READ_MAX_ITEMS", "80"))
    _max_chars = int(os.environ.get("MIROFISH_READ_MAX_CHARS", "6000"))
    sel, tot = [], 0
    for t in reversed(transcript[-_max_items:]):
        line = f"- {t}"
        if sel and tot + len(line) > _max_chars:
            break
        sel.append(line)
        tot += len(line) + 1
    joined = "\n".join(reversed(sel))[:_max_chars]
    if not transcript:
        note = ""
    elif _is_en():
        roster_block = (f"[1. Netizens in the discussion] (personas fixed at simulation start)\n{roster}\n\n"
                        if roster else "")
        note = ("The following are materials produced by the 「MiroFish social simulation」 for this "
                "event, serving as a reference signal for the crowd's judgment (may contain noise, "
                "use your own discretion). These are NOT real-world facts, but the product of a group "
                "of virtual netizens (driven by personas of the roles involved) discussing the event.\n\n"
                + roster_block +
                "[2. Posts & comments produced by netizens] (chronological; each tagged "
                "[author · post/comment · 👍likes/👎dislikes as of the prediction cutoff]; comments note the post they reply to)\n"
                + joined + "\n====\n\n"
                "Treat the above netizen opinion as a reference signal about the future, combine it "
                "with your own analysis, and complete the prediction task below, answering strictly "
                "in the required format.\n\n")
    else:
        roster_block = (f"【一、参与讨论的网民名册】(人设在推演开始时设定)\n{roster}\n\n"
                        if roster else "")
        note = ("以下是「MiroFish 社会仿真」针对该事件推演产生的材料,作为群体判断的参考信号"
                "(可能含噪声,请自行甄别)。这些不是新闻事实,而是一群由该事件相关角色人设驱动的"
                "虚拟网民围绕事件讨论后的产物。\n\n"
                + roster_block +
                "【二、网民产生的帖子与评论】(按时间先后;每条标注[作者·类型·截至预测时点的"
                "👍点赞/👎点踩];评论标注其回应的帖)\n"
                + joined + "\n====\n\n"
                "请把上述「网民舆情」当作对未来的群体判断参考信号,结合你自己的分析,完成下面的"
                "预测任务,严格按其格式作答。\n\n")
    return _hedged_llm(note + prompt, model, max_tokens, thinking_budget)  # v7.1 hedged exit (trigger already fired)


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    base = job["base_model"]
    context = job.get("context", "") or ""
    prompt = job["prompt"]
    n_agents = int(job.get("n_agents", 8))
    n_steps = int(job.get("n_steps", 3))
    max_tokens = int(job.get("max_tokens", DEFAULT_MAX_TOKENS))
    thinking_budget = int(job.get("thinking_budget", DEFAULT_THINKING_BUDGET))
    db_path = job.get("db_path") or tempfile.mktemp(suffix=".db")

    # Multiple question batches for the same prediction point share one social simulation: the parent process caches the transcript, and later batches pass it in to skip the simulation.
    transcript = job.get("transcript")
    if transcript is None:
        transcript = []
        try:
            personas = (make_population(context, base, max_tokens, thinking_budget) if _SPARSE else make_personas(context, n_agents, base, max_tokens, thinking_budget))
            asyncio.run(run_sim(personas, context, base, n_steps, db_path))
            transcript = read_transcript(db_path)
            sys.stderr.write(f"[mirofish_sim] sim OK: {len(personas)} agents, {n_steps} steps, "
                             f"{len(transcript)} discussion items\n")
        except Exception as e:  # noqa: BLE001
            # Simulation-failure fallback: leave transcript empty, reader degrades to plain LLM (guaranteed not to crash), but warn loudly.
            sys.stderr.write(f"[mirofish_sim] SIM-FAILED (fallback to plain reader): {type(e).__name__}: {e}\n")
    else:
        sys.stderr.write(f"[mirofish_sim] 复用缓存 transcript({len(transcript)} 条),跳过仿真\n")

    ans = reader(prompt, transcript, base, max_tokens, thinking_budget)
    sys.stdout.write("<<<TRANSCRIPT>>>\n" + json.dumps(transcript, ensure_ascii=False)
                     + "\n<<<ANSWER>>>\n" + ans + "\n<<<END>>>\n")


# ===========================================================================
# Resident mode (M3 single continuous rollout): build once -> multiple step spans along the
# timeline -> accumulate DB reads across spans.
#   The parent (fw_mirofish_oasis.make_session) writes JSON commands to stdin line by line:
#     {"cmd":"ADVANCE","delta":...,"cutoff":...,"steps":N, first call also carries base_model/n_agents/...}
#     {"cmd":"ANSWER","prompt":...,"kind":...}
#     {"cmd":"CLOSE"}
#   This process writes each response to stdout as a <<<SBRESP>>>\n{json}\n<<<SBEND>>> frame.
#   Key: point sys.stdout at stderr entirely so any library (OASIS/camel) print goes to stderr;
#   protocol frames use only the saved real stdout fd — no noise can pollute the protocol.
# ===========================================================================

def run_resident() -> None:
    import traceback

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # from here on any print() goes to stderr, never touching the protocol channel

    def emit(obj: dict) -> None:
        real_stdout.write("<<<SBRESP>>>\n" + json.dumps(obj, ensure_ascii=False) + "\n<<<SBEND>>>\n")
        real_stdout.flush()

    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st = {"env": None, "a0": None, "db_path": None, "prof": None,
          "transcript": None, "base": None,
          "max_tokens": DEFAULT_MAX_TOKENS, "thinking_budget": DEFAULT_THINKING_BUDGET, "n_steps": 3}

    async def build_env(context, base_model, n_agents, n_steps, max_tokens, thinking_budget):
        personas = (make_population(context, base_model, max_tokens, thinking_budget) if _SPARSE else make_personas(context, n_agents, base_model, max_tokens, thinking_budget))
        # Persistence: with MIROFISH_PROFILE_PATH set, full personas are written to the persistent area (one per event), otherwise /tmp.
        _pp = os.environ.get("MIROFISH_PROFILE_PATH")
        if _pp:
            os.makedirs(os.path.dirname(_pp), exist_ok=True) if os.path.dirname(_pp) else None
            with open(_pp, "w", encoding="utf-8") as _pf:
                json.dump(personas, _pf, ensure_ascii=False)
            profname = _pp
        else:
            prof = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(personas, prof, ensure_ascii=False); prof.close()
            profname = prof.name
        gmodel = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI, model_type=base_model,
            api_key=DMX_KEY, url=DMX_BASE,
            # The reference passes no model_config_dict (camel treats max_tokens as the whole context budget -> chunking flood; temperature is left to the server default)
        )
        # The reference's original 13-action Reddit list (run_reddit_simulation.py:389-403; REPOST is Twitter-side, absent on Reddit)
        actions = [ActionType.LIKE_POST, ActionType.DISLIKE_POST, ActionType.CREATE_POST,
                   ActionType.CREATE_COMMENT, ActionType.LIKE_COMMENT, ActionType.DISLIKE_COMMENT,
                   ActionType.SEARCH_POSTS, ActionType.SEARCH_USER, ActionType.TREND,
                   ActionType.REFRESH, ActionType.DO_NOTHING, ActionType.FOLLOW, ActionType.MUTE]
        graph = await generate_reddit_agent_graph(profile_path=profname, model=gmodel,
                                                  available_actions=actions)
        # Persistence: with MIROFISH_DB_PATH set, the DB is written to the persistent area (one per event, no mixing), otherwise falls back to /tmp.
        db_path = os.environ.get("MIROFISH_DB_PATH") or tempfile.mktemp(suffix=".db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        if os.path.exists(db_path):
            os.remove(db_path)
        env = oasis.make(agent_graph=graph, platform=oasis.DefaultPlatformType.REDDIT,
                     database_path=db_path, semaphore=30)  # semaphore=30 as in the reference (throttles against API overload)
        await env.reset()
        return env, env.agent_graph.get_agent(0), db_path, profname, len(personas)

    async def advance(seed, steps):
        env, a0 = st["env"], st["a0"]
        await env.step({a0: ManualAction(action_type=ActionType.CREATE_POST,
                                         action_args={"content": ("[Event Bulletin] " if _is_en() else "【事件速报】") + seed[:8000]})})  # v7: 1500->8000, incremental news no longer silently truncated
        for _ in range(max(1, steps)):
            await env.step({ag: LLMAction() for ag in _step_actors(env)})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            cmd = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            emit({"ok": False, "error": f"bad json: {e}"}); continue
        c = cmd.get("cmd")
        try:
            if c == "ADVANCE":
                delta = (cmd.get("delta") or "").strip()
                steps = int(cmd.get("steps", st["n_steps"]))
                built = False
                if st["env"] is None:
                    st["base"] = cmd.get("base_model")
                    st["max_tokens"] = int(cmd.get("max_tokens", DEFAULT_MAX_TOKENS))
                    st["thinking_budget"] = int(cmd.get("thinking_budget", DEFAULT_THINKING_BUDGET))
                    st["n_steps"] = int(cmd.get("n_steps", 3))
                    n_agents = int(cmd.get("n_agents", 8))
                    env, a0, db, profname, npx = loop.run_until_complete(
                        build_env(delta, st["base"], n_agents, steps, st["max_tokens"], st["thinking_budget"]))
                    st.update({"env": env, "a0": a0, "db_path": db, "prof": profname})
                    built = True
                if delta:
                    def _trace_n():
                        try:
                            _c = sqlite3.connect(st["db_path"])
                            _n = _c.execute("SELECT COUNT(*) FROM trace WHERE action != 'sign_up'").fetchone()[0]
                            _c.close()
                            return _n
                        except Exception:  # noqa: BLE001
                            return -1
                    _pre = _trace_n()
                    loop.run_until_complete(advance(delta, steps))
                    _post = _trace_n()
                    # Empty-scene gate (2026-07-03): if this point's rollout produced only the bulletin post
                    # (new actions <= 1), treat the rollout as failed -> hand to persistent retry (wait for
                    # upstream recovery); never enter answering with an empty scene.
                    if _pre >= 0 and _post >= 0 and (_post - _pre) <= 1:
                        raise RuntimeError(
                            f"empty advance: no netizen actions this point (pre={_pre}, post={_post}); "
                            "upstream likely down — retry")
                # Decided 2026-07-03 (under test validation): after each advance, trim netizens'
                # **private conversation memory** (camel ChatAgent.reset() = back to only the
                # persona system message). Platform opinion (posts/comments, sqlite db) is fully
                # kept — transcript/readout unaffected, opinion trajectory continuous throughout.
                # Motivation: camel accumulates agent private history without bound; in long
                # resident runs advance ballooned linearly from 2min to 20min/point. Mainstream
                # usage (short one-shot OASIS / original MiroFish rebuilding per point) never has
                # long netizen private history, and camel natively supports bounded memory via
                # message_window_size. Set MIROFISH_MEM_TRIM=0 to revert.
                # Decided 2026-07-05: K's unit is **rounds** (how many "observe+act" rounds each
                # netizen keeps), not memory records. One round = 1 USER record (feed-observation
                # start) + the action records after it.
                # Trim = each netizen keeps [persona system message + last K rounds]; USER record
                # count = round count. This controls chatty and quiet netizens uniformly by
                # "last K rounds of experience", decoupled from record counts.
                # K=0 falls back to the old full reset (MEM_TRIM).
                _keep = int(os.environ.get("MIROFISH_MEM_KEEP", "0"))
                if _keep > 0 and st["env"] is not None:
                    try:
                        from camel.types import OpenAIBackendRole as _OBR
                        _pre, _cut_a, _cut_r = [], 0, 0
                        for _, _ag in st["env"].agent_graph.get_agents():
                            _lst = _ag.memory._chat_history_block.storage.memory_list
                            # each USER record = the start of one round (skip record 0, the persona system message)
                            _uidx = [i for i, _r in enumerate(_lst)
                                     if i > 0 and getattr(_r, "role_at_backend", None) == _OBR.USER]
                            _pre.append(len(_uidx))  # this netizen's cumulative round count
                            if len(_uidx) > _keep:
                                # keep from the USER start of the K-th-from-last round; cut everything before it (except the persona system message)
                                _cut_to = _uidx[len(_uidx) - _keep]
                                _cut_a += 1
                                _cut_r += _cut_to - 1
                                _lst[1:_cut_to] = []  # keep record 0 (persona system message) + last K rounds
                        sys.stderr.write(
                            f"[mem-K] agents={len(_pre)} pre_max_rounds={max(_pre) if _pre else 0} "
                            f"pre_avg_rounds={(sum(_pre)/len(_pre)) if _pre else 0:.1f} keep_rounds={_keep} "
                            f"cut_agents={_cut_a} cut_records={_cut_r}\n")
                    except Exception as _te:  # noqa: BLE001  trim failure is non-fatal, continue (just slower)
                        sys.stderr.write(f"[mem-K] trim failed (continue): {_te}\n")
                elif os.environ.get("MIROFISH_MEM_TRIM", "1") != "0" and st["env"] is not None:
                    try:
                        _n_trim = 0
                        for _, _ag in st["env"].agent_graph.get_agents():
                            _ag.reset()
                            _n_trim += 1
                        sys.stderr.write(f"[mirofish_sim] mem-trim: reset {_n_trim} agents' private memory\n")
                    except Exception as _te:  # noqa: BLE001  trim failure is non-fatal, continue (just slower)
                        sys.stderr.write(f"[mirofish_sim] mem-trim failed (continue): {_te}\n")
                st["transcript"] = None  # new developments invalidate the old transcript cache
                emit({"ok": True, "event": "built_and_advanced" if built else "advanced",
                      "cutoff": cmd.get("cutoff")})
            elif c == "ANSWER":
                prompt = cmd.get("prompt", "")
                if st["base"] is None:
                    emit({"ok": False, "error": "answer before any advance (no sim built)"}); continue
                if st["transcript"] is None:
                    st["transcript"] = read_transcript(st["db_path"]) if st["db_path"] else []
                ans = reader(prompt, st["transcript"], st["base"], st["max_tokens"], st["thinking_budget"])
                emit({"ok": True, "answer": ans, "transcript_n": len(st["transcript"])})
            elif c == "SNAPSHOT":
                # Rollout/answer decoupling (2026-07-05): dump the full opinion transcript up to the
                # current point and hand it to the driver to save. The answering stage reads the
                # snapshot directly, no live simulation needed; store as much as possible (large
                # limit), how much the answering side reads is decided by answering params.
                if st["db_path"] is None:
                    emit({"ok": False, "error": "snapshot before any advance"}); continue
                _tr = read_transcript(st["db_path"], limit=int(cmd.get("limit", 3000)))
                emit({"ok": True, "transcript": _tr, "transcript_n": len(_tr)})
            elif c == "CLOSE":
                if st["env"] is not None:
                    try:
                        loop.run_until_complete(st["env"].close())
                    except Exception:  # noqa: BLE001
                        pass
                for p in (st.get("prof"),):
                    try:
                        # Persisted personas (MIROFISH_PROFILE_PATH) are kept; only /tmp temporary profiles are removed
                        if p and p != os.environ.get("MIROFISH_PROFILE_PATH"):
                            os.unlink(p)
                    except OSError:
                        pass
                emit({"ok": True, "event": "closed"})
                break
            else:
                emit({"ok": False, "error": f"unknown cmd {c}"})
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(traceback.format_exc())
            emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
    try:
        loop.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    if "--resident" in sys.argv[1:]:
        run_resident()
    else:
        main()
