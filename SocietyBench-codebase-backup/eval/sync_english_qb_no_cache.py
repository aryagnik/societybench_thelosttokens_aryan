#!/usr/bin/env python3
"""Synchronize English questionbank text from Chinese source without old cache.

This is a repair utility for final datasets. It preserves every scoring and
schema field from the current Chinese questionbank and only translates
model-facing text fields into English:

- brier_questions[].q
- brier_questions[].event_desc
- events.{base,all,major}[].event
- events.{base,all,major}[].event_desc
- brier_questions[].dimension / time_window are mapped deterministically

It deliberately does not read final/_translation_cache_en, because that cache
can contain stale cross-row translations.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs_new"
DROP_KEYS = {"q", "event", "event_desc", "dimension", "time_window"}
TEXT_FIELDS_Q = ("q", "event_desc")
TEXT_FIELDS_E = ("event", "event_desc")
EVENT_GROUPS = ("base", "all", "major")
CJK_RE = re.compile(r"[\u3400-\u9fff\uff00-\uffef]")
BAD_SHORT_EN_RE = re.compile(
    r"^(?:(?:Will|Did|Has|Have|Does|Do|Was|Were|Is|Are)\s+"
    r"|(?:In the next \d+ days, will|Within \d+ days before the target date, has)\s+)?"
    r"(?:Country|country|Leader|Company|Platform|Organization|Institution|Person|"
    r"University|Media|Rule|City|Agency|Region|Court|Party|Product|Market|Index|"
    r"Currency|Exchange|Analyst|Auditor|Case|Action|Strait)(?:\s+[A-Z]\d*)?\??$"
)
PH_REPLACEMENTS = [
    (re.compile(r"\b人物\s*([A-Z][0-9]?)\b"), r"Person \1"),
    (re.compile(r"\b机构\s*([A-Z][0-9]?)\b"), r"Institution \1"),
    (re.compile(r"\b组织\s*([A-Z][0-9]?)\b"), r"Organization \1"),
    (re.compile(r"\b国家\s*([A-Z][0-9]?)\b"), r"Country \1"),
    (re.compile(r"\b城市\s*([A-Z][0-9]?)\b"), r"City \1"),
    (re.compile(r"\b平台\s*([A-Z][0-9]?)\b"), r"Platform \1"),
    (re.compile(r"\b公司\s*([A-Z][0-9]?)\b"), r"Company \1"),
    (re.compile(r"\b大学\s*([A-Z][0-9]?)\b"), r"University \1"),
    (re.compile(r"\b学院\s*([A-Z][0-9]?)\b"), r"School \1"),
    (re.compile(r"\b媒体\s*([A-Z][0-9]?)\b"), r"Media \1"),
    (re.compile(r"\b法规\s*([A-Z][0-9]?)\b"), r"Rule \1"),
    (re.compile(r"\b考试\s*([A-Z][0-9]?)\b"), r"Exam \1"),
    (re.compile(r"\b疾病\s*([A-Z][0-9]?)\b"), r"Illness \1"),
    (re.compile(r"\b项目\s*([A-Z][0-9]?)\b"), r"Project \1"),
    (re.compile(r"\b编号\s*([A-Z][0-9]?)\b"), r"Identifier \1"),
]
PLACEHOLDER_TYPE_MAP = {
    "当选总统领导人": "President-elect",
    "候任总统领导人": "President-elect",
    "前总统领导人": "Former President",
    "总统领导人": "President",
    "最高领袖领导人": "Supreme Leader",
    "国家主席领导人": "State Leader",
    "总理领导人": "Prime Minister",
    "首相领导人": "Prime Minister",
    "省长领导人": "Governor",
    "领导人": "Leader",
    "首席执行官": "CEO",
    "新闻秘书人物": "Press Secretary",
    "发言人人物": "Spokesperson",
    "委员会主席人物": "Committee Chair",
    "众议员人物": "Representative",
    "参议员人物": "Senator",
    "参议员议员": "Senator",
    "议长人物": "Speaker",
    "贸易代表人物": "Trade Representative",
    "贸易代表官员": "Trade Representative",
    "副总理人物": "Vice Premier",
    "长官员": "Official",
    "主席人物": "Chair",
    "人物": "Person",
    "官员": "Official",
    "区域组织": "Regional Organization",
    "行政机构": "Agency",
    "审计机构": "Audit Firm",
    "贸易机构": "Trade Agency",
    "应用商店": "App Store",
    "图文应用": "Photo App",
    "社交媒体平台产品": "Social Media Product",
    "对照社交平台": "Comparison Platform",
    "对照大公司公司": "Comparison Company",
    "对照大公司地区": "Comparison Region",
    "母公司公司": "Parent Company",
    "平台": "Platform",
    "公司": "Company",
    "机构": "Institution",
    "国家": "Country",
    "地区": "Region",
    "城市": "City",
    "法院": "Court",
    "媒体": "Media",
    "政党": "Party",
    "法规": "Law",
    "规则": "Rule",
    "协定": "Agreement",
    "产品": "Product",
    "项目": "Project",
    "编号": "Identifier",
    "大学": "University",
    "学院": "School",
    "组织": "Organization",
    "疾病": "Illness",
    "考试": "Exam",
    "案件": "Case",
    "型号": "Model",
    "梗": "Meme",
    "市场": "Market",
    "行动": "Action",
    "关系": "Relationship",
    "部队": "Force",
    "海峡": "Strait",
    "交易所": "Exchange",
    "芯片商": "Chipmaker",
    "投行": "Investment Bank",
    "高管": "Executive",
    "企业家": "Entrepreneur",
    "货币": "Currency",
    "指数": "Index",
    "财年表格": "Fiscal Form",
    "分析师": "Analyst",
    "职务": "Position",
    "别称": "Alias",
}
PLACEHOLDER_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(PLACEHOLDER_TYPE_MAP, key=len, reverse=True)) + r")([A-Z]\d*)"
)

DIM_MAP = {
    "事件": "event",
    "政策": "policy",
    "舆论": "opinion",
    "数字": "figures",
    "司法": "judicial",
    "军事": "military",
    "网络": "cyber",
    "平台治理": "platform governance",
    "产品": "product",
    "商业": "business",
    "外交": "diplomacy",
    "安全": "security",
    "经济": "economy",
    "市场": "market",
    "监管": "regulation",
    "贸易": "trade",
    "法律": "legal",
    "交通": "transportation",
    "公司": "company",
    "制裁": "sanctions",
    "政治": "politics",
    "核政策": "nuclear policy",
    "治理": "governance",
    "社会": "society",
    "舆情": "public opinion",
    "金融": "finance",
    "金融制裁": "financial sanctions",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def strip_translatable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_translatable(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip_translatable(v) for v in obj]
    return obj


def collect_strings(qb: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in seen:
            seen.add(value)
            out.append(value)

    for group in EVENT_GROUPS:
        for event in qb.get("events", {}).get(group, []) or []:
            for field in TEXT_FIELDS_E:
                add(event.get(field))
    for q in qb.get("brier_questions", []) or []:
        for field in TEXT_FIELDS_Q:
            add(q.get(field))
    return out


def map_dimension(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return DIM_MAP.get(value, value)


def map_time_window(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    m = re.fullmatch(r"(\d+)\s*天", value.strip())
    if m:
        return f"{m.group(1)} days"
    if value == "全时段":
        return "full period"
    return value


def clean_translation(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    # Normalize common placeholder translations. Google often leaves these as
    # pinyin-like Chinese words plus letters when placeholders are anonymized.
    for pattern, repl in PH_REPLACEMENTS:
        s = pattern.sub(repl, s)
    s = s.replace("Deadline", "cutoff")
    s = s.replace("deadline", "cutoff")
    return s


def protect_placeholders(text: str) -> tuple[str, dict[str, str]]:
    """Protect anonymized Chinese placeholders before machine translation.

    Google Translate often collapses strings like ``国家X...`` or ``平台X...`` to
    just "Country" / "Platform". Replacing those spans with inert ASCII tokens
    before translation preserves the sentence body and restores readable
    placeholders afterward.
    """
    replacements: dict[str, str] = {}
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        zh_type, code = match.group(1), match.group(2)
        token = f"__PH{idx}__"
        idx += 1
        replacements[token] = f"{PLACEHOLDER_TYPE_MAP.get(zh_type, zh_type)} {code}"
        return f" {token} "

    protected = PLACEHOLDER_RE.sub(repl, text)
    return protected, replacements


def restore_placeholders(text: str, replacements: dict[str, str]) -> str:
    out = text
    for token, value in replacements.items():
        out = out.replace(token, value)
    return re.sub(r"\s+", " ", out).strip()


def translate_source(src: str, google_translate) -> str:
    protected, replacements = protect_placeholders(src)
    en = clean_translation(restore_placeholders(google_translate(protected), replacements))
    return en


def translate_all(
    strings: list[str],
    cache_path: Path,
    *,
    sleep_s: float,
    resume: bool,
    batch_size: int,
    workers: int,
    limit: int | None = None,
) -> dict[str, str]:
    if resume and cache_path.exists():
        cache = load_json(cache_path)
        if not isinstance(cache, dict):
            cache = {}
    else:
        cache = {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    todo = [s for s in strings if s not in cache]
    if limit is not None:
        todo = todo[:limit]
    total = len(todo)
    done = 0

    def google_translate(src: str) -> str:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            "?client=gtx&sl=zh-CN&tl=en&dt=t&q="
            + urllib.parse.quote(src)
        )
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read())
        parts = data[0] if isinstance(data, list) and data else []
        return "".join(str(part[0]) for part in parts if part and part[0])

    def translate_one(src: str) -> str:
        en = ""
        last_error = ""
        for attempt in range(3):
            try:
                en = translate_source(src, google_translate)
                if en:
                    break
            except Exception as exc:  # deep-translator has provider-specific errors
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5 * (attempt + 1))
        if not en:
            # Retry as a slightly more explicit sentence. This helps Google
            # Translate with terse question-bank fragments.
            try:
                wrapped = f"请翻译成英文：{src}"
                en = translate_source(wrapped, google_translate)
                en = re.sub(r"^(Please translate into English:|Translate into English:)\s*", "", en).strip()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if not en:
            dump_json(cache_path, cache)
            raise RuntimeError(f"translation failed for source: {src} ({last_error})")
        return en

    for start in range(0, total, max(1, batch_size)):
        batch = todo[start:start + max(1, batch_size)]
        if workers > 1 and len(batch) > 1:
            results: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(translate_one, src): src for src in batch}
                for fut in as_completed(futs):
                    src = futs[fut]
                    results[src] = fut.result()
            ordered = [(src, results[src]) for src in batch]
        else:
            ordered = [(src, translate_one(src)) for src in batch]
        for src, en in ordered:
            cache[src] = en
            done += 1
        if done % 25 == 0 or done == total:
            dump_json(cache_path, cache)
            print(f"translated {done}/{total} (cache {len(cache)})", flush=True)
        if sleep_s:
            time.sleep(sleep_s)
    dump_json(cache_path, cache)
    missing = [s for s in strings if s not in cache]
    if missing:
        raise RuntimeError(f"translation cache incomplete: {len(missing)} strings missing")
    return {s: str(cache[s]) for s in strings}


def apply_translation(cn_qb: dict[str, Any], tmap: dict[str, str]) -> dict[str, Any]:
    en_qb = deepcopy(cn_qb)

    def tr(value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return tmap[value]
        return value

    for group in EVENT_GROUPS:
        for event in en_qb.get("events", {}).get(group, []) or []:
            for field in TEXT_FIELDS_E:
                if field in event:
                    event[field] = tr(event[field])
    for q in en_qb.get("brier_questions", []) or []:
        for field in TEXT_FIELDS_Q:
            if field in q:
                q[field] = tr(q[field])
        if "dimension" in q:
            q["dimension"] = map_dimension(q["dimension"])
        if "time_window" in q:
            q["time_window"] = map_time_window(q["time_window"])
    return en_qb


def validate(cn_qb: dict[str, Any], en_qb: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if strip_translatable(cn_qb) != strip_translatable(en_qb):
        issues.append("non-text structure differs from Chinese source")
    model_facing: list[str] = []
    for group in EVENT_GROUPS:
        for event in en_qb.get("events", {}).get(group, []) or []:
            for field in TEXT_FIELDS_E:
                model_facing.append(str(event.get(field, "")))
    for q in en_qb.get("brier_questions", []) or []:
        for field in (*TEXT_FIELDS_Q, "dimension", "time_window"):
            model_facing.append(str(q.get(field, "")))
    if CJK_RE.search("\n".join(model_facing)):
        issues.append("residual CJK in model-facing English questionbank fields")
    bad_short = [s for s in model_facing if s.strip() and BAD_SHORT_EN_RE.fullmatch(s.strip())]
    if bad_short:
        issues.append(f"truncated placeholder translation in English fields: {bad_short[:5]}")
    return issues


def sync_event(event: str, points: set[str] | None, args: argparse.Namespace) -> dict[str, Any]:
    cn_dir = RUNS / event / "final" / "中文" / "questionbank"
    en_dir = RUNS / event / "final" / "英文" / "questionbank"
    files = sorted(cn_dir.glob("P*_questionbank.json"))
    if points:
        files = [p for p in files if p.stem.replace("_questionbank", "") in points]
    strings: list[str] = []
    seen: set[str] = set()
    point_stats: dict[str, Any] = {}
    for path in files:
        qb = load_json(path)
        local = collect_strings(qb)
        point_stats[path.stem.replace("_questionbank", "")] = {
            "strings": len(local),
            "brier": len(qb.get("brier_questions", []) or []),
            "events_all": len(qb.get("events", {}).get("all", []) or []),
        }
        for s in local:
            if s not in seen:
                seen.add(s)
                strings.append(s)

    cache_path = RUNS / "_repair_reports" / "no_cache_translation" / f"{event}.json"
    tmap = translate_all(
        strings,
        cache_path,
        sleep_s=args.sleep,
        resume=args.resume,
        batch_size=args.batch_size,
        workers=args.workers,
        limit=args.limit,
    )
    updated: list[str] = []
    for cn_path in files:
        cn_qb = load_json(cn_path)
        en_qb = apply_translation(cn_qb, tmap)
        issues = validate(cn_qb, en_qb)
        if issues:
            raise RuntimeError(f"{event}/{cn_path.name}: {issues}")
        out = en_dir / cn_path.name
        if not args.dry_run:
            dump_json(out, en_qb)
        updated.append(str(out.relative_to(ROOT)))
    return {
        "event": event,
        "points": point_stats,
        "unique_strings": len(strings),
        "cache_path": str(cache_path.relative_to(ROOT)),
        "updated_files": updated,
        "dry_run": args.dry_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="all", help="comma-separated event dirs or all")
    ap.add_argument("--points", default="all", help="comma-separated PIDs or all")
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="debug: translate only N missing strings")
    args = ap.parse_args()
    events = sorted(p.name for p in RUNS.glob("event*")) if args.events == "all" else [
        e.strip() for e in args.events.split(",") if e.strip()
    ]
    points = None if args.points == "all" else {p.strip() for p in args.points.split(",") if p.strip()}
    report = {"events": []}
    for event in events:
        print(f"[sync] {event}", flush=True)
        report["events"].append(sync_event(event, points, args))
    report_dir = RUNS / "_repair_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "sync_english_qb_no_cache_report.json"
    dump_json(out, report)
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
