#!/usr/bin/env python3
"""predict-fullanon-reaudit: generic "full anonymization + strong-detector multi-round re-audit" deep re-wash.

== What this is ==
Re-runs step0's closed loop of "anonymize -> multi-round audit -> derive new rules ->
re-anonymize -> re-check", with two upgrades:
  1) **Stronger detector**: gpt-5.5 + reasoning enabled + a strengthened prompt, to recall
     **long-tail** leaks — compound/derived forms (美帝/反美/美式/中美沙), homophones/aliases
     (超微 <-> 超威), acronyms/puns (SMCI = So Much Criminal Investigation), transliterated
     roman names, specific model numbers / proper names (055型驱逐舰/B-52/a named brokerage),
     descriptive identification (fifth-largest -> a specific company), cultural allusions
     (恒河 -> a country, 六芒星 -> a country). These are missed by the deterministic table and
     weaker models (kimi), and only caught by strong-model reasoning.
  2) **Scope changed to full-anon**: even **named public terms** (countries / named regulators /
     named financial-report forms / currencies / regions / indices) get anonymized; only
     numbers/amounts/percentages/dates, placeholders like 公司A/国家X, and **generic common
     nouns** that do not point to a specific party (government/court/president/analyst/
     central bank) are kept.

== Genericity ==
**No entity names are hardcoded**: LLM detection finds residuals -> `derive_replacement_rules`
produces "type + code" placeholders -> appended to the same replacement table -> applied
deterministically to **all GT files (timeline+contexts+gt) and the question bank**
(placeholders consistent throughout) -> re-detect, iterating to convergence. Works for all
events. Dates were already shifted in step0 and numbers are prediction signals — **never touched**.

== Usage ==
    python predict_fullanon_reaudit.py --workspace <anon_dir> \
        [--model gpt-5.5] [--reasoning-effort medium] [--max-rounds 4]
Usually copy the existing anon directory to a **new timestamped folder** first, then run this
script on the new folder (old artifacts untouched).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from predict_step0_prepare import derive_replacement_rules, _expand_variants
from predict_step2_anonymize_qb import (
    build_replacer, apply_table, _extract_json,
    placeholder_regex, gt_placeholder_set, prune_orphan_questions)
from common import call_llm, load_replacements


FULLANON_PROMPT = """你是**极其严格**的匿名化审计员。下面是一份文本，真实实体本应已被替换为占位符
（公司A/机构B/国家X/人物A 等）。本任务目标是**完全匿名**——**任何能让读者定位到现实中某个
特定主体的专有名词都要揪出来**，包括：

【常规具名专名】真实的 国家 / 地区 / 城市；具名的政府部门 / 监管机构 / 国际组织；公司 / 品牌 /
产品；人名；媒体；具名法规 / 财报表格 / 指数 / 货币（如 某国、某证监会、某10-K表、某货币、某指数）。
【容易漏的长尾——逐条推理排查】
  1. **复合 / 派生 / 简称**：实体匿了但其复合派生写法没匿（「美国」匿了漏「美帝/反美/美式/中美沙」）。
  2. **谐音 / 别名 / 旧名**：正名匿了，谐音或别称没匿（「超微」漏其谐音「超威」）。
  3. **缩写 / 梗 / 字母游戏**：能拼出真实名称或股票代码的缩写、玩笑式全称、圈内黑话。
  4. **音译 / 罗马名**：看起来像真实人名的拼音 / 英文名。
  5. **具体型号 / 专名**：具体武器 / 产品 / 型号 / 机构 / 媒体 / 公司 / 律所名，**哪怕只出现一两次**。
  6. **描述性指认**：名字虽匿，但**描述性属性**能唯一锁定真实主体（「全球第五大、11.5万员工的会计所」）。
  7. **文化 / 国别影射**：用文化符号影射真实国家 / 民族（恒河→某国、六芒星→某国、波斯→某国）。

🔒 **不算残留（保留）**：纯数字 / 金额 / 百分比 / 日期（**注意：金额里的"数字"保留，但"货币名称"
如 美元 / 欧元 / 人民币 仍是具名实体、要算残留**，如「400美元」要报"美元"）；形如「公司A / 机构B /
国家X / 人物A」的占位符本身；**泛指常名**——不指向某个具体真实主体的普通名词 / 角色（政府、法院、
总统、分析师、央行、监管机构[泛指]、公司[泛指]、记者、议员、首都）。

请**逐个候选先在心里推理**「它是否真能让人定位到现实中某个特定主体」，再**只输出 JSON**（无多余文字）：
{{"clean": <true|false>, "residual_real_names": ["残留串", ...], "note": "一句话"}}

待审计文本：
{sample}
"""

# Placeholders (type + code, e.g. 国家X/公司A/审计机构B/货币A) — the detector may
# false-positive on these; filter them out instead of sending to derive.
PLACEHOLDER_RE = re.compile(r"^[一-鿿]{1,10}[A-Z]\d{0,3}$")
# Pure numbers/amounts/percentages/dates — always kept by rule; filter out.
NUMERIC_RE = re.compile(r"^[\d\s.,%+\-—~:：/年月日时分秒万亿元美分$€¥]+$")


def verify_fullanon(samples: List[str], model: str, fallback: str, effort: str) -> Dict[str, Any]:
    """Strong detection (full-anon prompt); main model failure/refusal -> fallback (no reasoning)."""
    msg = [{"role": "user", "content": FULLANON_PROMPT.format(sample="\n".join(samples)[:14000])}]
    last = ""
    for m, eff in [(model, effort), (fallback, None)]:
        if not m:
            continue
        try:
            kw = {"reasoning_effort": eff} if eff else {}
            raw = call_llm(msg, model=m, temperature=0.0, max_tokens=8000, timeout=180, **kw)
            return _extract_json(raw)
        except Exception as e:
            last = str(e)[:120]
            continue
    return {"clean": None, "_error": last}


def detect_unique(blobs: List[str], model: str, fallback: str, effort: str,
                  char_budget: int = 2500) -> List[str]:
    """Run full-coverage detection on **deduplicated unique text lines** (the GT's contexts/gt
    replicate the timeline per point and are highly redundant; after dedup only the master-copy
    volume remains -> saves time without losing variety). Returns deduped residual candidates."""
    seen, lines = set(), []
    for blob in blobs:
        for ln in blob.splitlines():
            ln = ln.strip()
            if len(ln) >= 4 and ln not in seen:
                seen.add(ln)
                lines.append(ln)
    chunks: List[List[str]] = []
    cur, n = [], 0
    for ln in lines:
        if cur and n + len(ln) > char_budget:
            chunks.append(cur); cur, n = [], 0
        cur.append(ln); n += len(ln) + 1
    if cur:
        chunks.append(cur)
    out, ks = [], set()
    for i, ch in enumerate(chunks):
        v = verify_fullanon(ch, model, fallback, effort)
        for nm in (v.get("residual_real_names") or []):
            nm = str(nm).strip()
            if nm and nm.lower() not in ks:
                ks.add(nm.lower()); out.append(nm)
    return out


def is_noise(s: str) -> bool:
    s = s.strip()
    return (not s) or bool(PLACEHOLDER_RE.match(s)) or bool(NUMERIC_RE.match(s))


def gt_files(ws: Path) -> List[Path]:
    fs = [ws / "timeline_anon.md"]
    for sub in ("contexts", "gt"):
        d = ws / sub
        if d.is_dir():
            fs += sorted(p for p in d.iterdir() if p.is_file())
    return [f for f in fs if f.is_file()]


def apply_text(files: List[Path], repl) -> int:
    n = 0
    for f in files:
        t = f.read_text(encoding="utf-8")
        t2 = repl(t)
        if t2 != t:
            n += 1
        f.write_text(t2, encoding="utf-8")
    return n


def qb_corpus(qbs: List[Path]) -> List[str]:
    out = []
    for f in qbs:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for q in d.get("brier_questions", []):
            s = str(q.get("q", "")).strip()
            if s:
                out.append(s)
    return out


FULLANON_DERIVE_PROMPT = """下面是从一份「目标=完全匿名」的文本里检出的「疑似可识别信息」清单。
请为其中**确实是能定位现实特定主体的专有名词**的项，各分配一个「类型+代号」占位符。

⚠️ 这是**完全匿名**任务：具名的 **国家 / 地区 / 海峡 / 城市、监管机构 / 政府部门 / 国际组织、
政党、公司 / 品牌 / 产品、人名、媒体、具名法规 / 财报表格 / 指数 / 货币、具体型号 / 案号**——
**都要分配占位符**，**不要因为"是公共词 / 常见 / 通用"就跳过**（本任务就是要把它们也匿掉）。

代号规则（类型 + 字母代号）：国家→国家X/Y/Z；地区→地区A；海峡→海峡A；城市→城市A；
机构/部门/监管机构/组织→机构A；政党→政党A；公司→公司A；媒体→媒体A；人物→人物A/领导人A；
法规/条款→法规A；财报表格→表格A；货币→货币A；指数→指数A；产品/型号→型号A；案号/编号→编号A；
**缩写 / 梗 / 字母游戏（能拼出真实名称或股票代码的玩笑式全称、黑话）→ 梗A/别称A** …
- **同类不同实体用不同代号**；**同一实体（含简称/变体/谐音/子串）复用同一代号**；
- 占位符不含原文真名、不含连续英文、不含多位数字。

**只跳过这些（不分配、不输出）**：纯数字 / 金额 / 百分比 / 日期；已是占位符（国家X/公司A…）；
**纯普通名词 / 角色**（不指向某个具体真实主体，如 政府、法院、总统、分析师、记者、内衣、年轻人、华人）；
**纯描述性句子**（**且不含可还原的真名或代码**——若一句话里嵌着能拼出真名/代码的梗，仍要把那个梗匿掉）。

已有映射（真名 → 已分配代号）。给新实体避开已用代号、对变体/简称复用同一代号：
{existing_map}

只输出 JSON 数组（无额外文字）：[["真实专名","占位符"], ...]；跳过的不要出现。

清单：
{names_block}
"""


def derive_fullanon(cands: List[str], model: str, existing_rules) -> List[List[str]]:
    """full-anon variant of derive: like derive_replacement_rules it produces "type + code"
    placeholders and passes in the current table for consistency, but it does **not skip named
    public terms** (countries/regulators/report forms/currencies/regions... all anonymized) —
    only pure common nouns / sentences are skipped."""
    names, seen = [], set()
    for c in cands:
        s = str(c).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower()); names.append(s[:80])
    if not names:
        return []
    pairs = [p for p in (existing_rules or [])
             if isinstance(p, (list, tuple)) and len(p) >= 2 and str(p[1]).strip()]
    existing_map = "\n".join(f"- {str(p[0]).strip()} → {str(p[1]).strip()}" for p in pairs[-120:]) or "（暂无）"
    prompt = FULLANON_DERIVE_PROMPT.format(
        names_block="\n".join(f"- {n}" for n in names), existing_map=existing_map)
    try:
        # gpt-5.5 is a reasoning model; with many candidates the output token need is large.
        # Give ample max_tokens, else reasoning eats the budget and content comes back empty.
        raw = call_llm([{"role": "user", "content": prompt}], model=model,
                       temperature=0.0, max_tokens=16000, timeout=300)
    except Exception:
        return []
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out = []
    for it in data:
        if not (isinstance(it, list) and len(it) == 2):
            continue
        bad, good = str(it[0]).strip(), str(it[1]).strip()
        if not (bad and good) or bad == good:
            continue
        if len(good) > max(10, len(bad)):
            continue
        if re.search(r"[A-Za-z]{2,}|\d{2,}", good):
            continue
        if bad.lower() in good.lower():
            continue
        out.append([bad, good])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Full anonymization + strong-detector multi-round re-audit (generic, works for all events)")
    p.add_argument("--workspace", required=True, help="anon workspace directory (recommended: copy to a new timestamped folder first)")
    p.add_argument("--replacements", default=None, help="replacement table (default <ws>/replacements_effective.json)")
    p.add_argument("--model", default="gpt-5.5", help="strong detection model (default gpt-5.5)")
    p.add_argument("--fallback", default="gpt-5.2", help="fallback model (when the main model refuses/fails)")
    p.add_argument("--reasoning-effort", default="medium", help="reasoning effort low/medium/high (default medium)")
    p.add_argument("--max-rounds", type=int, default=4, help="max rounds of detect -> derive -> re-anonymize")
    p.add_argument("--char-budget", type=int, default=2500, help="chars per chunk (gpt-5.5 reasoning is super-linearly sensitive to length; small chunks are faster)")
    a = p.parse_args()

    ws = Path(a.workspace)
    tbl_path = Path(a.replacements) if a.replacements else ws / "replacements_effective.json"
    replacements = load_replacements(tbl_path)
    repl = build_replacer(replacements)
    gfs = gt_files(ws)
    qbs = sorted((ws / "questionbank").glob("P*.json")) if (ws / "questionbank").is_dir() else []
    print(f"[init] GT 文件 {len(gfs)} 个，题库 {len(qbs)} 个，表 {len(replacements)} 条规则")

    # 0) First re-apply the existing table deterministically (rolls out full-anon rules
    #    already in the table; idempotent)
    apply_text(gfs, repl)
    if qbs:
        apply_table(qbs, repl)

    have = {str(r[0]) for r in replacements if isinstance(r, (list, tuple)) and r}
    report: Dict[str, Any] = {"model": a.model, "effort": a.reasoning_effort, "rounds": []}
    added: List[List[str]] = []

    for rnd in range(1, a.max_rounds + 1):
        # Detect on **GT only** (~20K chars after master-copy dedup) — the question bank can be
        # thousands of questions / hundreds of thousands of chars and would stall detection,
        # and the bank's own knowledge injection is handled by step2.5 (gpt-5.5); rules derived
        # here are still applied to the bank.
        blobs = [f.read_text(encoding="utf-8") for f in gfs]
        resid = detect_unique(blobs, a.model, a.fallback, a.reasoning_effort, char_budget=a.char_budget)
        cands = [c for c in resid if not is_noise(c) and c not in have]
        print(f"[round {rnd}] 检出残留 {len(resid)}，去噪/去已知后待匿 {len(cands)}：{cands[:15]}")
        report["rounds"].append({"round": rnd, "residual": resid, "to_anon": cands})
        if not cands:
            print("  ✅ 无新增可匿专名，收敛"); break
        new_rules = derive_fullanon(cands, model=a.model, existing_rules=replacements)
        new_rules = [r for r in new_rules if r and str(r[0]) not in have]
        if not new_rules:
            print("  derive 全判为非专名/泛指，停"); break
        print(f"  derive 生成 {len(new_rules)} 条新规则：{new_rules[:8]}")
        replacements += new_rules
        added += new_rules
        have |= {str(r[0]) for r in new_rules}
        repl = build_replacer(replacements)
        apply_text(gfs, repl)
        if qbs:
            apply_table(qbs, repl)

    # Write the table back + deterministic correctness check (all in-table real-name LHS
    # should be washed out; count should be 0)
    if added:
        tbl_path.write_text(json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")
    lhs = sorted({str(o) for o, _ in _expand_variants(replacements) if o}, key=len, reverse=True)
    blob = "".join(f.read_text(encoding="utf-8") for f in gfs) + "".join(
        f.read_text(encoding="utf-8") for f in qbs)
    det = {o: blob.count(o) for o in lhs if o and o in blob}
    report["deterministic_residual"] = det

    # Off-topic question gate: question placeholders must be a subset of GT placeholders (per
    # the mapping produced by GT anonymization; drop questions about entities the GT lacks)
    if qbs:
        ph_re = placeholder_regex(replacements)
        gt_ph = gt_placeholder_set("".join(f.read_text(encoding="utf-8") for f in gfs), ph_re)
        prune = prune_orphan_questions(qbs, gt_ph, ph_re)
        report["orphan_prune"] = prune
        print(f"[离题题门禁] 剔除引用 GT 不存在实体的题 {prune['removed']}，保留 {prune['kept']}")

    report["added_rules"] = added
    (ws / "fullanon_reaudit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[确定性检查] 表内真名残留 {len(det)} 种" + (f"：{dict(list(det.items())[:8])}" if det else " ✅ 0"))
    print(json.dumps({
        "gt_files": len(gfs), "questionbanks": len(qbs),
        "rounds": len(report["rounds"]), "added_rules": len(added),
        "deterministic_residual_kinds": len(det),
        "report": str(ws / "fullanon_reaudit_report.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
