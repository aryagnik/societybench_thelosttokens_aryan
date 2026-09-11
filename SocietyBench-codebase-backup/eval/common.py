"""SocietyBench — shared utilities for all eval/*.py scripts.

This module is **fully generic**: no event-specific constants, no hardcoded
entity names. It provides:

  - JSON / JSONL I/O helpers
  - Date parsing
  - Text normalization
  - A unified LLM client (`call_llm`) that reads `DMXAPI_KEY` + `DMXAPI_BASE_URL`
    from environment variables (see config.example.env).

All event-specific behavior (event name, replacement table, topic keywords,
etc.) is passed in at runtime via CLI arguments to the individual step scripts.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ===========================================================================
# Pipeline configuration
# ===========================================================================

def load_pipeline_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load generic pipeline_config.json (operational parameters).

    Defaults to `<this_file_dir>/pipeline_config.json`. Override with --pipeline-config.

    The config contains runtime parameters (model list, platforms, thresholds,
    scoring formula coefficients, etc.) — **no event-specific content**.
    """
    if path is None:
        path = Path(__file__).parent / "pipeline_config.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ===========================================================================
# JSON / JSONL I/O
# ===========================================================================

def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(path: Path, done_count: int, total: int,
                   completed_ids: Optional[List[str]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> None:
    """SKILL §"每完成一个步骤必须追加..." — write an explicit progress.json
    snapshot. Resume mechanism still uses output-file scanning (more robust);
    this file is for humans / audit only."""
    payload: Dict[str, Any] = {
        "done": done_count,
        "total": total,
        "pct": round(done_count / total * 100, 2) if total else 0.0,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if completed_ids is not None:
        payload["completed_ids"] = list(completed_ids)[-500:]  # cap at last 500 for size
    if extra:
        payload.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ===========================================================================
# Text normalization
# ===========================================================================

def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_markdown(text: str) -> str:
    s = normalize_space(text)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return normalize_space(s)


def normalize_text(text: Any) -> str:
    s = strip_markdown(str(text or ""))
    s = s.replace("　", " ").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ===========================================================================
# Date parsing
# ===========================================================================

def parse_date_obj(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    text = normalize_space(raw)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})[-/\.年](\d{1,2})[-/\.月](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_date_str(raw: Any) -> str:
    obj = parse_date_obj(raw)
    return obj.isoformat() if obj else ""


def date_neighbors(date_str: str, delta_days: int = 1) -> List[str]:
    d = parse_date_obj(date_str)
    if d is None:
        return []
    return [
        (d + timedelta(days=off)).isoformat()
        for off in range(-delta_days, delta_days + 1)
    ]


# ===========================================================================
# Anonymization helpers (generic — user supplies the replacement table)
# ===========================================================================

def load_replacements(path: Path) -> List[List[str]]:
    """Load a user-supplied entity replacement table.

    Accepts either a list of [original, replacement] pairs, or a dict with a
    `replacements` key holding the same.
    """
    p = Path(path)
    if not p.exists():
        return []
    data = load_json(p)
    if isinstance(data, dict):
        data = data.get("replacements", [])
    if not isinstance(data, list):
        return []
    out: List[List[str]] = []
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            out.append([str(item[0]), str(item[1])])
    # Longest match first
    out.sort(key=lambda r: len(r[0]), reverse=True)
    return out


def apply_replacements(text: str, replacements: List[List[str]]) -> str:
    output = text
    for source, target in replacements:
        if re.fullmatch(r"[A-Za-z0-9 .,'’&()/-]+", source):
            pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])")
            output = pat.sub(target, output)
            # Plain follow-up replace catches SEO-mangled fragments like
            # "OpinionTrump" → "OpinionLeaderA". Skip when source is a substring
            # of target (otherwise it would cascade and double up).
            if len(source) > 3 and source not in target:
                output = output.replace(source, target)
        else:
            output = output.replace(source, target)
    return output


# ===========================================================================
# LLM client (OpenAI-compatible chat completions, e.g. DMXAPI)
# ===========================================================================

class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    pass


def _quota_block_path() -> Path:
    configured = os.environ.get("SB_DMX_QUOTA_BLOCK_FILE", "").strip()
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[2]
    return root / "runs_new" / "logs_manual" / "dmxapi_quota_block.json"


def _quota_block_ttl_seconds() -> int:
    try:
        return max(0, int(os.environ.get("SB_DMX_QUOTA_BLOCK_TTL", "300")))
    except ValueError:
        return 300


def _quota_error_text(raw: str) -> bool:
    text = str(raw or "").lower()
    return any(
        marker in text
        for marker in (
            "insufficient_user_quota",
            "quota",
            "额度不足",
            "预扣费额度失败",
            "用户额度不足",
        )
    )


def _write_quota_block(model: str, detail: str) -> None:
    path = _quota_block_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "blocked_at": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "reason": "dmxapi_quota",
            "detail": str(detail)[-1000:],
            "ttl_seconds": _quota_block_ttl_seconds(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _active_quota_block() -> Optional[str]:
    path = _quota_block_path()
    if not path.exists():
        return None
    ttl = _quota_block_ttl_seconds()
    if ttl <= 0:
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return f"DMXAPI quota block active ({int(age)}s old): {data.get('detail', '')}"
    except Exception:
        return "DMXAPI quota block active"


def _resolve_endpoint() -> str:
    raw = os.environ.get(
        "DMXAPI_BASE_URL",
        "https://your-api-endpoint.example.com/v1/chat/completions",
    ).rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    return raw + "/chat/completions"


def _load_dmx_keys() -> List[str]:
    """加载 DMXAPI key。
    **2026-06-30 用户拍板:单 API + 300 线程。** 实验D 实测单 key 到 300 并发几乎无限流
    (1050 请求仅 1 次瞬时 429),故【只用第 1 个 key,key 2/3/4 闲置不用】。
    下面照旧解析 key 池(env DMXAPI_KEYS > secrets DMX_API_KEYS > env DMXAPI_KEY > secrets DMX_API_KEY),
    但**只返回第 1 个**。如需恢复多 key 轮询并行,把结尾的 `[:1]` 去掉即可。
    secrets 路径由 SB_SECRETS_FILE 指定,默认与实验方案 §0.2 同源。
    """
    pool: List[str] = []
    multi = os.environ.get("DMXAPI_KEYS", "").strip()
    if multi:
        pool = [k.strip() for k in multi.split(",") if k.strip()]
    if not pool:
        secrets_path = os.environ.get("SB_SECRETS_FILE", "")
        try:
            with open(secrets_path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("DMX_API_KEYS") or (
                [data["DMX_API_KEY"]] if data.get("DMX_API_KEY") else []
            )
            pool = [k.strip() for k in raw if k and str(k).strip()]
        except Exception:
            pool = []
    if not pool:
        single = os.environ.get("DMXAPI_KEY", "").strip()
        if single:
            pool = [single]
    return pool[:1]   # ★ 单 key:只用第 1 个;key 2/3/4 闲置(2026-06-30 用户定)


# (2026-07-01 删除 _openrouter_key:评测全走 DMX,不再用 OpenRouter——账户无 OpenRouter 余额)


def _wants_responses(model: Optional[str]) -> bool:
    """哪些模型走 OpenAI /responses 协议(实测思考比 /chat 多~2×)。2026-07-01 用户定:目前仅豆包(已实测可用);
    其余 OpenAI-兼容(gpt/deepseek/grok…)待逐个实测后再加;kimi/claude/gemini/glm 走各自原生 API,不适用。

    2026-07-01 用户定「豆包两种请求方式都准备好」:开关 env `SB_DOUBAO_ENDPOINT`:
      - 'responses'(默认,保持现有行为):走 /responses,思考翻倍;
      - 'chat':走 /chat/completions,更稳(小 max_tokens 不会空输出)。
    只影响豆包;其余模型不受此开关影响。"""
    if "doubao" not in (model or "").lower():
        return False
    return os.environ.get("SB_DOUBAO_ENDPOINT", "responses").strip().lower() != "chat"


def _extract_responses_text(payload: Dict[str, Any]) -> str:
    """从 /responses 返回里取最终答案文本(忽略 reasoning 思考块)。"""
    parts: List[str] = []
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
    text = "".join(parts) or (payload.get("output_text") or "")
    if not text:
        raise KeyError("responses: empty output text")
    return text


def call_llm(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retries: int = 3,
    timeout: int = 1200,  # 2026-07-02 用户定:600→1200s(重思考模型留足余量;非流式,超此判超时→归DMX类重试)
    reasoning_effort: Optional[str] = None,
    include_reasoning: bool = False,
    thinking_budget_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Send a chat-completions request and return the assistant message content.

    Env vars (set in your shell or via .env):
      DMXAPI_KEY            (required)
      DMXAPI_BASE_URL       (required; OpenAI-compatible endpoint)
      DMXAPI_DEFAULT_MODEL  (optional; defaults to "kimi-k2.5" if not set)

    For Step 3 evaluation (per predict-step3B/3F SKILL):
      reasoning_effort="high"      adds {"reasoning": {"effort": "high"}}
      include_reasoning=True       adds {"include_reasoning": true}
      thinking_budget_tokens=24000  adds {"thinking": {"type": "enabled", "budget_tokens": 24000}}
    Pass `extra_body` for any other OpenAI-compatible field (e.g. stream).
    """
    model = model or os.environ.get("DMXAPI_DEFAULT_MODEL", "kimi-k2.5")
    quota_block = _active_quota_block()
    if quota_block:
        raise LLMQuotaError(quota_block)

    # 2026-07-01 用户定:评测【全走 DMX】(账户无 OpenRouter 余额;claude 也用 DMX,不再走 OpenRouter)。
    #   已知代价:DMX 上 claude 自适应思考不透传(reasoning_tokens≈0);为统一通道接受此限制。
    is_anthropic = model.startswith("anthropic/") or "claude" in model.lower()
    keys = _load_dmx_keys()
    if not keys:
        raise LLMError("No DMXAPI key found (set DMXAPI_KEYS/DMXAPI_KEY env, or DMX_API_KEYS in secrets)")
    url = _resolve_endpoint()
    if is_anthropic:
        temperature = 1.0   # claude 扩展思考要求 temperature=1
    # 官方 Moonshot 的 kimi-k2.x 是思考模型,只接受 temperature=1(传 0 会 HTTP400)。
    elif model and model.lower().startswith("kimi-k2") and "moonshot" in url.lower():
        temperature = 1.0

    # 豆包走 /responses 协议(2026-07-01 用户定):实测思考比 /chat 多~2×;闭卷不联网,纯为多挖思考。
    use_responses = _wants_responses(model)
    if use_responses:
        url = url.rsplit("/chat/completions", 1)[0].rstrip("/") + "/responses"
        payload_obj: Dict[str, Any] = {
            "model": model,
            "input": messages,            # Responses 协议:input 取代 messages
            "max_output_tokens": max_tokens,
        }
        if reasoning_effort:
            payload_obj["reasoning"] = {"effort": reasoning_effort}   # 思考强度;不收 temperature/thinking budget
        if extra_body:
            payload_obj.update(extra_body)
    else:
        payload_obj = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            payload_obj["reasoning"] = {"effort": reasoning_effort}
        if include_reasoning:
            payload_obj["include_reasoning"] = True
        if thinking_budget_tokens is not None and not is_anthropic:
            # Anthropic opus-4-8 已移除 budget_tokens(传了 400),思考由上面的 reasoning.effort 控;故跳过。
            payload_obj["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_budget_tokens),
            }
        if extra_body:
            payload_obj.update(extra_body)

    body = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    # 多 key 轮询:每次调用随机打乱 key 顺序,重试时依次换 key(既分散限流又容错单 key 额度耗尽)
    import random
    key_order = random.sample(keys, len(keys)) if len(keys) > 1 else list(keys)
    n_attempts = max(retries, len(key_order))

    last_err: Optional[Exception] = None
    quota_hits = 0
    for attempt in range(n_attempts):
        key = key_order[attempt % len(key_order)]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            if use_responses:
                return _extract_responses_text(payload)
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            detail = f"HTTP Error {e.code}: {e.reason} {body_text}".strip()
            last_err = LLMError(detail)
            if e.code == 403 and _quota_error_text(detail):
                quota_hits += 1
                # 仅当所有 key 都额度耗尽,才写全局 quota block 并抛出;否则换下一个 key 继续
                if quota_hits >= len(key_order):
                    _write_quota_block(model, detail)
                    raise LLMQuotaError(detail) from e
                continue
            if attempt < n_attempts - 1:
                time.sleep(2 ** min(attempt, 4))
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < n_attempts - 1:
                time.sleep(2 ** min(attempt, 4))
    raise LLMError(f"LLM call failed after {n_attempts} attempts: {last_err}")


def call_llm_json(prompt: str, *, model: Optional[str] = None, temperature: float = 0.0) -> Any:
    """Call LLM with a single user message and parse the response as JSON.

    The prompt should instruct the model to return JSON only.
    """
    raw = call_llm([{"role": "user", "content": prompt}], model=model, temperature=temperature)
    # Strip code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ===========================================================================
# Convenience: ID generation for timeline nodes
# ===========================================================================

def make_node_id(date_str: str, index: int) -> str:
    """Stable ID for a timeline node: `<YYYY-MM-DD>_<index>`."""
    d = parse_date_str(date_str) or "0000-00-00"
    return f"{d}_{index}"
