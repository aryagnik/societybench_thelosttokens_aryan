"""Shared base-model access: DMXAPI (used by all three frameworks).

- LangGraph nodes call call_llm from eval/common.py directly — the **exact same** call path as bare models.
- AutoGen needs a ChatCompletionClient; here we point an OpenAIChatCompletionClient at DMXAPI.

Credentials are read from os.environ (shell has the real DMXAPI_KEY / DMXAPI_BASE_URL);
if missing, eval/.env is tried (placeholders ignored). Example official Chinese base model ids:
  Doubao = doubao-seed-2-0-pro-260215
  Qwen   = qwen3.5-plus-2026-02-15
"""
from __future__ import annotations

import os
import pathlib
import sys

# Make eval/common.py importable from this module
_EVAL = pathlib.Path(__file__).resolve().parent.parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))


def _maybe_load_dotenv() -> None:
    """Only when env vars are missing and .env holds real values (not example placeholders), fill them into os.environ."""
    if os.environ.get("DMXAPI_KEY", "").strip():
        return
    envf = _EVAL / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if not v or "example.com" in v or "your-" in v:
            continue
        os.environ.setdefault(k, v)


def require_dmxapi() -> tuple[str, str]:
    _maybe_load_dotenv()
    key = os.environ.get("DMXAPI_KEY", "").strip()
    if not key:
        raise SystemExit(
            "DMXAPI_KEY 未设置。请在 shell 导出 DMXAPI_KEY / DMXAPI_BASE_URL "
            "(或填好 code/eval/.env)后再跑智能体评测。"
        )
    base = os.environ.get("DMXAPI_BASE_URL", "https://www.dmxapi.com/v1").strip()
    return key, base


def openai_base_url() -> str:
    """The openai SDK needs the .../v1 form (it appends /chat/completions itself)."""
    _, base = require_dmxapi()
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def make_autogen_client(model: str, temperature: float = 0.0, max_tokens: int = 32000):
    """OpenAI-compatible client for AutoGen, pointed at DMXAPI.

    Non-OpenAI models must explicitly provide model_info (otherwise AutoGen refuses).
    """
    from autogen_core.models import ModelFamily, ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    key, _ = require_dmxapi()
    return OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url(),
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=float(os.environ.get("AGENT_AUTOGEN_TIMEOUT", "600")),
        reasoning_effort="high",
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family=ModelFamily.UNKNOWN,
            structured_output=False,
        ),
    )
