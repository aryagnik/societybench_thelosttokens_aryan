"""Unified black-box interface: (framework, base model, prompt) -> answer text (same format as bare models).

Two ways to obtain answers; run_agents picks one:
  1) agent_answer(...)         — old "stateless per-point" (legacy): each prediction point
     reruns the full multi-agent rollout from scratch. Kept as fallback / comparison baseline.
  2) make_session(...) -> Session — new "stateful per-event continuous rollout": one session per
     event; advance(delta, cutoff) feeds real new developments between two cutoffs in time order
     to advance the rollout state, then answer(point_prompt, kind) reads the answer for that point
     off the current state. Only 1 rollout per event overall.

Both paths only handle "how to get the raw answer text"; all downstream parsing/scoring reuses 3B/3F.
"""
from __future__ import annotations

import importlib
import pathlib
import sys
from threading import Lock
from typing import Optional, Protocol, runtime_checkable

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_FRAMEWORKS = {
    "langgraph": "fw_langgraph",   # plan then solve
    "autogen": "fw_autogen",       # multi-agent debate
    "mirofish": "fw_mirofish_oasis",  # OASIS-based social simulation (route B)
}


# ---------------------------------------------------------------------------
# Process-level call counters (diagnostic): prove "rollouts went from N per point down to 1 per event".
#   advance / answer : session advance / readout counts (framework-agnostic, tallied by run_agents)
#   llm              : underlying LLM request count (langgraph/autogen counted directly; mirofish inside subprocess)
#   sim              : social-simulation launch count (mirofish only)
#   episode          : legacy agent_answer call count (= old "full rollout" count)
# Under threads += is protected by the GIL, good enough for diagnostics; the precise
# accounting is run_agents' per-session stats.
# ---------------------------------------------------------------------------
class _Stats:
    __slots__ = ("advance", "answer", "llm", "sim", "episode", "_lock")

    def __init__(self) -> None:
        self.advance = self.answer = self.llm = self.sim = self.episode = 0
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self.advance = self.answer = self.llm = self.sim = self.episode = 0

    def bump(self, field: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def as_dict(self) -> dict:
        return {"advance": self.advance, "answer": self.answer,
                "llm": self.llm, "sim": self.sim, "episode": self.episode}


STATS = _Stats()


@runtime_checkable
class Session(Protocol):
    """A continuous rollout session for one event. run_agents relies only on these methods."""

    def advance(self, delta_context: str, cutoff: str) -> None:
        """Feed real new developments between the previous cutoff and this cutoff; advance rollout state."""
        ...

    def answer(self, point_prompt: str, kind: str = "brier") -> str:
        """Read this prediction point's answer off the current state; text format matches bare models (for B/F parsing)."""
        ...

    def close(self) -> None:
        """Release resources (subprocess/clients etc.). Frameworks without resources may no-op."""
        ...

    def stats(self) -> dict:
        """Call counters for this session, reported by run_agents."""
        ...


def _load_fw(framework: str):
    fw = framework.lower()
    if fw not in _FRAMEWORKS:
        raise ValueError(f"未知框架: {framework}(支持: {list(_FRAMEWORKS)})")
    try:
        return fw, importlib.import_module(_FRAMEWORKS[fw])
    except ModuleNotFoundError as e:
        if fw == "mirofish":
            raise SystemExit(
                "MiroFish(OASIS)尚未部署:camel-oasis 需 Python 3.10/3.11,"
                "当前系统为 3.12。见 agents/_install_oasis.log。LangGraph/AutoGen 不受影响。"
            ) from e
        raise


def agent_answer(
    framework: str,
    base_model: str,
    prompt: str,
    kind: str = "brier",
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
) -> str:
    """Legacy: stateless per-point. Reruns the framework's full rollout from scratch every time."""
    _fw, mod = _load_fw(framework)
    STATS.bump("episode")
    return mod.answer(base_model, prompt, kind, max_tokens=max_tokens, thinking_budget=thinking_budget)


def make_session(
    framework: str,
    base_model: str,
    *,
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
    sim_params: Optional[dict] = None,
) -> Session:
    """New: stateful per-event session. Dispatches to each framework module's make_session.

    sim_params: mirofish only (n_agents/n_steps etc.); ignored by other frameworks.
    """
    _fw, mod = _load_fw(framework)
    if not hasattr(mod, "make_session"):
        raise NotImplementedError(
            f"框架 {framework} 尚未实现 make_session(连续推演)。"
            "可用 --legacy-per-point 走旧的无状态路径。"
        )
    return mod.make_session(
        base_model,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        sim_params=sim_params,
    )
