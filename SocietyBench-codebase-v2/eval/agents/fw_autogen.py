"""AutoGen — multi-agent debate (initial estimate -> calibration review -> final decision).

For fairness the initial estimate goes through the **exact same** high-reasoning path as bare
models (common.call_llm), then AutoGen's "reviewer -> final arbiter" two-agent debate does
calibration: the reviewer only challenges over/under-confident questions, and the arbiter
copies the initial estimates and only changes the challenged questions.
So AutoGen's starting point = bare-model quality, and the debate is incremental on top
(preventing the initial estimate from being weakened by the framework).
Base models go through DMXAPI (see llm_client.make_autogen_client).

Public API: answer(base_model, prompt, kind="brier") -> str
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import pathlib
import sys
import threading

_HERE = pathlib.Path(__file__).resolve().parent
_EVAL = _HERE.parent
for _p in (str(_HERE), str(_EVAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adapter  # noqa: E402  (process-level call counter STATS)
from common import call_llm  # noqa: E402
from llm_client import make_autogen_client  # noqa: E402

_SKEPTIC_SYS = (
    "你是校准审查员。下面给了预测员对一批题的概率初判。**只挑出**可能过度自信"
    "(给了极端值却证据不足)或过度保守(明明有迹象却给接近 50%)的题,逐条给『题号 + 具体理由』。"
    "不要自己重报全部数字,也不要把所有题拉向中间——只针对确有问题的题质疑。"
)
_DECIDER_SYS = (
    "你是终审。**直接照抄预测员初判的每一题概率作为最终值**,仅对校准审查员明确质疑、"
    "且你认同其理由的少数题做调整;其余题**原样保留初判数字,不要重新评估、不要整体拉向中间**。"
    "严格按任务要求格式逐题一行,只输出最终答案行,不要解释、不要复述讨论。"
)


async def _debate(base_model: str, prompt: str, draft: str, max_tokens: int) -> str:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    client = make_autogen_client(base_model, max_tokens=max_tokens)
    try:
        en = _is_en()
        skeptic = AssistantAgent("Skeptic", model_client=client,
                                 system_message=_SKEPTIC_SYS_EN if en else _SKEPTIC_SYS)
        decider = AssistantAgent("Decider", model_client=client,
                                 system_message=_DECIDER_SYS_EN if en else _DECIDER_SYS)
        team = RoundRobinGroupChat(
            [skeptic, decider],
            termination_condition=MaxMessageTermination(max_messages=3),
        )
        task = (prompt + ("\n\n======== Forecaster's initial estimates (per-question probability, "
                          "as baseline) ========\n" if en
                          else "\n\n======== 预测员初判(逐题概率,作为基准)========\n")
                + draft + "\n========")
        timeout = float(os.environ.get("AGENT_AUTOGEN_TIMEOUT", "1200"))
        res = await asyncio.wait_for(team.run(task=task), timeout=timeout)
        for m in reversed(res.messages):
            if getattr(m, "source", None) == "Decider":
                c = getattr(m, "content", None)
                if c:
                    return c if isinstance(c, str) else str(c)
        return draft  # fallback: debate produced no valid final decision, keep the draft (at least not below bare model)
    finally:
        await client.close()


def answer(
    base_model: str,
    prompt: str,
    kind: str = "brier",
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
) -> str:
    timeout = float(os.environ.get("AGENT_AUTOGEN_TIMEOUT", "1200"))
    # Initial estimate: same high-reasoning single shot as bare models (keeps AutoGen's starting point from being weakened).
    draft = call_llm(
        [{"role": "user", "content": prompt}],
        model=base_model, temperature=0.0, max_tokens=max_tokens,
        reasoning_effort="high", include_reasoning=True,
        thinking_budget_tokens=thinking_budget,
        retries=1,
        timeout=timeout,
    )
    try:
        return asyncio.run(_debate(base_model, prompt, draft, max_tokens))
    except Exception as e:
        print(f"[autogen] debate failed, fallback to draft: {type(e).__name__}: {e}", file=sys.stderr)
        return draft


# ===========================================================================
# Continuous rollout session (M3 single-rollout refactor)
# ===========================================================================
# Holds a persistent client and a rolling decided_belief (the debate consensus / situation assessment).
# advance: treat delta as new intelligence, skeptic->decider debate updates the consensus (2 messages);
# answer : decider answers the point's questions on the consensus (1 message). Client close is lifted
# to the session's close.
#
# Key engineering point: the autogen client is async and its underlying connections bind to the
# event loop of the first request; multiple asyncio.run() calls would switch loops and invalidate
# the client. So the session starts a **dedicated event-loop thread**; the client lives on that
# loop for its whole lifetime, and advance/answer/close submit coroutines to it.

_SKEPTIC_UPDATE_SYS = (
    "你是情报审查员。决策者要把【新进展】整合进【既有态势研判】。请只挑出整合中的问题:"
    "新进展里被忽略/被低估的关键事实,或既有研判中已被新进展推翻却仍保留的判断,逐条给具体理由。"
    "不要自己重写整篇研判,只针对确有问题处质疑。"
)
_DECIDER_UPDATE_SYS = (
    "你是首席决策者。把【新进展】整合进【既有态势研判】,产出**更新后的完整态势研判**:"
    "保留仍有效的旧信息、纳入新事实、修正被推翻的旧判断、采纳审查员合理的质疑;"
    "完整保留关键事实/日期/数字/人物/已发生节点,并更新对发展方向与不确定性的研判。"
    "只输出更新后的态势研判正文,不要复述讨论过程。"
)
_DECIDER_ANSWER_SYS = (
    "你是首席决策者。下面给出你团队对该事件的【当前态势研判】与一道预测任务。"
    "请**完全基于该研判**完成任务,严格按任务要求的格式逐条作答,只输出答案行,不要解释、不要复述研判。"
)
_ADVANCE_TASK = (
    "【既有态势研判】(截至上一时点)\n===== \n{belief}\n=====\n\n"
    "【新进展】(时间推进到 {cutoff})\n=====\n{delta}\n=====\n\n"
    "请审查员先质疑、决策者再产出更新后的完整态势研判。"
)
_READ_PREAMBLE = (
    "【当前态势研判】(截至 {cutoff},你团队持续推演形成)\n=====\n{belief}\n=====\n\n"
    "请完全基于上述研判完成下面的预测任务,严格按其要求的格式作答。\n\n"
)

# ===== English version (English workspaces, triggered by env SB_AGENT_LANG=en, consistent with the main experiment's English setup) =====
_SKEPTIC_SYS_EN = (
    "You are a calibration reviewer. Below are a forecaster's initial probability estimates for a "
    "batch of questions. **Only pick out** questions that may be overconfident (an extreme value "
    "with weak evidence) or overly conservative (near 50% despite clear signs), giving "
    "'question number + specific reason' for each. Do not restate all the numbers yourself, and "
    "do not pull all questions toward the middle — only challenge questions with genuine problems."
)
_DECIDER_SYS_EN = (
    "You are the final arbiter. **Copy the forecaster's initial probability for every question as "
    "the final value**, adjusting only the few questions the calibration reviewer explicitly "
    "challenged and whose reasoning you accept; keep every other question's initial number "
    "**as-is — do not re-evaluate or pull toward the middle**. Answer strictly in the required "
    "format, one line per question, output only the final answer lines, no explanation, no "
    "restating the discussion."
)
_SKEPTIC_UPDATE_SYS_EN = (
    "You are an intelligence reviewer. The decider is integrating the [new developments] into the "
    "[existing situation assessment]. Only point out problems in the integration: key facts in the "
    "new developments that are ignored/underweighted, or judgments in the existing assessment "
    "already overturned by the new developments yet still kept — give a specific reason for each. "
    "Do not rewrite the whole assessment yourself; only challenge where there are genuine problems."
)
_DECIDER_UPDATE_SYS_EN = (
    "You are the chief decider. Integrate the [new developments] into the [existing situation "
    "assessment] and produce the **updated, complete situation assessment**: keep still-valid old "
    "information, incorporate new facts, correct overturned old judgments, and adopt the reviewer's "
    "reasonable challenges; fully preserve key facts/dates/numbers/people/occurred milestones, and "
    "update the assessment of direction and uncertainty. Output only the updated situation-"
    "assessment text, do not recount the discussion."
)
_DECIDER_ANSWER_SYS_EN = (
    "You are the chief decider. Below is your team's [current situation assessment] of the event "
    "and one prediction task. Complete the task **entirely based on that assessment**, answering "
    "each item strictly in the required format, output only the answer lines, no explanation, no "
    "restating the assessment."
)
_ADVANCE_TASK_EN = (
    "[Existing situation assessment] (as of the previous time point)\n=====\n{belief}\n=====\n\n"
    "[New developments] (time advances to {cutoff})\n=====\n{delta}\n=====\n\n"
    "The reviewer challenges first, then the decider produces the updated complete situation assessment."
)
_READ_PREAMBLE_EN = (
    "[Current situation assessment] (as of {cutoff}, formed by your team's continuous reasoning)\n"
    "=====\n{belief}\n=====\n\n"
    "Complete the prediction task below entirely based on the above assessment, answering strictly "
    "in the required format.\n\n"
)


def _is_en() -> bool:
    return os.environ.get("SB_AGENT_LANG", "zh").strip().lower() == "en"


class _LoopThread:
    """A resident event-loop thread: keeps the async client on the same loop for the whole session lifetime."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # 2026-07-03 hang fix: .result() previously had no timeout — upstream gateway flaps
        # (nginx error pages) could leave autogen's internal coroutine hung forever, with the
        # advance/answer thread waiting indefinitely (confirmed via py-spy; both "silent hangs"
        # had this cause). Add a hard timeout: hang -> TimeoutError -> the existing retry/persist
        # machinery upstream takes over; cancel the coroutine after timeout to avoid leaks.
        # Cap = AGENT_AUTOGEN_TIMEOUT (default 1200) + 120s buffer.
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout=float(os.environ.get("AGENT_AUTOGEN_TIMEOUT", "1200")) + 120)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError("autogen coroutine hard-timeout (hung upstream call)")

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        try:
            self.loop.close()
        except Exception:  # noqa: BLE001
            pass


class _AutoGenSession:
    """AutoGen continuous rollout session: persistent client + rolling decided_belief (debate consensus)."""

    def __init__(self, base_model: str, *, max_tokens: int, thinking_budget: int) -> None:
        self.base = base_model
        self.max_tokens = int(max_tokens)
        self.thinking_budget = int(thinking_budget)  # autogen client doesn't take thinking directly; kept for parity
        self.belief = ""
        self.cutoff = ""
        self.timeout = float(os.environ.get("AGENT_AUTOGEN_TIMEOUT", "1200"))
        self.n_advance = 0
        self.n_answer = 0
        self.n_llm = 0
        self._loop = _LoopThread()
        self._client = self._loop.run(self._mk_client())

    async def _mk_client(self):
        return make_autogen_client(self.base, max_tokens=self.max_tokens)

    @staticmethod
    def _last_decider(res, fallback: str) -> str:
        for m in reversed(res.messages):
            if getattr(m, "source", None) == "Decider":
                c = getattr(m, "content", None)
                if c:
                    return c if isinstance(c, str) else str(c)
        return fallback

    @staticmethod
    def _count_model_msgs(res) -> int:
        return sum(1 for m in res.messages
                   if getattr(m, "source", None) in ("Skeptic", "Decider"))

    def _add_llm(self, n: int) -> None:
        self.n_llm += n
        adapter.STATS.bump("llm", n)

    # ---- advance: debate-style consensus update ----
    async def _advance_async(self, delta: str) -> str:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        en = _is_en()
        skeptic = AssistantAgent("Skeptic", model_client=self._client,
                                 system_message=_SKEPTIC_UPDATE_SYS_EN if en else _SKEPTIC_UPDATE_SYS)
        decider = AssistantAgent("Decider", model_client=self._client,
                                 system_message=_DECIDER_UPDATE_SYS_EN if en else _DECIDER_UPDATE_SYS)
        team = RoundRobinGroupChat([skeptic, decider],
                                   termination_condition=MaxMessageTermination(max_messages=3))
        task = (_ADVANCE_TASK_EN if en else _ADVANCE_TASK).format(
            belief=self.belief or ("(first point, no prior assessment)" if en else "（首次,暂无既有研判）"),
            cutoff=self.cutoff, delta=delta)
        res = await asyncio.wait_for(team.run(task=task), timeout=self.timeout)
        self._add_llm(self._count_model_msgs(res))
        return self._last_decider(res, self.belief)

    def advance(self, delta_context: str, cutoff: str) -> None:
        self.cutoff = cutoff or self.cutoff
        delta = (delta_context or "").strip()
        if not delta:
            return
        self.n_advance += 1
        adapter.STATS.bump("advance")
        try:
            self.belief = (self._loop.run(self._advance_async(delta)) or self.belief).strip()
        except Exception as e:  # noqa: BLE001
            # Debate-failure fallback: fold delta into belief with a single call (keep progress, don't crash).
            print(f"[autogen] advance debate failed ({type(e).__name__}: {e}), fold via single call",
                  file=sys.stderr)
            self._add_llm(1)
            en = _is_en()
            _fold = (_ADVANCE_TASK_EN if en else _ADVANCE_TASK).format(
                belief=self.belief or ("(none yet)" if en else "（暂无）"), cutoff=self.cutoff, delta=delta)
            _fold += ("\n\nOutput the updated, complete situation-assessment text directly."
                      if en else "\n\n直接输出更新后的完整态势研判正文。")
            self.belief = call_llm(
                [{"role": "user", "content": _fold}],
                model=self.base, temperature=0.0, max_tokens=self.max_tokens,
                reasoning_effort="high", include_reasoning=True,
                thinking_budget_tokens=self.thinking_budget, retries=1, timeout=self.timeout,
            ).strip()

    # ---- answer: decider reads the point's answer off the consensus ----
    async def _answer_async(self, point_prompt: str, temperature: float) -> str:
        from autogen_agentchat.agents import AssistantAgent
        en = _is_en()
        decider = AssistantAgent("Decider", model_client=self._client,
                                 system_message=_DECIDER_ANSWER_SYS_EN if en else _DECIDER_ANSWER_SYS)
        belief = self.belief or (
            "(No situation assessment yet; answer based solely on the task content below.)" if en
            else "（暂无态势研判，请仅依据下面任务内容作答。）")
        task = (_READ_PREAMBLE_EN if en else _READ_PREAMBLE).format(belief=belief, cutoff=self.cutoff) + point_prompt
        res = await asyncio.wait_for(decider.run(task=task), timeout=self.timeout)
        self._add_llm(self._count_model_msgs(res))
        return self._last_decider(res, "")

    def answer(self, point_prompt: str, kind: str = "brier") -> str:
        self.n_answer += 1
        adapter.STATS.bump("answer")
        temp = 0.6 if kind == "time" else 0.0
        try:
            out = self._loop.run(self._answer_async(point_prompt, temp))
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[autogen] answer failed ({type(e).__name__}: {e}), fallback to single call",
                  file=sys.stderr)
        # Fallback: direct single call (decider role) to guarantee answer text for B/F parsing.
        self._add_llm(1)
        en = _is_en()
        belief = self.belief or ("(No situation assessment yet)" if en else "（暂无态势研判）")
        return call_llm(
            [{"role": "user", "content":
              (_READ_PREAMBLE_EN if en else _READ_PREAMBLE).format(belief=belief, cutoff=self.cutoff) + point_prompt}],
            model=self.base, temperature=temp, max_tokens=self.max_tokens,
            reasoning_effort="high", include_reasoning=True,
            thinking_budget_tokens=self.thinking_budget, retries=1, timeout=self.timeout,
        )

    def close(self) -> None:
        try:
            self._loop.run(self._client.close())
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._loop.close()

    def stats(self) -> dict:
        return {"advance": self.n_advance, "answer": self.n_answer, "llm": self.n_llm, "sim": 0}


def make_session(
    base_model: str,
    *,
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
    sim_params=None,  # unused by autogen
) -> _AutoGenSession:
    return _AutoGenSession(base_model, max_tokens=max_tokens, thinking_budget=thinking_budget)
