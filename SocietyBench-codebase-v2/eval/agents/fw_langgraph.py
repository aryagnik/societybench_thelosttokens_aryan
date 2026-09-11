"""LangGraph — draft then calibrate (a "draft->refine" variant of plan-and-solve).

Two-step graph: draft (single-shot initial judgment by the base model, anchoring single-shot
quality) -> refine (calibration review against the evidence).
Nodes call call_llm from eval/common.py directly (same base-model call path as bare models).
Design intent: the orchestration **adds** a calibration review on top of the single-shot
judgment, rather than washing out the signal with multi-step reasoning (smoke tests found
naive plan-and-solve / symmetric debate pushes probabilities toward the prior and loses
true/false discrimination).

Public API: answer(base_model, prompt, kind="brier") -> str — answer text, same format as bare models.
"""
from __future__ import annotations

import pathlib
import sys
from typing import TypedDict

_EVAL = pathlib.Path(__file__).resolve().parent.parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import adapter  # noqa: E402  (process-level call counter STATS; adapter loads before this module, no cycle)
from common import call_llm  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402


class _State(TypedDict):
    base: str
    prompt: str
    max_tokens: int
    thinking_budget: int
    draft: str
    answer: str


_REFINE_PROMPT = (
    "{prompt}\n\n"
    "======== 这是你的初版答案(逐题)========\n{draft}\n========\n"
    "请对照已知信息做一次**校准复核**,然后输出最终答案:\n"
    "- 某题若证据其实较强,却给了接近 50%,拉向更确定的一侧(更高或更低);\n"
    "- 某题若给了极端值但证据其实不足,适度回收;\n"
    "- 严格区分『确有迹象会发生』与『设计上看似合理、却无信息支撑』——后者应明显偏低;\n"
    "- 充分利用整个 0-100% 区间,不要把多数题堆在中间或一律偏低。\n"
    "严格按上面任务要求的格式逐题一行,只输出答案行,不要解释、不要复述复核过程。"
)

_REFINE_PROMPT_EN = (
    "{prompt}\n\n"
    "======== Your draft answers (per question) ========\n{draft}\n========\n"
    "Do one **calibration review** against the known information, then output the final answers:\n"
    "- If a question's evidence is actually strong but you gave near 50%, push toward the more "
    "certain side (higher or lower);\n"
    "- If a question got an extreme value but the evidence is actually weak, pull it back moderately;\n"
    "- Strictly distinguish 'there are signs it will happen' from 'plausible by design but "
    "unsupported by information' — the latter should be clearly low;\n"
    "- Use the full 0-100% range; do not pile most questions in the middle or uniformly low.\n"
    "Answer strictly in the format required by the task above, one line per question, output only "
    "the answer lines, no explanation, no restating the review."
)


def _draft_node(state: _State) -> dict:
    txt = call_llm(
        [{"role": "user", "content": state["prompt"]}],
        model=state["base"], temperature=0.0, max_tokens=int(state.get("max_tokens", 32000)),
        reasoning_effort="high", include_reasoning=True,
        thinking_budget_tokens=int(state.get("thinking_budget", 24000)),
    )
    return {"draft": txt}


def _refine_node(state: _State) -> dict:
    txt = call_llm(
        [{"role": "user", "content": (_REFINE_PROMPT_EN if _is_en() else _REFINE_PROMPT).format(prompt=state["prompt"], draft=state.get("draft", ""))}],
        model=state["base"], temperature=0.0, max_tokens=int(state.get("max_tokens", 32000)),
        reasoning_effort="high", include_reasoning=True,
        thinking_budget_tokens=int(state.get("thinking_budget", 24000)),
    )
    return {"answer": txt}


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        g = StateGraph(_State)
        g.add_node("draft", _draft_node)
        g.add_node("refine", _refine_node)
        g.add_edge(START, "draft")
        g.add_edge("draft", "refine")
        g.add_edge("refine", END)
        _GRAPH = g.compile()
    return _GRAPH


def answer(
    base_model: str,
    prompt: str,
    kind: str = "brier",
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
) -> str:
    out = _graph().invoke({
        "base": base_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "thinking_budget": thinking_budget,
        "draft": "",
        "answer": "",
    })
    return out.get("answer", "") or out.get("draft", "")


# ===========================================================================
# Continuous rollout session (M3 single-rollout refactor): one session per event,
# advance along the timeline, answer along the way.
# ===========================================================================

# The rolling "situation assessment" is this framework's rollout state / working memory.
# advance integrates new developments into the assessment (this step carries the original
# draft->refine "calibration review" duty, just done incrementally along the timeline);
# answer reads the point's answer off the assessment.
_BELIEF_INIT = (
    "你在对一个持续演变的公共事件做连续推演。下面是该事件**截至 {cutoff}** 的进展。\n"
    "请据此写一份**当前态势研判**,作为你后续预测的工作记忆:\n"
    "- 完整保留关键事实、日期、数字、人物、机构、已发生的具体事件节点(尽量不丢信息);\n"
    "- 梳理各方动态、矛盾焦点、舆论走向;\n"
    "- 研判事态的发展方向、已有迹象会发生 / 看似合理但无信息支撑 的事,以及不确定性所在。\n"
    "这是你私人的分析笔记,要信息完备、可据此判断未来各类事件是否会发生。\n\n"
    "===== 进展(截至 {cutoff})=====\n{delta}\n=====\n\n"
    "直接输出态势研判正文,不要解释。"
)
_BELIEF_UPDATE = (
    "你在对一个公共事件做连续推演。这是你**此前**形成的态势研判:\n"
    "===== 此前态势研判 =====\n{belief}\n=====\n\n"
    "现在时间推进到 **{cutoff}**,出现以下**新进展**:\n"
    "===== 新进展 =====\n{delta}\n=====\n\n"
    "请把新进展**整合**进研判,输出**更新后的完整态势研判**:\n"
    "- 保留仍然有效的旧信息,纳入新事实,修正被新进展推翻的旧判断;\n"
    "- 继续完整保留关键事实/日期/数字/人物/已发生节点;\n"
    "- 更新对发展方向与不确定性的研判,严格区分『确有迹象会发生』与『看似合理却无信息支撑』。\n"
    "直接输出更新后的态势研判正文,不要复述你改了什么。"
)
_READ_PREAMBLE = (
    "下面是你对该事件持续推演形成的【当前态势研判】(截至 {cutoff},你的工作记忆):\n"
    "===== 当前态势研判 =====\n{belief}\n=====\n\n"
    "请**完全基于上述态势研判**完成下面的预测任务,严格按其要求的格式作答。\n\n"
)

# English version (for English workspaces, consistent with the main experiment's English setup; triggered by env SB_AGENT_LANG=en)
_BELIEF_INIT_EN = (
    "You are performing continuous reasoning about an evolving public event. Below are the "
    "developments **as of {cutoff}**.\n"
    "Based on them, write a **current situation assessment** to serve as your working memory "
    "for later predictions:\n"
    "- Fully preserve key facts, dates, numbers, people, institutions, and concrete event "
    "milestones that have occurred (try not to lose information);\n"
    "- Lay out each party's moves, the points of contention, and the direction of public opinion;\n"
    "- Assess where the situation is heading, what there are already signs will happen vs. what "
    "seems plausible but is unsupported by information, and where the uncertainties lie.\n"
    "This is your private analytical note; it must be information-complete and sufficient to judge "
    "whether various future events will occur.\n\n"
    "===== Developments (as of {cutoff}) =====\n{delta}\n=====\n\n"
    "Output the situation-assessment text directly, with no explanation."
)
_BELIEF_UPDATE_EN = (
    "You are performing continuous reasoning about a public event. This is the situation "
    "assessment you formed **previously**:\n"
    "===== Previous situation assessment =====\n{belief}\n=====\n\n"
    "Time now advances to **{cutoff}**, with the following **new developments**:\n"
    "===== New developments =====\n{delta}\n=====\n\n"
    "**Integrate** the new developments into the assessment and output the **updated, complete "
    "situation assessment**:\n"
    "- Keep still-valid old information, incorporate the new facts, and correct old judgments "
    "overturned by the new developments;\n"
    "- Continue to fully preserve key facts/dates/numbers/people/occurred milestones;\n"
    "- Update the assessment of direction and uncertainty, strictly distinguishing 'there are "
    "signs it will happen' from 'seems plausible but unsupported by information'.\n"
    "Output the updated situation-assessment text directly; do not restate what you changed."
)
_READ_PREAMBLE_EN = (
    "Below is the **current situation assessment** you have formed through continuous reasoning "
    "about this event (as of {cutoff}, your working memory):\n"
    "===== Current situation assessment =====\n{belief}\n=====\n\n"
    "Complete the prediction task below **entirely based on the above situation assessment**, "
    "answering strictly in the required format.\n\n"
)


def _is_en() -> bool:
    import os
    return os.environ.get("SB_AGENT_LANG", "zh").strip().lower() == "en"


class _LangGraphSession:
    """LangGraph continuous rollout session: keeps a rolling running_belief.

    advance = 1 LLM call updating running_belief+delta into a new belief;
    answer  = 1 LLM call answering the point's questions on the belief (brier uses temp=0
              for stability; time uses temp>0 so repeated readouts carry randomness,
              matching runs_per_point sampling).
    """

    def __init__(self, base_model: str, *, max_tokens: int, thinking_budget: int,
                 time_temperature: float = 0.6) -> None:
        self.base = base_model
        self.max_tokens = int(max_tokens)
        self.thinking_budget = int(thinking_budget)
        self.time_temperature = float(time_temperature)
        self.belief = ""
        self.cutoff = ""
        self.n_advance = 0
        self.n_answer = 0
        self.n_llm = 0

    def _llm(self, prompt: str, temperature: float = 0.0) -> str:
        self.n_llm += 1
        adapter.STATS.bump("llm")
        return call_llm(
            [{"role": "user", "content": prompt}],
            model=self.base, temperature=temperature, max_tokens=self.max_tokens,
            reasoning_effort="high", include_reasoning=True,
            thinking_budget_tokens=self.thinking_budget,
        )

    def advance(self, delta_context: str, cutoff: str) -> None:
        self.cutoff = cutoff or self.cutoff
        delta = (delta_context or "").strip()
        if not delta:
            return  # no new developments in this span (rare): keep state unchanged, save an LLM call
        self.n_advance += 1
        adapter.STATS.bump("advance")
        en = _is_en()
        init_t = _BELIEF_INIT_EN if en else _BELIEF_INIT
        upd_t = _BELIEF_UPDATE_EN if en else _BELIEF_UPDATE
        if not self.belief:
            self.belief = self._llm(init_t.format(cutoff=self.cutoff, delta=delta)).strip()
        else:
            self.belief = self._llm(
                upd_t.format(belief=self.belief, cutoff=self.cutoff, delta=delta)
            ).strip()

    def answer(self, point_prompt: str, kind: str = "brier") -> str:
        self.n_answer += 1
        adapter.STATS.bump("answer")
        temp = self.time_temperature if kind == "time" else 0.0
        en = _is_en()
        if self.belief:
            belief = self.belief
        else:
            belief = ("(No situation assessment available yet; answer based solely on the task "
                      "content below.)" if en else "（暂无可用态势研判，请仅依据下面任务内容作答。）")
        pre = _READ_PREAMBLE_EN if en else _READ_PREAMBLE
        full = pre.format(belief=belief, cutoff=self.cutoff) + point_prompt
        return self._llm(full, temperature=temp)

    def close(self) -> None:  # no external resources
        return None

    def stats(self) -> dict:
        return {"advance": self.n_advance, "answer": self.n_answer, "llm": self.n_llm, "sim": 0}


def make_session(
    base_model: str,
    *,
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
    sim_params=None,  # unused by langgraph
) -> _LangGraphSession:
    return _LangGraphSession(base_model, max_tokens=max_tokens, thinking_budget=thinking_budget)
