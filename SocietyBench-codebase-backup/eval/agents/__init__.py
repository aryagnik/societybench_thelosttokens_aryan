"""SocietyBench agent evaluation (M3).

Wraps each agent configuration (framework x base model) as a "question-answering
black box" whose input/output format is identical to a bare LLM (calibration
questions `1. 75%`, time questions `1. 2025-08-12`), so the scoring logic of
predict_step3B_brier and predict_step3F_time can be reused and results land in
the same output directory as bare models.

Modules:
  llm_client          DMXAPI -> OpenAI-compatible client (shared by all 3 frameworks)
  adapter             unified entry point agent_answer(framework, base_model, ...)
  fw_langgraph        LangGraph (plan then solve)
  fw_autogen          AutoGen (multi-agent debate)
  fw_mirofish_oasis   MiroFish (social simulation based on open-source OASIS)
  run_agents          main runner: reuses 3B/3F scoring, only replaces the "call model" step

Rules are documented in skills/predict-step3M-agents/SKILL.md (rules are maintained only in the skill).
"""
