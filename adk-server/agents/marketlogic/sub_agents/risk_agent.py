from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent

from ..config import config

_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "RiskAgent_prompt.txt"
).read_text(encoding="utf-8").strip()

risk_agent = Agent(
    name="RiskAgent",
    model=config.worker_model,
    description=(
        "Identifies censorship, cultural sensitivity, and market risks for a film "
        "in a territory. Consumes the evidence_bundle from DataAgent via session state."
    ),
    instruction=_PROMPT,
    output_key="risk_flags",
)
