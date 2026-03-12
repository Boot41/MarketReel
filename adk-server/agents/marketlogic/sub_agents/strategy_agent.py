from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from ..config import config
from ..tools import get_theatrical_window_trends

_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "StrategyAgent_prompt.txt"
).read_text(encoding="utf-8").strip()

strategy_agent = Agent(
    name="StrategyAgent",
    model=config.worker_model,
    description=(
        "Produces a release strategy recommendation using valuation_result and "
        "risk_flags from session state."
    ),
    instruction=_PROMPT,
    tools=[
        FunctionTool(get_theatrical_window_trends),
    ],
    output_key="strategy_result",
)
