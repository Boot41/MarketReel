from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from ..config import config
from ..tools import (
    exchange_rate_tool,
    get_box_office_by_genre_territory,
    get_comparable_films,
    get_vod_price_benchmarks,
    mg_calculator_tool,
)

_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "ValuationAgent_prompt.txt"
).read_text(encoding="utf-8").strip()

valuation_agent = Agent(
    name="ValuationAgent",
    model=config.worker_model,
    description=(
        "Estimates the Minimum Guarantee and projects theatrical and VOD revenue "
        "for a film in a territory. Consumes the evidence_bundle from DataAgent "
        "via session state."
    ),
    instruction=_PROMPT,
    tools=[
        FunctionTool(mg_calculator_tool),
        FunctionTool(exchange_rate_tool),
        FunctionTool(get_box_office_by_genre_territory),
        FunctionTool(get_comparable_films),
        FunctionTool(get_vod_price_benchmarks),
    ],
    output_key="valuation_result",
)
