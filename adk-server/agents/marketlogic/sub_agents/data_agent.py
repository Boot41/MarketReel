from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from ..config import config
from ..tools import (
    get_actor_qscore,
    get_box_office_by_genre_territory,
    get_comparable_films,
    get_exchange_rates,
    get_theatrical_window_trends,
    get_vod_price_benchmarks,
)
from .document_retrieval_agent import document_retrieval_agent

_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "DataAgent_prompt.txt"
).read_text(encoding="utf-8").strip()

data_agent = Agent(
    name="DataAgent",
    model=config.worker_model,
    description=(
        "Single data gateway for document and structured DB evidence retrieval. "
        "Called by the Orchestrator with an evidence request type. Never reasons "
        "about strategy, valuation, or risk."
    ),
    instruction=_PROMPT,
    sub_agents=[document_retrieval_agent],
    tools=[
        FunctionTool(get_box_office_by_genre_territory),
        FunctionTool(get_actor_qscore),
        FunctionTool(get_theatrical_window_trends),
        FunctionTool(get_exchange_rates),
        FunctionTool(get_vod_price_benchmarks),
        FunctionTool(get_comparable_films),
    ],
    output_key="evidence_bundle",
)
