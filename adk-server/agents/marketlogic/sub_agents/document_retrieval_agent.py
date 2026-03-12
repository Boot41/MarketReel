from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from ..config import config
from ..tools import index_navigator, index_registry, sufficiency_checker, targeted_fetcher

_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "DocumentRetrievalAgent_prompt.txt"
).read_text(encoding="utf-8").strip()

document_retrieval_agent = Agent(
    name="DocumentRetrievalAgent",
    model=config.worker_model,
    description=(
        "Index-guided document retrieval sub-agent for MarketReel. Called exclusively by "
        "DataAgent to fetch script scenes, censorship guidelines, cultural sensitivity "
        "reports, reviews, marketing briefs, and synopses from the local corpus. Uses "
        "index_registry → index_navigator → targeted_fetcher → sufficiency_checker."
    ),
    instruction=_PROMPT,
    tools=[
        FunctionTool(index_registry),
        FunctionTool(index_navigator),
        FunctionTool(targeted_fetcher),
        FunctionTool(sufficiency_checker),
    ],
    output_key="retrieved_documents",
)
