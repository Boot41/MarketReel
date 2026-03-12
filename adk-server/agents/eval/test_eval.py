"""Agent eval suite for MarketLogicOrchestrator (ADK-native pattern).

Tests cover:
- Agent configuration contract (root_agent shape, sub_agents wired up)
- Tool output schemas/contracts
- Validation checker behavior
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/marketreeldb")
os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("ADK_API_KEY", "test")
os.environ.setdefault("ADK_MODEL", "gemini-2.0-flash")
os.environ.setdefault("BACKEND_BASE_URL", "http://localhost:8010")

from agents.marketlogic.agent import root_agent
from agents.marketlogic.tools import (
    exchange_rate_tool,
    index_navigator,
    index_registry,
    mg_calculator_tool,
    sufficiency_checker,
    targeted_fetcher,
)
from agents.marketlogic.sub_agents.validation_checkers import (
    RiskOutputChecker,
    StrategyOutputChecker,
    ValuationOutputChecker,
)


# ---------------------------------------------------------------------------
# Agent configuration contract
# ---------------------------------------------------------------------------


def test_root_agent_name() -> None:
    assert root_agent.name == "MarketLogicOrchestrator"


def test_root_agent_has_all_sub_agents() -> None:
    names = {a.name for a in root_agent.sub_agents}
    assert names == {
        "DataAgent",
        "RiskAgent",
        "ValuationAgent",
        "StrategyAgent",
        "ExplainabilityAgent",
    }


def test_root_agent_instruction_is_non_empty() -> None:
    assert root_agent.instruction and len(root_agent.instruction) > 100


def test_data_agent_has_document_retrieval_sub_agent() -> None:
    from agents.marketlogic.sub_agents import data_agent

    sub_agent_names = {a.name for a in (data_agent.sub_agents or [])}
    assert "DocumentRetrievalAgent" in sub_agent_names


def test_document_retrieval_agent_config() -> None:
    from agents.marketlogic.sub_agents import document_retrieval_agent

    assert document_retrieval_agent.name == "DocumentRetrievalAgent"
    assert document_retrieval_agent.output_key == "retrieved_documents"
    tool_names = {t.func.__name__ if hasattr(t, "func") else str(t) for t in document_retrieval_agent.tools}
    assert {"index_registry", "index_navigator", "targeted_fetcher", "sufficiency_checker"} == tool_names


def test_valuation_agent_has_mg_calculator_tool() -> None:
    from agents.marketlogic.sub_agents import valuation_agent

    tool_names = {t.func.__name__ if hasattr(t, "func") else str(t) for t in valuation_agent.tools}
    assert "mg_calculator_tool" in tool_names


def test_sub_agents_have_output_keys() -> None:
    from agents.marketlogic.sub_agents import (
        data_agent,
        document_retrieval_agent,
        explainability_agent,
        risk_agent,
        strategy_agent,
        valuation_agent,
    )

    assert data_agent.output_key == "evidence_bundle"
    assert document_retrieval_agent.output_key == "retrieved_documents"
    assert risk_agent.output_key == "risk_flags"
    assert valuation_agent.output_key == "valuation_result"
    assert strategy_agent.output_key == "strategy_result"
    assert explainability_agent.output_key == "explanation"


# ---------------------------------------------------------------------------
# Tool output schema / contract
# ---------------------------------------------------------------------------


def test_mg_calculator_returns_float() -> None:
    result = mg_calculator_tool(
        avg_box_office_usd=5_000_000.0,
        avg_qscore=70.0,
        comparable_avg_gross_usd=8_000_000.0,
        risk_penalty=0.1,
    )
    assert isinstance(result, float)
    assert result >= 250_000.0  # floor enforced


def test_mg_calculator_baseline_fallback() -> None:
    result = mg_calculator_tool(
        avg_box_office_usd=0.0,
        avg_qscore=0.0,
        comparable_avg_gross_usd=0.0,
        risk_penalty=0.0,
        allow_baseline_fallback=True,
    )
    assert result >= 250_000.0


def test_exchange_rate_tool_returns_float() -> None:
    result = exchange_rate_tool(amount_usd=1_000_000.0, rate_to_usd=83.0)
    assert isinstance(result, float)
    assert result == pytest.approx(12_048.19, rel=1e-3)


def test_exchange_rate_tool_zero_rate_fallback() -> None:
    result = exchange_rate_tool(amount_usd=1_000_000.0, rate_to_usd=0.0)
    assert result == 1_000_000.0


def test_index_navigator_schema() -> None:
    """index_navigator returns a well-formed retrieval plan."""
    result = index_navigator(movie="Deadpool", territory="India", retrieval_intent="risk")
    assert set(result.keys()) >= {"movie", "territory", "retrieval_intent", "doc_types", "max_docs", "max_scenes"}
    assert isinstance(result["doc_types"], list)
    assert len(result["doc_types"]) > 0
    assert isinstance(result["max_docs"], int)
    assert isinstance(result["max_scenes"], int)
    # risk intent must include censorship-related doc types
    assert any("censorship" in t for t in result["doc_types"])


def test_index_navigator_full_scorecard_includes_all_types() -> None:
    result = index_navigator(movie="Interstellar", territory="Japan", retrieval_intent="full_scorecard")
    assert "script_scenes" in result["doc_types"]
    assert any("censorship" in t for t in result["doc_types"])
    assert result["max_docs"] >= 10


def test_targeted_fetcher_returns_expected_schema() -> None:
    """targeted_fetcher reads local corpus and returns correct structure."""
    result = targeted_fetcher(
        movie="Deadpool",
        territory="India",
        doc_types=["censorship"],
        max_docs=5,
        max_scenes=0,
    )
    assert set(result.keys()) == {"documents", "scenes", "total_documents", "total_scenes"}
    assert isinstance(result["documents"], list)
    assert isinstance(result["scenes"], list)
    assert result["total_documents"] == len(result["documents"])
    assert result["total_scenes"] == len(result["scenes"])


def test_targeted_fetcher_returns_content_for_known_movie() -> None:
    """targeted_fetcher returns docs for a movie that exists in the corpus."""
    result = targeted_fetcher(
        movie="Deadpool",
        territory="india",
        doc_types=["censorship"],
        max_docs=10,
        max_scenes=0,
    )
    assert result["total_documents"] > 0
    first_doc = result["documents"][0]
    assert "content" in first_doc
    assert "source_reference" in first_doc


def test_targeted_fetcher_territory_filter_for_guidelines() -> None:
    """targeted_fetcher filters censorship_guidelines_countries by territory."""
    result = targeted_fetcher(
        movie="",
        territory="india",
        doc_types=["censorship_guidelines_countries"],
        max_docs=5,
        max_scenes=0,
    )
    assert result["total_documents"] > 0
    for doc in result["documents"]:
        assert "india" in str(doc.get("doc_id", "")).lower()


def test_sufficiency_checker_pass() -> None:
    result = sufficiency_checker(total_documents=4, total_scenes=2, retrieval_intent="valuation")
    assert result["status"] == "PASS"
    assert 0.0 <= result["score"] <= 1.0
    assert result["total_items"] == 6


def test_sufficiency_checker_expand() -> None:
    result = sufficiency_checker(total_documents=1, total_scenes=0, retrieval_intent="risk")
    assert result["status"] == "EXPAND"
    assert result["guidance"] != ""


def test_index_registry_returns_known_corpus_structure() -> None:
    """index_registry reads local manifest and returns catalog structure."""
    result = index_registry(movie="Deadpool", territory="india")
    assert "available_docs" in result
    assert "known_movies" in result
    assert "known_territories" in result
    assert isinstance(result["available_docs"], list)
    # corpus should have Deadpool docs
    assert len(result["available_docs"]) > 0


# ---------------------------------------------------------------------------
# Validation checker behavior
# ---------------------------------------------------------------------------


def _mock_context(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


@pytest.mark.asyncio
async def test_risk_output_checker_escalates_when_valid() -> None:
    checker = RiskOutputChecker(name="risk_checker")
    ctx = _mock_context(
        {
            "risk_flags": [
                {
                    "category": "CENSORSHIP",
                    "severity": "HIGH",
                    "scene_ref": "s1",
                    "source_ref": "doc",
                    "mitigation": "cut",
                    "confidence": 0.9,
                }
            ]
        }
    )
    events = [e async for e in checker._run_async_impl(ctx)]
    assert len(events) == 1
    assert events[0].actions.escalate is True


@pytest.mark.asyncio
async def test_risk_output_checker_no_escalate_when_empty() -> None:
    checker = RiskOutputChecker(name="risk_checker")
    ctx = _mock_context({"risk_flags": []})
    events = [e async for e in checker._run_async_impl(ctx)]
    assert len(events) == 1
    assert not events[0].actions.escalate


@pytest.mark.asyncio
async def test_valuation_output_checker_escalates_when_valid() -> None:
    checker = ValuationOutputChecker(name="val_checker")
    ctx = _mock_context(
        {
            "valuation_result": {
                "mg_estimate_usd": 1_000_000.0,
                "confidence_interval_low_usd": 800_000.0,
                "confidence_interval_high_usd": 1_200_000.0,
                "theatrical_projection_usd": 2_400_000.0,
                "vod_projection_usd": 700_000.0,
            }
        }
    )
    events = [e async for e in checker._run_async_impl(ctx)]
    assert events[0].actions.escalate is True


@pytest.mark.asyncio
async def test_valuation_output_checker_no_escalate_when_missing_field() -> None:
    checker = ValuationOutputChecker(name="val_checker")
    ctx = _mock_context({"valuation_result": {"mg_estimate_usd": 1_000_000.0}})
    events = [e async for e in checker._run_async_impl(ctx)]
    assert not events[0].actions.escalate


@pytest.mark.asyncio
async def test_strategy_output_checker_escalates_when_valid() -> None:
    checker = StrategyOutputChecker(name="strat_checker")
    ctx = _mock_context(
        {
            "strategy_result": {
                "release_mode": "theatrical_first",
                "release_window_days": 45,
                "marketing_spend_usd": 300_000.0,
                "platform_priority": ["theatrical", "svod"],
                "roi_scenarios": {"base": 0.35},
            }
        }
    )
    events = [e async for e in checker._run_async_impl(ctx)]
    assert events[0].actions.escalate is True


@pytest.mark.asyncio
async def test_strategy_output_checker_no_escalate_when_missing() -> None:
    checker = StrategyOutputChecker(name="strat_checker")
    ctx = _mock_context({"strategy_result": None})
    events = [e async for e in checker._run_async_impl(ctx)]
    assert not events[0].actions.escalate
