from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.marketlogic import orchestrator  # noqa: E402
from agents.marketlogic.sub_agents import data_agent as data_agent_module  # noqa: E402
from agents.marketlogic.sub_agents import document_retrieval_agent as retrieval_module  # noqa: E402


@pytest.mark.asyncio
async def test_run_data_agent_expands_when_initial_fetch_is_insufficient(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    def _fake_navigator(movie: str, territory: str, intent: str) -> dict[str, Any]:
        return {
            "movie": movie,
            "territory": territory,
            "intent": intent,
            "doc_types": ["reviews"],
            "max_docs": 2,
            "max_scenes": 1,
        }

    def _fake_fetcher(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        calls.append(dict(plan))
        if int(plan.get("max_docs", 0)) <= 2:
            return {"documents": [{"doc_id": "d1", "source_path": "reviews/a.md", "text": "x"}], "scenes": []}
        return {
            "documents": [
                {"doc_id": "d1", "source_path": "reviews/a.md", "text": "x"},
                {"doc_id": "d2", "source_path": "reviews/b.md", "text": "y"},
                {"doc_id": "d3", "source_path": "reviews/c.md", "text": "z"},
            ],
            "scenes": [{"doc_id": "s1", "source_path": "scripts/a.md", "text": "scene"}],
        }

    monkeypatch.setattr(retrieval_module, "IndexNavigator", _fake_navigator)
    monkeypatch.setattr(retrieval_module, "TargetedFetcher", _fake_fetcher)
    monkeypatch.setattr(
        data_agent_module,
        "source_citation_tool",
        lambda items: [{"source_path": item["source_path"]} for item in items],
    )

    bundle = await orchestrator.run_data_agent(
        {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "risk",
            "needs_docs": True,
            "needs_db": False,
        }
    )

    assert len(calls) == 2
    assert calls[0]["max_docs"] == 2
    assert calls[1]["max_docs"] > calls[0]["max_docs"]
    assert bundle["data_sufficiency_score"] >= 0.5
    assert len(bundle["citations"]) == 4


@pytest.mark.asyncio
async def test_orchestrator_reuses_session_artifacts_for_strategy_followup(monkeypatch: pytest.MonkeyPatch):
    async def _unexpected_run_data_agent(_: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("run_data_agent should not be called for scenario follow-up reuse")

    async def _unexpected_run_risk_reasoner(
        _: dict[str, Any], *, provider_enabled: bool
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        _ = provider_enabled
        raise AssertionError("run_risk_reasoner should not be called for scenario follow-up reuse")

    async def _unexpected_run_valuation_reasoner(
        *, evidence: dict[str, Any], risk_flags: list[dict[str, Any]], provider_enabled: bool
    ) -> tuple[dict[str, Any] | None, str | None]:
        _ = evidence
        _ = risk_flags
        _ = provider_enabled
        raise AssertionError("run_valuation_reasoner should not be called for scenario follow-up reuse")

    monkeypatch.setattr(orchestrator, "run_data_agent", _unexpected_run_data_agent)
    monkeypatch.setattr(orchestrator, "run_risk_reasoner", _unexpected_run_risk_reasoner)
    monkeypatch.setattr(orchestrator, "run_valuation_reasoner", _unexpected_run_valuation_reasoner)

    session_state = {
        "resolved_context": {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "scenario_override": None,
        },
        "evidence_bundle": {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "document_evidence": {"documents": [], "scenes": []},
            "db_evidence": {
                "theatrical_windows": [{"window_type": "standard", "days": 42}],
                "exchange_rates": {"currency_code": "INR", "rate_to_usd": 83.0},
            },
            "citations": [{"source_path": "docs/reviews/interstellar.md", "doc_id": "r1", "page": 1, "excerpt": "good"}],
            "data_sufficiency_score": 0.7,
        },
        "risk": [
            {
                "category": "MARKET",
                "severity": "LOW",
                "scene_ref": "market_baseline",
                "source_ref": "derived:baseline",
                "mitigation": "Baseline checks.",
                "confidence": 0.5,
            }
        ],
        "valuation": {
            "mg_estimate_usd": 1000000.0,
            "confidence_interval_low_usd": 800000.0,
            "confidence_interval_high_usd": 1200000.0,
            "theatrical_projection_usd": 2200000.0,
            "vod_projection_usd": 900000.0,
            "comparable_films": ["X"],
            "sufficiency_score": 0.7,
        },
    }

    scorecard, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="If we skip theatrical and go straight to streaming, how does ROI change?",
        session_state=session_state,
        provider_enabled=False,
    )

    assert scorecard["release_timeline"]["release_mode"] == "streaming_first"
    assert state_delta["resolved_context"]["movie"] == "Interstellar"
    assert state_delta["resolved_context"]["territory"] == "India"


@pytest.mark.asyncio
async def test_orchestrator_handles_small_talk_without_workflow() -> None:
    reply, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="hi",
        session_state={},
        provider_enabled=False,
    )

    assert reply["response_type"] == "conversation_response"
    assert "movie and territory" in str(reply.get("message", "")).lower()
    assert state_delta["last_agent_response_type"] == "conversation_response"


@pytest.mark.asyncio
async def test_orchestrator_asks_for_missing_context_on_analytic_prompt() -> None:
    reply, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="Should we acquire this?",
        session_state={},
        provider_enabled=False,
    )

    assert reply["response_type"] == "clarification_response"
    assert "movie title and target territory" in str(reply.get("message", ""))
    assert state_delta["last_agent_response_type"] == "clarification_response"


@pytest.mark.asyncio
async def test_orchestrator_blocks_scorecard_on_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_data_agent(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "document_evidence": {"documents": [], "scenes": []},
            "db_evidence": {},
            "citations": [],
            "data_sufficiency_score": 0.1,
            "tool_diagnostics": [
                {
                    "source": "db",
                    "endpoint": "/internal/v1/market/box-office",
                    "error_type": "network",
                    "status_code": None,
                    "message": "connect failed",
                }
            ],
            "tool_failure_count": 1,
        }

    monkeypatch.setattr(orchestrator, "run_data_agent", _fake_run_data_agent)

    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="give me a full scorecard for interstellar in india",
        session_state={},
        provider_enabled=False,
    )

    assert payload["response_type"] == "clarification_response"
    assert payload["reason_code"] == "backend_unavailable"
    assert "reachable_backend_internal_api" in payload["missing_requirements"]
    assert state_delta["last_agent_response_type"] == "clarification_response"


@pytest.mark.asyncio
async def test_orchestrator_blocks_scorecard_on_internal_auth_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_data_agent(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "document_evidence": {"documents": [], "scenes": []},
            "db_evidence": {},
            "citations": [],
            "data_sufficiency_score": 0.1,
            "tool_diagnostics": [
                {
                    "source": "db",
                    "endpoint": "/internal/v1/market/box-office",
                    "error_type": "auth",
                    "status_code": 401,
                    "message": "http_401",
                }
            ],
            "tool_failure_count": 1,
        }

    monkeypatch.setattr(orchestrator, "run_data_agent", _fake_run_data_agent)

    payload, _ = await orchestrator.run_marketlogic_orchestrator(
        message="give me a full scorecard for interstellar in india",
        session_state={},
        provider_enabled=False,
    )

    assert payload["response_type"] == "clarification_response"
    assert payload["reason_code"] == "internal_auth_failed"
    assert "valid_internal_api_auth" in payload["missing_requirements"]


@pytest.mark.asyncio
async def test_orchestrator_handles_evidence_followup_without_workflow_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_run_data_agent(_: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("run_data_agent should not be called for evidence follow-up")

    monkeypatch.setattr(orchestrator, "run_data_agent", _unexpected_run_data_agent)

    session_state = {
        "resolved_context": {
            "movie": "Interstellar",
            "territory": "India",
            "workflow_intent": "full_scorecard",
            "scenario_override": None,
        },
        "last_scorecard": {
            "citations": [
                {
                    "source_path": "docs/reviews/interstellar.md",
                    "doc_id": "r1",
                    "page": 1,
                    "excerpt": "good",
                }
            ]
        },
    }

    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="show me sources",
        session_state=session_state,
        provider_enabled=False,
    )

    assert payload["response_type"] == "conversation_response"
    assert "citations_by_source" in payload
    assert state_delta["last_agent_response_type"] == "conversation_response"


@pytest.mark.asyncio
async def test_orchestrator_returns_clarification_on_strategy_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_data_agent(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "document_evidence": {"documents": [], "scenes": []},
            "db_evidence": {
                "box_office": {"samples": 3, "avg_gross_usd": 1000000.0},
                "comparable_films": [{"title": "X", "territory_gross_usd": 2000000.0}],
                "exchange_rates": {"currency_code": "INR", "rate_to_usd": 83.0},
            },
            "citations": [
                {"source_path": "docs/reviews/interstellar.md", "doc_id": "r1", "page": 1, "excerpt": "good"},
                {"source_path": "docs/synopses/interstellar.md", "doc_id": "s1", "page": 1, "excerpt": "good"},
                {"source_path": "docs/marketing/interstellar.md", "doc_id": "m1", "page": 1, "excerpt": "good"},
            ],
            "data_sufficiency_score": 0.8,
            "tool_diagnostics": [],
            "tool_failure_count": 0,
        }

    async def _fake_run_risk_reasoner(
        _: dict[str, Any], *, provider_enabled: bool
    ) -> tuple[list[dict[str, Any]], str | None]:
        _ = provider_enabled
        return (
            [
                {
                    "category": "MARKET",
                    "severity": "LOW",
                    "scene_ref": "market_baseline",
                    "source_ref": "derived:baseline",
                    "mitigation": "Baseline checks.",
                    "confidence": 0.5,
                }
            ],
            None,
        )

    async def _fake_run_valuation_reasoner(
        *, evidence: dict[str, Any], risk_flags: list[dict[str, Any]], provider_enabled: bool
    ) -> tuple[dict[str, Any], str | None]:
        _ = evidence
        _ = risk_flags
        _ = provider_enabled
        return (
            {
                "mg_estimate_usd": 1000000.0,
                "confidence_interval_low_usd": 800000.0,
                "confidence_interval_high_usd": 1200000.0,
                "theatrical_projection_usd": 2200000.0,
                "vod_projection_usd": 900000.0,
                "comparable_films": ["X"],
                "sufficiency_score": 0.8,
            },
            None,
        )

    async def _fake_run_strategy_reasoner(
        _orchestrator_input: dict[str, Any],
        _evidence: dict[str, Any],
        _valuation: dict[str, Any],
        _risk_flags: list[dict[str, Any]],
        *,
        provider_enabled: bool,
    ) -> tuple[None, str]:
        _ = provider_enabled
        return None, "strategy_reasoner_schema_invalid"

    monkeypatch.setattr(orchestrator, "run_data_agent", _fake_run_data_agent)
    monkeypatch.setattr(orchestrator, "run_risk_reasoner", _fake_run_risk_reasoner)
    monkeypatch.setattr(orchestrator, "run_valuation_reasoner", _fake_run_valuation_reasoner)
    monkeypatch.setattr(orchestrator, "run_strategy_reasoner", _fake_run_strategy_reasoner)

    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="give me a full scorecard for interstellar in india",
        session_state={},
        provider_enabled=True,
    )

    assert payload["response_type"] == "clarification_response"
    assert payload["reason_code"] == "strategy_reasoner_schema_invalid"
    assert "valid_structured_reasoning_output" in payload["missing_requirements"]
    assert state_delta["last_agent_response_type"] == "clarification_response"
