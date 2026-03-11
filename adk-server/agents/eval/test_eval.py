from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.marketlogic import orchestrator


@pytest.mark.asyncio
async def test_eval_conversation_gate() -> None:
    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="hello",
        session_state={},
        provider_enabled=False,
    )
    assert payload["response_type"] == "conversation_response"
    assert state_delta["last_agent_response_type"] == "conversation_response"


@pytest.mark.asyncio
async def test_eval_clarification_gate() -> None:
    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="evaluate",
        session_state={},
        provider_enabled=False,
    )
    assert payload["response_type"] == "clarification_response"
    assert state_delta["last_agent_response_type"] == "clarification_response"


@pytest.mark.asyncio
async def test_eval_explainability_uses_session_state_read() -> None:
    session_state = {
        "resolved_context": {
            "movie": "Interstellar",
            "territory": "India",
            "workflow_intent": "full_scorecard",
            "scenario_override": None,
        },
        "last_scorecard": {
            "response_type": "scorecard_response",
            "citations": [{"source_path": "docs/reviews/interstellar.md"}],
            "warnings": [],
            "confidence": 0.7,
        },
        "evidence_bundle": {"data_sufficiency_score": 0.8},
    }

    payload, state_delta = await orchestrator.run_marketlogic_orchestrator(
        message="show citations and explain why",
        session_state=session_state,
        provider_enabled=False,
    )

    assert payload["response_type"] == "conversation_response"
    assert "explainability" in payload
    assert state_delta["last_agent_response_type"] == "conversation_response"


@pytest.mark.asyncio
async def test_eval_scorecard_contract_fields() -> None:
    session_state = {
        "resolved_context": {
            "movie": "Interstellar",
            "territory": "India",
            "workflow_intent": "full_scorecard",
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
            "data_sufficiency_score": 0.8,
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
            "sufficiency_score": 0.8,
        },
    }

    scorecard, _ = await orchestrator.run_marketlogic_orchestrator(
        message="If we skip theatrical and go straight to streaming, how does ROI change?",
        session_state=session_state,
        provider_enabled=False,
    )

    assert scorecard["response_type"] == "scorecard_response"
    assert "marketing_spend_usd" in scorecard
    assert "platform_priority" in scorecard
    assert "roi_scenarios" in scorecard
    assert scorecard["evidence_basis"] in {"grounded", "benchmark_derived"}
    assert isinstance(scorecard["degraded_mode"], dict)
    assert "enabled" in scorecard["degraded_mode"]


@pytest.mark.asyncio
async def test_eval_no_hidden_scorecard_when_tools_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_data_agent(_: dict[str, object]) -> dict[str, object]:
        return {
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "document_evidence": {"documents": [], "scenes": []},
            "db_evidence": {},
            "citations": [],
            "data_sufficiency_score": 0.0,
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

    payload, _ = await orchestrator.run_marketlogic_orchestrator(
        message="give me a full scorecard for interstellar in india",
        session_state={},
        provider_enabled=False,
    )

    assert payload["response_type"] == "clarification_response"
    assert payload["reason_code"] == "backend_unavailable"
