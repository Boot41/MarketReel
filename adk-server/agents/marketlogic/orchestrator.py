from __future__ import annotations

from typing import Any

from loguru import logger

from .sub_agents.data_agent import DataAgent
from .sub_agents.risk_agent import RiskAgent
from .sub_agents.strategy_agent import StrategyAgent
from .sub_agents.valuation_agent import ValuationAgent
from .tools import (
    IndexRegistry,
    combine_validation_warnings,
    confidence_threshold_check,
    exchange_rate_tool,
    financial_sanity_check,
    format_scorecard,
    hallucination_check,
)
from .types import (
    EvidenceBundle,
    EvidenceRequest,
    IntentType,
    OrchestratorInput,
    OrchestratorRoute,
    RiskFlag,
    Scorecard,
    StrategyResult,
    ValidationReport,
    ValuationResult,
)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_small_talk(message: str) -> bool:
    msg = _normalize(message)
    conversational_exact = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "help",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    }
    if msg in conversational_exact:
        return True
    if any(msg.startswith(prefix) for prefix in ["hi ", "hello ", "hey ", "thanks ", "help "]):
        return True
    return len(msg.split()) <= 2 and msg in {"sup", "hola"}


def _classify_intent(message: str) -> IntentType:
    msg = _normalize(message)
    if _is_small_talk(message):
        return "small_talk"
    if any(token in msg for token in ["censor", "sensitivity", "risk", "ban", "edit"]):
        return "risk"
    if any(token in msg for token in ["mg", "minimum guarantee", "price", "valuation", "pay"]):
        return "valuation"
    if any(token in msg for token in ["release", "window", "marketing", "streaming", "theatrical", "roi"]):
        return "strategy"
    if any(token in msg for token in ["acquire", "acquisition", "deal", "recommend", "scorecard", "evaluate"]):
        return "full_scorecard"
    return "small_talk"


def _match_entity(message: str, options: list[str]) -> str | None:
    msg = _normalize(message)
    best_match: str | None = None
    best_len = 0
    for option in options:
        norm = _normalize(option)
        if norm and norm in msg and len(norm) > best_len:
            best_match = option
            best_len = len(norm)
    return best_match


def _detect_scenario_override(message: str) -> str | None:
    msg = _normalize(message)
    if "skip theatrical" in msg or "straight to streaming" in msg or "streaming-first" in msg:
        return "streaming_first"
    if "theatrical" in msg and "first" in msg:
        return "theatrical_first"
    return None


def _build_direct_response() -> str:
    return (
        "Hi! I can evaluate a film's acquisition and release strategy. "
        "Tell me a movie and territory, for example: 'Evaluate Interstellar for India'."
    )


def _build_clarification(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "Please share the movie and territory to continue."
    if len(missing_fields) == 2:
        return "Please share both the movie title and target territory to run analysis."
    if missing_fields[0] == "movie":
        return "Please share the movie title so I can continue the analysis."
    return "Please share the target territory so I can continue the analysis."


def resolve_orchestrator_input(message: str, session_state: dict[str, Any]) -> OrchestratorInput:
    registry = IndexRegistry()
    known_movies = list(registry.get("known_movies", []))
    known_territories = list(registry.get("known_territories", []))

    movie = _match_entity(message, known_movies)
    territory = _match_entity(message, known_territories)

    previous_context = session_state.get("resolved_context")
    if isinstance(previous_context, dict):
        previous_movie = previous_context.get("movie")
        previous_territory = previous_context.get("territory")
        if movie is None and isinstance(previous_movie, str) and previous_movie.strip():
            movie = previous_movie.strip()
        if territory is None and isinstance(previous_territory, str) and previous_territory.strip():
            territory = previous_territory.strip()

    return {
        "message": message,
        "movie": movie,
        "territory": territory,
        "intent": _classify_intent(message),
        "scenario_override": _detect_scenario_override(message),
    }


def decide_action(orchestrator_input: OrchestratorInput) -> OrchestratorRoute:
    if orchestrator_input["intent"] == "small_talk":
        return {
            "action": "respond_directly",
            "intent": orchestrator_input["intent"],
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": [],
            "direct_response": _build_direct_response(),
        }

    missing_fields: list[str] = []
    if not orchestrator_input.get("movie"):
        missing_fields.append("movie")
    if not orchestrator_input.get("territory"):
        missing_fields.append("territory")

    if missing_fields:
        return {
            "action": "ask_clarification",
            "intent": orchestrator_input["intent"],
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": missing_fields,
            "direct_response": _build_clarification(missing_fields),
        }

    return {
        "action": "run_workflow",
        "intent": orchestrator_input["intent"],
        "movie": orchestrator_input["movie"],
        "territory": orchestrator_input["territory"],
        "missing_fields": [],
        "direct_response": None,
    }


def build_evidence_request(orchestrator_input: OrchestratorInput) -> EvidenceRequest:
    intent = orchestrator_input["intent"]
    movie = orchestrator_input.get("movie")
    territory = orchestrator_input.get("territory")
    if movie is None or territory is None:
        raise ValueError("Workflow requires both movie and territory")
    return {
        "movie": movie,
        "territory": territory,
        "intent": intent,
        "needs_docs": True,
        "needs_db": intent in {"valuation", "strategy", "full_scorecard"},
    }


def context_matches(orchestrator_input: OrchestratorInput, session_state: dict[str, Any]) -> bool:
    current_movie = orchestrator_input.get("movie")
    current_territory = orchestrator_input.get("territory")
    if current_movie is None or current_territory is None:
        return False

    previous_context = session_state.get("resolved_context")
    if not isinstance(previous_context, dict):
        return False

    prev_movie = str(previous_context.get("movie", "")).strip()
    prev_territory = str(previous_context.get("territory", "")).strip()
    return bool(prev_movie and prev_territory) and _normalize(prev_movie) == _normalize(
        current_movie
    ) and _normalize(prev_territory) == _normalize(current_territory)


def session_dict(session_state: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = session_state.get(key)
    return value if isinstance(value, dict) else None


def session_risk(session_state: dict[str, Any]) -> list[RiskFlag] | None:
    value = session_state.get("risk")
    if not isinstance(value, list):
        return None
    return value


async def run_data_agent(request: EvidenceRequest) -> EvidenceBundle:
    return await DataAgent.run(request)


async def run_risk_agent(evidence: EvidenceBundle) -> list[RiskFlag]:
    return await RiskAgent.run(evidence)


async def run_valuation_agent(evidence: EvidenceBundle, risk_flags: list[RiskFlag]) -> ValuationResult:
    return await ValuationAgent.run(evidence, risk_flags)


async def run_strategy_agent(
    orchestrator_input: OrchestratorInput,
    evidence: EvidenceBundle,
    valuation: ValuationResult,
    risk_flags: list[RiskFlag],
) -> StrategyResult:
    return await StrategyAgent.run(orchestrator_input, evidence, valuation, risk_flags)


def run_validation(
    evidence: EvidenceBundle,
    valuation: ValuationResult,
    confidence: float,
    provider_enabled: bool,
) -> ValidationReport:
    warnings: list[str] = []
    if not provider_enabled:
        warnings.append("Model provider key not configured; using deterministic orchestrator path.")
    if evidence["data_sufficiency_score"] < 0.55:
        warnings.append("Data sufficiency is low for this territory/movie combination.")

    report: ValidationReport = {
        "financial_sanity_pass": financial_sanity_check(
            valuation["mg_estimate_usd"],
            valuation["theatrical_projection_usd"],
            valuation["vod_projection_usd"],
        ),
        "hallucination_pass": hallucination_check(evidence["citations"]),
        "confidence_threshold_pass": confidence_threshold_check(confidence),
        "warnings": warnings,
    }
    return report


async def run_marketlogic_orchestrator(
    message: str,
    session_state: dict[str, Any],
    provider_enabled: bool,
) -> tuple[Scorecard | str, dict[str, Any]]:
    orchestrator_input = resolve_orchestrator_input(message=message, session_state=session_state)
    route = decide_action(orchestrator_input)

    logger.debug(
        "orchestrator_route action={} movie={} territory={} intent={} scenario={}",
        route["action"],
        orchestrator_input.get("movie") or "none",
        orchestrator_input.get("territory") or "none",
        orchestrator_input["intent"],
        orchestrator_input.get("scenario_override") or "none",
    )

    if route["action"] != "run_workflow":
        response_text = route["direct_response"] or "Please share your request in more detail."
        return response_text, {"last_agent_response_type": route["action"]}

    same_context = context_matches(orchestrator_input, session_state)
    strategy_followup = same_context and orchestrator_input.get("scenario_override") is not None

    previous_evidence = session_dict(session_state, "evidence_bundle")
    previous_risk = session_risk(session_state)
    previous_valuation = session_dict(session_state, "valuation")

    if strategy_followup and previous_evidence is not None:
        logger.debug(
            "orchestrator_reuse_evidence movie={} territory={}",
            orchestrator_input["movie"],
            orchestrator_input["territory"],
        )
        evidence: EvidenceBundle = previous_evidence  # type: ignore[assignment]
    else:
        evidence_request = build_evidence_request(orchestrator_input)
        evidence = await run_data_agent(evidence_request)

    if strategy_followup and previous_risk is not None:
        logger.debug(
            "orchestrator_reuse_risk movie={} territory={}",
            orchestrator_input["movie"],
            orchestrator_input["territory"],
        )
        risk_flags = previous_risk
    else:
        risk_flags = await run_risk_agent(evidence)

    if strategy_followup and previous_valuation is not None:
        logger.debug(
            "orchestrator_reuse_valuation movie={} territory={}",
            orchestrator_input["movie"],
            orchestrator_input["territory"],
        )
        valuation: ValuationResult = previous_valuation  # type: ignore[assignment]
    else:
        valuation = await run_valuation_agent(evidence=evidence, risk_flags=risk_flags)

    strategy = await run_strategy_agent(orchestrator_input, evidence, valuation, risk_flags)

    exchange = evidence.get("db_evidence", {}).get("exchange_rates", {})
    acquisition_local = exchange_rate_tool(
        amount_usd=valuation["mg_estimate_usd"],
        rate_to_usd=float(exchange.get("rate_to_usd", 1.0)),
    )

    confidence = round((valuation["sufficiency_score"] + evidence["data_sufficiency_score"]) / 2.0, 3)
    validation = run_validation(
        evidence=evidence,
        valuation=valuation,
        confidence=confidence,
        provider_enabled=provider_enabled,
    )

    warnings = combine_validation_warnings(validation)
    territory = orchestrator_input.get("territory") or "Unknown"
    scorecard = format_scorecard(
        territory=territory,
        theatrical_projection_usd=valuation["theatrical_projection_usd"],
        vod_projection_usd=valuation["vod_projection_usd"],
        acquisition_price_usd=valuation["mg_estimate_usd"],
        release_mode=strategy["release_mode"],
        release_window_days=strategy["release_window_days"],
        risk_flags=risk_flags,
        citations=evidence["citations"],
        confidence=confidence,
        warnings=warnings,
    )

    state_delta = {
        "resolved_context": {
            "movie": orchestrator_input.get("movie"),
            "territory": territory,
            "intent": orchestrator_input["intent"],
            "scenario_override": orchestrator_input.get("scenario_override"),
        },
        "evidence_bundle": evidence,
        "valuation": valuation,
        "risk": risk_flags,
        "strategy": strategy,
        "last_scorecard": scorecard,
        "recommended_acquisition_local": {
            "currency": exchange.get("currency_code", "USD"),
            "amount": acquisition_local,
        },
        "last_agent_response_type": "run_workflow",
    }

    return scorecard, state_delta
