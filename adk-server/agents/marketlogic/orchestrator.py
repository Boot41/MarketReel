from __future__ import annotations

import json
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
    OrchestratorInput,
    OrchestratorRoute,
    ResponseType,
    RiskFlag,
    Scorecard,
    StrategyResult,
    ValidationReport,
    ValuationResult,
    WorkflowIntent,
)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _contains_any(value: str, tokens: list[str]) -> bool:
    return any(token in value for token in tokens)


def _is_greeting(message: str) -> bool:
    msg = _normalize(message)
    exact = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    }
    if msg in exact:
        return True
    return any(msg.startswith(prefix) for prefix in ["hi ", "hello ", "hey "])


def _is_acknowledgement(message: str) -> bool:
    msg = _normalize(message)
    if msg in {"thanks", "thank you", "ok", "okay", "got it", "cool"}:
        return True
    return msg.startswith("thanks ") or msg.startswith("thank you ")


def _is_help(message: str) -> bool:
    msg = _normalize(message)
    return _contains_any(msg, ["help", "what can you do", "capabilities", "how to use"])


def _is_clarification_turn(message: str) -> bool:
    msg = _normalize(message)
    return _contains_any(msg, ["clarify", "not clear", "what do you need", "which movie"])


def _is_explainability_request(message: str) -> bool:
    msg = _normalize(message)
    return _contains_any(
        msg,
        [
            "why this",
            "explain",
            "how did you",
            "show evidence",
            "show citations",
            "citations",
            "sources",
            "reasoning",
        ],
    )


def _is_followup_hint(message: str) -> bool:
    msg = _normalize(message)
    return _contains_any(
        msg,
        [
            "what if",
            "if we",
            "how does",
            "instead",
            "compare",
            "versus",
            "vs",
            "same movie",
            "same territory",
        ],
    )


def _detect_turn_type(message: str, session_state: dict[str, Any]) -> str:
    has_context = isinstance(session_state.get("resolved_context"), dict)
    if _is_greeting(message):
        return "greeting"
    if _is_acknowledgement(message):
        return "acknowledgement"
    if _is_help(message):
        return "help"
    if _is_clarification_turn(message):
        return "clarification"
    if has_context and (_is_explainability_request(message) or _is_followup_hint(message)):
        return "workflow_followup"
    return "workflow_request"


def _resolve_workflow_intent(message: str) -> WorkflowIntent:
    msg = _normalize(message)
    if _contains_any(msg, ["censor", "sensitivity", "risk", "ban", "edit"]):
        return "risk"
    if _contains_any(msg, ["mg", "minimum guarantee", "price", "valuation", "pay"]):
        return "valuation"
    if _contains_any(msg, ["release", "window", "marketing", "streaming", "theatrical", "roi"]):
        return "strategy"
    if _contains_any(msg, ["acquire", "acquisition", "deal", "recommend", "scorecard", "evaluate"]):
        return "full_scorecard"
    return "full_scorecard"


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


def _build_help_response() -> str:
    return (
        "I can evaluate a film by territory across valuation, risk, and release strategy. "
        "Share a movie and territory, for example: 'Evaluate Interstellar for India'."
    )


def _build_clarification(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "Please share the movie and territory to continue."
    if len(missing_fields) == 2:
        return "Please share both the movie title and target territory to run analysis."
    if missing_fields[0] == "movie":
        return "Please share the movie title so I can continue the analysis."
    return "Please share the target territory so I can continue the analysis."


def _conversation_payload(message: str, response_type: ResponseType) -> dict[str, Any]:
    return {
        "response_type": response_type,
        "message": message,
    }


def _diagnostic_payload(
    *,
    message: str,
    reason_code: str,
    missing_requirements: list[str],
    next_action: str,
) -> dict[str, Any]:
    payload = _conversation_payload(message, "clarification_response")
    payload["reason_code"] = reason_code
    payload["missing_requirements"] = missing_requirements
    payload["next_action"] = next_action
    return payload


def _build_explainability_payload(session_state: dict[str, Any]) -> dict[str, Any]:
    last_scorecard = session_state.get("last_scorecard")
    evidence_bundle = session_state.get("evidence_bundle")
    if not isinstance(last_scorecard, dict) or not isinstance(evidence_bundle, dict):
        return _conversation_payload(
            "I do not have prior analytical artifacts in this session yet. Run an analysis first.",
            "clarification_response",
        )

    citations = last_scorecard.get("citations", [])
    top_citations = citations[:3] if isinstance(citations, list) else []
    confidence = last_scorecard.get("confidence")
    warnings = last_scorecard.get("warnings", [])

    return {
        "response_type": "conversation_response",
        "message": "Here is the evidence summary from the latest analysis in this session.",
        "explainability": {
            "confidence": confidence,
            "warnings": warnings if isinstance(warnings, list) else [],
            "top_citations": top_citations,
            "data_sufficiency_score": evidence_bundle.get("data_sufficiency_score", 0.0),
        },
    }


def _required_stages_for_intent(workflow_intent: WorkflowIntent) -> list[str]:
    if workflow_intent == "risk":
        return ["data", "risk"]
    if workflow_intent == "valuation":
        return ["data", "valuation"]
    return ["data", "risk", "valuation", "strategy"]


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

    turn_type = _detect_turn_type(message, session_state)
    workflow_intent = _resolve_workflow_intent(message) if turn_type.startswith("workflow_") else None

    return {
        "message": message,
        "movie": movie,
        "territory": territory,
        "turn_type": turn_type,
        "workflow_intent": workflow_intent,
        "scenario_override": _detect_scenario_override(message),
    }


def decide_action(orchestrator_input: OrchestratorInput, session_state: dict[str, Any]) -> OrchestratorRoute:
    turn_type = orchestrator_input["turn_type"]

    if turn_type in {"greeting", "acknowledgement", "help"}:
        return {
            "action": "respond_directly",
            "turn_type": turn_type,
            "workflow_intent": None,
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": [],
            "required_stages": [],
            "response_type": "conversation_response",
            "direct_response": _build_help_response(),
        }

    if turn_type == "clarification":
        return {
            "action": "ask_clarification",
            "turn_type": turn_type,
            "workflow_intent": None,
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": [],
            "required_stages": [],
            "response_type": "clarification_response",
            "direct_response": "Tell me the movie and territory you want to analyze.",
        }

    if turn_type == "workflow_followup" and _is_explainability_request(orchestrator_input["message"]):
        payload = _build_explainability_payload(session_state)
        return {
            "action": "respond_directly",
            "turn_type": turn_type,
            "workflow_intent": None,
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": [],
            "required_stages": [],
            "response_type": payload["response_type"],
            "direct_response": json.dumps(payload, ensure_ascii=True),
        }

    missing_fields: list[str] = []
    if not orchestrator_input.get("movie"):
        missing_fields.append("movie")
    if not orchestrator_input.get("territory"):
        missing_fields.append("territory")

    if missing_fields:
        return {
            "action": "ask_clarification",
            "turn_type": turn_type,
            "workflow_intent": orchestrator_input["workflow_intent"],
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
            "missing_fields": missing_fields,
            "required_stages": [],
            "response_type": "clarification_response",
            "direct_response": _build_clarification(missing_fields),
        }

    workflow_intent = orchestrator_input.get("workflow_intent") or "full_scorecard"
    return {
        "action": "run_workflow",
        "turn_type": turn_type,
        "workflow_intent": workflow_intent,
        "movie": orchestrator_input["movie"],
        "territory": orchestrator_input["territory"],
        "missing_fields": [],
        "required_stages": _required_stages_for_intent(workflow_intent),
        "response_type": "scorecard_response",
        "direct_response": None,
    }


def build_evidence_request(orchestrator_input: OrchestratorInput) -> EvidenceRequest:
    workflow_intent = orchestrator_input.get("workflow_intent")
    movie = orchestrator_input.get("movie")
    territory = orchestrator_input.get("territory")
    if workflow_intent is None or movie is None or territory is None:
        raise ValueError("Workflow requires intent, movie, and territory")
    return {
        "movie": movie,
        "territory": territory,
        "intent": workflow_intent,
        "needs_docs": True,
        "needs_db": workflow_intent in {"valuation", "strategy", "full_scorecard"},
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


def scenario_compatible(orchestrator_input: OrchestratorInput, session_state: dict[str, Any]) -> bool:
    current = orchestrator_input.get("scenario_override")
    previous_context = session_state.get("resolved_context")
    if not isinstance(previous_context, dict):
        return True
    previous = previous_context.get("scenario_override")
    if current is None or previous is None:
        return True
    return current == previous


def session_dict(session_state: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = session_state.get(key)
    return value if isinstance(value, dict) else None


def session_risk(session_state: dict[str, Any]) -> list[RiskFlag] | None:
    value = session_state.get("risk")
    if not isinstance(value, list):
        return None
    return value


def evidence_cache_valid(evidence: dict[str, Any]) -> bool:
    return float(evidence.get("data_sufficiency_score", 0.0)) >= 0.55 and int(
        evidence.get("tool_failure_count", 0)
    ) == 0


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
    valuation: ValuationResult | None,
    confidence: float,
    provider_enabled: bool,
    include_financial_sanity: bool,
) -> ValidationReport:
    warnings: list[str] = []
    if not provider_enabled:
        warnings.append("Model provider key not configured; using deterministic orchestrator path.")
    if evidence["data_sufficiency_score"] < 0.55:
        warnings.append("Data sufficiency is low for this territory/movie combination.")

    financial_pass = True
    if include_financial_sanity:
        if valuation is None:
            financial_pass = False
        else:
            financial_pass = financial_sanity_check(
                valuation["mg_estimate_usd"],
                valuation["theatrical_projection_usd"],
                valuation["vod_projection_usd"],
            )

    report: ValidationReport = {
        "financial_sanity_pass": financial_pass,
        "hallucination_pass": hallucination_check(evidence["citations"]),
        "confidence_threshold_pass": confidence_threshold_check(confidence),
        "warnings": warnings,
    }
    return report


def _compute_confidence(
    evidence: EvidenceBundle,
    valuation: ValuationResult | None,
) -> float:
    if valuation is None:
        return round(max(0.2, evidence["data_sufficiency_score"]), 3)
    return round((valuation["sufficiency_score"] + evidence["data_sufficiency_score"]) / 2.0, 3)


def _build_scorecard(
    *,
    territory: str,
    evidence: EvidenceBundle,
    valuation: ValuationResult | None,
    risk_flags: list[RiskFlag],
    strategy: StrategyResult | None,
    warnings: list[str],
    confidence: float,
) -> Scorecard:
    theatrical_projection = valuation["theatrical_projection_usd"] if valuation else 0.0
    vod_projection = valuation["vod_projection_usd"] if valuation else 0.0
    acquisition_price = valuation["mg_estimate_usd"] if valuation else 0.0

    release_mode = strategy["release_mode"] if strategy else "theatrical_first"
    release_window_days = strategy["release_window_days"] if strategy else 0
    marketing_spend = strategy["marketing_spend_usd"] if strategy else 0.0
    platform_priority = strategy["platform_priority"] if strategy else []
    roi_scenarios = strategy["roi_scenarios"] if strategy else {}

    return format_scorecard(
        territory=territory,
        theatrical_projection_usd=theatrical_projection,
        vod_projection_usd=vod_projection,
        acquisition_price_usd=acquisition_price,
        release_mode=release_mode,
        release_window_days=release_window_days,
        marketing_spend_usd=marketing_spend,
        platform_priority=platform_priority,
        roi_scenarios=roi_scenarios,
        risk_flags=risk_flags,
        citations=evidence["citations"],
        confidence=confidence,
        warnings=warnings,
        response_type="scorecard_response",
    )


def _evidence_failure_reason(
    evidence: EvidenceBundle,
    workflow_intent: WorkflowIntent,
    *,
    allow_strategy_reuse_without_market: bool,
) -> dict[str, Any] | None:
    diagnostics = evidence.get("tool_diagnostics", [])
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        error_type = str(item.get("error_type", ""))
        if error_type == "auth":
            return {
                "reason_code": "internal_auth_failed",
                "missing_requirements": ["valid_internal_api_auth"],
                "next_action": "Verify ADK and backend internal API keys match and retry.",
            }

    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        error_type = str(item.get("error_type", ""))
        if error_type in {"network", "timeout", "backend_unavailable", "unknown"}:
            return {
                "reason_code": "backend_unavailable",
                "missing_requirements": ["reachable_backend_internal_api"],
                "next_action": "Ensure backend service is running and reachable from ADK server, then retry.",
            }

    db = evidence.get("db_evidence", {})
    box_office = db.get("box_office", {})
    comparables = db.get("comparable_films", [])
    citation_count = len(evidence.get("citations", []))
    doc_records = evidence.get("document_evidence", {}).get("documents", [])
    risk_docs = 0
    for item in doc_records:
        source = str(item.get("source_path", ""))
        if "censorship" in source or "cultural_sensitivity" in source:
            risk_docs += 1

    market_signal = int(box_office.get("samples", 0)) > 0 or len(comparables) > 0

    missing: list[str] = []
    if workflow_intent == "valuation":
        if not market_signal:
            missing.append("market_signals")
    elif workflow_intent == "risk":
        if risk_docs == 0 and citation_count < 2:
            missing.append("risk_evidence")
    else:
        if not market_signal:
            missing.append("market_signals")
        if citation_count < 3:
            missing.append("citations")
        if float(evidence.get("data_sufficiency_score", 0.0)) < 0.55:
            missing.append("data_sufficiency")
        if workflow_intent == "strategy" and allow_strategy_reuse_without_market:
            missing = [item for item in missing if item not in {"market_signals", "citations"}]

    if missing:
        return {
            "reason_code": "insufficient_evidence",
            "missing_requirements": missing,
            "next_action": "Provide stronger movie/territory context or retry after data services are available.",
        }
    return None


async def run_marketlogic_orchestrator(
    message: str,
    session_state: dict[str, Any],
    provider_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    orchestrator_input = resolve_orchestrator_input(message=message, session_state=session_state)
    route = decide_action(orchestrator_input, session_state)

    logger.debug(
        "orchestrator_route action={} turn_type={} movie={} territory={} intent={} scenario={}",
        route["action"],
        route["turn_type"],
        orchestrator_input.get("movie") or "none",
        orchestrator_input.get("territory") or "none",
        route.get("workflow_intent") or "none",
        orchestrator_input.get("scenario_override") or "none",
    )

    if route["action"] != "run_workflow":
        response_text = route["direct_response"] or json.dumps(
            _conversation_payload("Please share your request in more detail.", route["response_type"]),
            ensure_ascii=True,
        )
        if response_text.startswith("{"):
            payload = json.loads(response_text)
        else:
            payload = _conversation_payload(response_text, route["response_type"])
        return payload, {"last_agent_response_type": route["response_type"]}

    required_stages = set(route["required_stages"])
    workflow_intent = route["workflow_intent"]
    if workflow_intent is None:
        raise ValueError("Workflow route requires workflow intent")

    same_context = context_matches(orchestrator_input, session_state)
    compatible_scenario = scenario_compatible(orchestrator_input, session_state)
    is_followup = route["turn_type"] == "workflow_followup"

    previous_evidence = session_dict(session_state, "evidence_bundle")
    previous_risk = session_risk(session_state)
    previous_valuation = session_dict(session_state, "valuation")

    reuse_evidence = (
        is_followup
        and same_context
        and compatible_scenario
        and previous_evidence is not None
        and evidence_cache_valid(previous_evidence)
    )

    if reuse_evidence:
        logger.debug(
            "orchestrator_reuse_evidence movie={} territory={}",
            orchestrator_input["movie"],
            orchestrator_input["territory"],
        )
        evidence: EvidenceBundle = previous_evidence  # type: ignore[assignment]
    else:
        evidence_request = build_evidence_request(orchestrator_input)
        evidence = await run_data_agent(evidence_request)

    evidence_failure = _evidence_failure_reason(
        evidence,
        workflow_intent,
        allow_strategy_reuse_without_market=bool(previous_valuation and is_followup and same_context),
    )
    if evidence_failure is not None:
        logger.warning(
            "orchestrator_evidence_blocked movie={} territory={} intent={} reason={}",
            orchestrator_input.get("movie") or "none",
            orchestrator_input.get("territory") or "none",
            workflow_intent,
            evidence_failure["reason_code"],
        )
        message_text = (
            "I could not generate a reliable scorecard because required evidence was unavailable."
        )
        payload = _diagnostic_payload(
            message=message_text,
            reason_code=evidence_failure["reason_code"],
            missing_requirements=evidence_failure["missing_requirements"],
            next_action=evidence_failure["next_action"],
        )
        state_delta = {
            "resolved_context": {
                "movie": orchestrator_input.get("movie"),
                "territory": orchestrator_input.get("territory"),
                "workflow_intent": workflow_intent,
                "scenario_override": orchestrator_input.get("scenario_override"),
            },
            "evidence_bundle": evidence,
            "last_agent_response_type": "clarification_response",
        }
        return payload, state_delta

    risk_flags: list[RiskFlag] = []
    if "risk" in required_stages:
        reuse_risk = is_followup and same_context and previous_risk is not None and reuse_evidence
        if reuse_risk:
            logger.debug(
                "orchestrator_reuse_risk movie={} territory={}",
                orchestrator_input["movie"],
                orchestrator_input["territory"],
            )
            risk_flags = previous_risk
        else:
            risk_flags = await run_risk_agent(evidence)

    valuation: ValuationResult | None = None
    if "valuation" in required_stages:
        reuse_valuation = (
            is_followup and same_context and previous_valuation is not None and reuse_evidence
        )
        if reuse_valuation:
            logger.debug(
                "orchestrator_reuse_valuation movie={} territory={}",
                orchestrator_input["movie"],
                orchestrator_input["territory"],
            )
            valuation = previous_valuation  # type: ignore[assignment]
        else:
            valuation = await run_valuation_agent(evidence=evidence, risk_flags=risk_flags)

    strategy: StrategyResult | None = None
    if "strategy" in required_stages:
        if valuation is None:
            valuation = await run_valuation_agent(evidence=evidence, risk_flags=risk_flags)
        if not risk_flags:
            risk_flags = await run_risk_agent(evidence)
        strategy = await run_strategy_agent(orchestrator_input, evidence, valuation, risk_flags)

    confidence = _compute_confidence(evidence=evidence, valuation=valuation)
    include_financial = workflow_intent in {"valuation", "strategy", "full_scorecard"}
    validation = run_validation(
        evidence=evidence,
        valuation=valuation,
        confidence=confidence,
        provider_enabled=provider_enabled,
        include_financial_sanity=include_financial,
    )

    warnings = combine_validation_warnings(validation)
    if workflow_intent != "full_scorecard":
        warnings.append(f"Partial workflow executed: {workflow_intent}.")

    territory = orchestrator_input.get("territory") or "Unknown"
    scorecard = _build_scorecard(
        territory=territory,
        evidence=evidence,
        valuation=valuation,
        risk_flags=risk_flags,
        strategy=strategy,
        warnings=warnings,
        confidence=confidence,
    )

    exchange = evidence.get("db_evidence", {}).get("exchange_rates", {})
    acquisition_local = exchange_rate_tool(
        amount_usd=float(scorecard.get("recommended_acquisition_price", 0.0)),
        rate_to_usd=float(exchange.get("rate_to_usd", 1.0)),
    )

    state_delta = {
        "resolved_context": {
            "movie": orchestrator_input.get("movie"),
            "territory": territory,
            "workflow_intent": workflow_intent,
            "scenario_override": orchestrator_input.get("scenario_override"),
        },
        "evidence_bundle": evidence,
        "valuation": valuation or {},
        "risk": risk_flags,
        "strategy": strategy or {},
        "last_scorecard": scorecard,
        "recommended_acquisition_local": {
            "currency": exchange.get("currency_code", "USD"),
            "amount": acquisition_local,
        },
        "last_agent_response_type": "scorecard_response",
    }

    return scorecard, state_delta
