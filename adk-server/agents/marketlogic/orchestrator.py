from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .tools import (
    IndexNavigator,
    IndexRegistry,
    SufficiencyChecker,
    TargetedFetcher,
    combine_validation_warnings,
    confidence_threshold_check,
    exchange_rate_tool,
    financial_sanity_check,
    format_scorecard,
    get_actor_qscore,
    get_box_office_by_genre_territory,
    get_comparable_films,
    get_exchange_rates,
    get_theatrical_window_trends,
    get_vod_price_benchmarks,
    hallucination_check,
    mg_calculator_tool,
    source_citation_tool,
)
from .types import (
    EvidenceBundle,
    EvidenceRequest,
    IntentType,
    OrchestratorInput,
    RiskFlag,
    Scorecard,
    StrategyResult,
    ValidationReport,
    ValuationResult,
)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _classify_intent(message: str) -> IntentType:
    msg = _normalize(message)
    if any(token in msg for token in ["censor", "sensitivity", "risk", "ban", "edit"]):
        return "risk"
    if any(token in msg for token in ["mg", "minimum guarantee", "price", "valuation", "pay"]):
        return "valuation"
    if any(token in msg for token in ["release", "window", "marketing", "streaming", "theatrical", "roi"]):
        return "strategy"
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


def resolve_orchestrator_input(message: str, session_state: dict[str, Any]) -> OrchestratorInput:
    registry = IndexRegistry()
    known_movies = list(registry.get("known_movies", []))
    known_territories = list(registry.get("known_territories", []))

    movie = _match_entity(message, known_movies)
    territory = _match_entity(message, known_territories)

    previous_context = session_state.get("resolved_context", {})
    if not movie:
        movie = str(previous_context.get("movie") or "Interstellar")
    if not territory:
        territory = str(previous_context.get("territory") or "India")

    return {
        "message": message,
        "movie": movie,
        "territory": territory,
        "intent": _classify_intent(message),
        "scenario_override": _detect_scenario_override(message),
    }


def _build_evidence_request(orchestrator_input: OrchestratorInput) -> EvidenceRequest:
    intent = orchestrator_input["intent"]
    return {
        "movie": orchestrator_input["movie"],
        "territory": orchestrator_input["territory"],
        "intent": intent,
        "needs_docs": True,
        "needs_db": intent in {"valuation", "strategy", "full_scorecard"},
    }


def _expand_retrieval_plan(plan: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(plan)
    max_docs = int(plan.get("max_docs", 10))
    max_scenes = int(plan.get("max_scenes", 6))
    expanded["max_docs"] = min(20, max_docs + 6)
    expanded["max_scenes"] = min(12, max_scenes + 4)
    return expanded


async def run_data_agent(request: EvidenceRequest) -> EvidenceBundle:
    movie = request["movie"]
    territory = request["territory"]

    fetched: dict[str, list[dict[str, Any]]] = {"documents": [], "scenes": []}
    sufficiency: dict[str, Any] = {"status": "EXPAND", "score": 0.0}
    if request["needs_docs"]:
        plan = IndexNavigator(movie=movie, territory=territory, intent=request["intent"])
        fetched = TargetedFetcher(plan)
        sufficiency = SufficiencyChecker(fetched)
        if sufficiency.get("status") != "PASS":
            expanded_plan = _expand_retrieval_plan(plan)
            expanded_fetch = TargetedFetcher(expanded_plan)
            expanded_sufficiency = SufficiencyChecker(expanded_fetch)
            if float(expanded_sufficiency.get("score", 0.0)) >= float(sufficiency.get("score", 0.0)):
                fetched = expanded_fetch
                sufficiency = expanded_sufficiency

    all_records = fetched.get("documents", []) + fetched.get("scenes", [])
    citations = source_citation_tool(all_records)

    db_evidence: dict[str, Any] = {}
    if request["needs_db"]:
        box_office, qscore, windows, fx, vod, comparables = await asyncio.gather(
            get_box_office_by_genre_territory(movie, territory),
            get_actor_qscore(movie),
            get_theatrical_window_trends(territory),
            get_exchange_rates(territory),
            get_vod_price_benchmarks(territory),
            get_comparable_films(movie, territory),
        )
        db_evidence = {
            "box_office": box_office,
            "actor_signals": qscore,
            "theatrical_windows": windows,
            "exchange_rates": fx,
            "vod_benchmarks": vod,
            "comparable_films": comparables,
        }

    grouped_documents: dict[str, list[dict[str, Any]]] = {
        "documents": fetched.get("documents", []),
        "scenes": fetched.get("scenes", []),
    }

    return {
        "movie": movie,
        "territory": territory,
        "intent": request["intent"],
        "document_evidence": grouped_documents,
        "db_evidence": db_evidence,
        "citations": citations,
        "data_sufficiency_score": float(sufficiency.get("score", 0.0)),
    }


def _context_matches(orchestrator_input: OrchestratorInput, session_state: dict[str, Any]) -> bool:
    previous_context = session_state.get("resolved_context")
    if not isinstance(previous_context, dict):
        return False
    prev_movie = str(previous_context.get("movie", "")).strip()
    prev_territory = str(previous_context.get("territory", "")).strip()
    return bool(prev_movie and prev_territory) and _normalize(prev_movie) == _normalize(
        orchestrator_input["movie"]
    ) and _normalize(prev_territory) == _normalize(orchestrator_input["territory"])


def _session_dict(session_state: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = session_state.get(key)
    return value if isinstance(value, dict) else None


def _session_risk(session_state: dict[str, Any]) -> list[RiskFlag] | None:
    value = session_state.get("risk")
    if not isinstance(value, list):
        return None
    return value


def _estimate_risk_penalty(risk_flags: list[RiskFlag]) -> float:
    if not risk_flags:
        return 0.05
    penalty = 0.0
    for flag in risk_flags:
        severity = flag["severity"]
        if severity == "HIGH":
            penalty += 0.18
        elif severity == "MEDIUM":
            penalty += 0.08
        else:
            penalty += 0.03
    return min(0.6, penalty)


async def run_valuation_agent(evidence: EvidenceBundle, risk_flags: list[RiskFlag]) -> ValuationResult:
    db = evidence.get("db_evidence", {})
    box_office = db.get("box_office", {})
    actor_signals = db.get("actor_signals", {})
    comparables = db.get("comparable_films", [])

    comparable_values = [float(item.get("territory_gross_usd", 0.0)) for item in comparables]
    comparable_avg = sum(comparable_values) / len(comparable_values) if comparable_values else 0.0

    mg_estimate_usd = mg_calculator_tool(
        avg_box_office_usd=float(box_office.get("avg_gross_usd", 0.0)),
        avg_qscore=float(actor_signals.get("avg_qscore", 0.0)),
        comparable_avg_gross_usd=float(comparable_avg),
        risk_penalty=_estimate_risk_penalty(risk_flags),
    )

    theatrical_projection = max(
        mg_estimate_usd * 2.4,
        float(box_office.get("avg_gross_usd", 0.0)) * 0.75,
    )

    vod = db.get("vod_benchmarks", {})
    vod_projection = max(
        mg_estimate_usd * 0.7,
        float(vod.get("avg_price_max_usd", 0.0)) * 1.1,
    )

    confidence = max(0.25, min(0.95, evidence["data_sufficiency_score"] * 0.9))
    interval_low = mg_estimate_usd * (0.8 - (1.0 - confidence) * 0.15)
    interval_high = mg_estimate_usd * (1.2 + (1.0 - confidence) * 0.2)

    return {
        "mg_estimate_usd": round(mg_estimate_usd, 2),
        "confidence_interval_low_usd": round(interval_low, 2),
        "confidence_interval_high_usd": round(interval_high, 2),
        "theatrical_projection_usd": round(theatrical_projection, 2),
        "vod_projection_usd": round(vod_projection, 2),
        "comparable_films": [str(item.get("title", "")) for item in comparables if item.get("title")],
        "sufficiency_score": round(confidence, 3),
    }


def _risk_from_text(territory: str, text: str) -> tuple[str, str] | None:
    row = _normalize(text)
    territory_norm = _normalize(territory)
    if territory_norm not in row:
        return None
    if "high" in row:
        return "HIGH", "Content likely needs significant edits or alternative distribution mode."
    if "medium" in row:
        return "MEDIUM", "Apply territory-specific edit plan and pre-clear with local legal/distribution."
    if "low" in row:
        return "LOW", "Standard territory compliance process should be sufficient."
    return None


async def run_risk_agent(evidence: EvidenceBundle) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    territory = evidence["territory"]

    for item in evidence["document_evidence"].get("documents", []):
        text = str(item.get("text", ""))
        source = str(item.get("source_path", ""))
        if "censorship" in source:
            detected = _risk_from_text(territory, text)
            if detected:
                severity, mitigation = detected
                flags.append(
                    {
                        "category": "CENSORSHIP",
                        "severity": severity,
                        "scene_ref": str(item.get("doc_id", "unknown")),
                        "source_ref": source,
                        "mitigation": mitigation,
                        "confidence": 0.8 if severity != "LOW" else 0.65,
                    }
                )
        if "cultural_sensitivity" in source and _normalize(territory) in _normalize(text):
            flags.append(
                {
                    "category": "CULTURAL_SENSITIVITY",
                    "severity": "MEDIUM",
                    "scene_ref": str(item.get("doc_id", "unknown")),
                    "source_ref": source,
                    "mitigation": "Localize campaign and trailer cut with culturally aligned messaging.",
                    "confidence": 0.62,
                }
            )

    if not flags:
        flags.append(
            {
                "category": "MARKET",
                "severity": "LOW",
                "scene_ref": "market_baseline",
                "source_ref": "derived:insufficient-risk-signal",
                "mitigation": "Proceed with baseline compliance and territory pre-screening.",
                "confidence": 0.45,
            }
        )

    return flags


async def run_strategy_agent(
    orchestrator_input: OrchestratorInput,
    evidence: EvidenceBundle,
    valuation: ValuationResult,
    risk_flags: list[RiskFlag],
) -> StrategyResult:
    db = evidence.get("db_evidence", {})
    windows = db.get("theatrical_windows", [])
    release_window = 45
    if windows:
        release_window = max(14, min(90, int(windows[0].get("days", 45))))

    high_risk = any(flag["severity"] == "HIGH" for flag in risk_flags)
    scenario_override = orchestrator_input.get("scenario_override")
    if scenario_override == "streaming_first":
        release_mode = "streaming_first"
    elif scenario_override == "theatrical_first":
        release_mode = "theatrical_first"
    else:
        release_mode = "streaming_first" if high_risk else "theatrical_first"

    theatrical = valuation["theatrical_projection_usd"]
    vod = valuation["vod_projection_usd"]

    if release_mode == "streaming_first":
        theatrical *= 0.55
        vod *= 1.25

    marketing_spend = max(250_000.0, (theatrical + vod) * (0.12 if release_mode == "theatrical_first" else 0.08))

    roi_theatrical = ((theatrical + vod) - valuation["mg_estimate_usd"] - marketing_spend) / max(
        1.0, valuation["mg_estimate_usd"] + marketing_spend
    )
    roi_streaming = ((theatrical * 0.6 + vod * 1.2) - valuation["mg_estimate_usd"] - marketing_spend * 0.8) / max(
        1.0, valuation["mg_estimate_usd"] + marketing_spend * 0.8
    )

    return {
        "release_mode": release_mode,
        "release_window_days": int(release_window),
        "marketing_spend_usd": round(marketing_spend, 2),
        "platform_priority": ["theatrical", "premium_vod", "svod"]
        if release_mode == "theatrical_first"
        else ["svod", "premium_vod", "theatrical_limited"],
        "roi_scenarios": {
            "base": round(roi_theatrical, 3),
            "streaming_first": round(roi_streaming, 3),
        },
    }


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
) -> tuple[Scorecard, dict[str, Any]]:
    orchestrator_input = resolve_orchestrator_input(message=message, session_state=session_state)
    logger.debug(
        "orchestrator_input_resolved movie={} territory={} intent={} scenario={}",
        orchestrator_input["movie"],
        orchestrator_input["territory"],
        orchestrator_input["intent"],
        orchestrator_input.get("scenario_override") or "none",
    )

    same_context = _context_matches(orchestrator_input, session_state)
    strategy_followup = same_context and orchestrator_input.get("scenario_override") is not None

    previous_evidence = _session_dict(session_state, "evidence_bundle")
    previous_risk = _session_risk(session_state)
    previous_valuation = _session_dict(session_state, "valuation")

    if strategy_followup and previous_evidence is not None:
        logger.debug("orchestrator_reuse_evidence movie={} territory={}", orchestrator_input["movie"], orchestrator_input["territory"])
        evidence: EvidenceBundle = previous_evidence  # type: ignore[assignment]
    else:
        evidence_request = _build_evidence_request(orchestrator_input)
        evidence = await run_data_agent(evidence_request)

    if strategy_followup and previous_risk is not None:
        logger.debug("orchestrator_reuse_risk movie={} territory={}", orchestrator_input["movie"], orchestrator_input["territory"])
        risk_flags = previous_risk
    else:
        risk_flags = await run_risk_agent(evidence)

    if strategy_followup and previous_valuation is not None:
        logger.debug("orchestrator_reuse_valuation movie={} territory={}", orchestrator_input["movie"], orchestrator_input["territory"])
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
    scorecard = format_scorecard(
        territory=orchestrator_input["territory"],
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
            "movie": orchestrator_input["movie"],
            "territory": orchestrator_input["territory"],
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
    }

    return scorecard, state_delta
