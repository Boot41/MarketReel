from __future__ import annotations

from typing import Any, Literal, TypedDict


TurnType = Literal[
    "greeting",
    "acknowledgement",
    "help",
    "clarification",
    "workflow_request",
    "workflow_followup",
]
WorkflowIntent = Literal["valuation", "risk", "strategy", "full_scorecard"]
OrchestratorAction = Literal["respond_directly", "ask_clarification", "run_workflow"]
ResponseType = Literal["conversation_response", "clarification_response", "scorecard_response"]
RiskCategory = Literal["CENSORSHIP", "CULTURAL_SENSITIVITY", "MARKET"]
RiskSeverity = Literal["LOW", "MEDIUM", "HIGH"]


class OrchestratorInput(TypedDict):
    message: str
    movie: str | None
    territory: str | None
    turn_type: TurnType
    workflow_intent: WorkflowIntent | None
    scenario_override: str | None


class OrchestratorRoute(TypedDict):
    action: OrchestratorAction
    turn_type: TurnType
    workflow_intent: WorkflowIntent | None
    movie: str | None
    territory: str | None
    missing_fields: list[str]
    required_stages: list[str]
    response_type: ResponseType
    direct_response: str | None


class Citation(TypedDict):
    source_path: str
    doc_id: str
    page: int | None
    excerpt: str


class EvidenceRequest(TypedDict):
    movie: str
    territory: str
    intent: WorkflowIntent
    needs_docs: bool
    needs_db: bool


class EvidenceBundle(TypedDict):
    movie: str
    territory: str
    intent: WorkflowIntent
    document_evidence: dict[str, list[dict[str, Any]]]
    db_evidence: dict[str, Any]
    citations: list[Citation]
    data_sufficiency_score: float
    tool_diagnostics: list[dict[str, Any]]
    tool_failure_count: int


class ValuationResult(TypedDict):
    mg_estimate_usd: float
    confidence_interval_low_usd: float
    confidence_interval_high_usd: float
    theatrical_projection_usd: float
    vod_projection_usd: float
    comparable_films: list[str]
    sufficiency_score: float


class RiskFlag(TypedDict):
    category: RiskCategory
    severity: RiskSeverity
    scene_ref: str
    source_ref: str
    mitigation: str
    confidence: float


class StrategyResult(TypedDict):
    release_mode: Literal["theatrical_first", "streaming_first"]
    release_window_days: int
    marketing_spend_usd: float
    platform_priority: list[str]
    roi_scenarios: dict[str, float]


class ValidationReport(TypedDict):
    financial_sanity_pass: bool
    hallucination_pass: bool
    confidence_threshold_pass: bool
    warnings: list[str]


class DegradedMode(TypedDict):
    enabled: bool
    reason_code: str | None


class Scorecard(TypedDict):
    projected_revenue_by_territory: dict[str, float]
    risk_flags: list[RiskFlag]
    recommended_acquisition_price: float
    release_timeline: dict[str, Any]
    marketing_spend_usd: float
    platform_priority: list[str]
    roi_scenarios: dict[str, float]
    citations: list[Citation]
    confidence: float
    warnings: list[str]
    evidence_basis: Literal["grounded", "benchmark_derived"]
    degraded_mode: DegradedMode
    response_type: ResponseType
