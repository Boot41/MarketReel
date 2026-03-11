# MarketLogicOrchestrator Build Plan (Phased MVP, ADK Workflow)

## Summary
Create `MarketLogicOrchestrator` as the new ADK root agent by replacing the current root in `adk-server/agents/marketlogic/agent.py`.
Use Google ADK workflow composition with specialized agents (`DataAgent`, `ValuationAgent`, `RiskAgent`, `StrategyAgent`) and keep `/v1/run` unchanged, returning strict scorecard JSON serialized in `reply`.

## Implementation Changes

1. Root Orchestrator Replacement
- Replace existing `root_agent` with `MarketLogicOrchestrator`.
- Workflow order:
  1. intent + entity resolution (movie, territory, scenario)
  2. `DataAgent` evidence assembly
  3. `ValuationAgent` + `RiskAgent` in parallel
  4. `StrategyAgent` synthesis
  5. validation checks (`financial_sanity_check`, `hallucination_check`, `confidence_threshold_check`)
  6. `format_scorecard` + JSON serialization for `reply`.

2. Typed Internal Contracts
- Define typed payloads for:
  - `OrchestratorInput`
  - `EvidenceRequest`
  - `EvidenceBundle`
  - `ValuationResult`
  - `RiskFlag`
  - `StrategyResult`
  - `ValidationReport`
  - `Scorecard`.
- `RiskFlag` fields: `category`, `severity`, `scene_ref`, `source_ref`, `mitigation`, `confidence`.
- `Scorecard` fields: `projected_revenue_by_territory`, `risk_flags`, `recommended_acquisition_price`, `release_timeline`, `citations`, `confidence`, `warnings`.

3. Tooling and Data Boundaries
- Implement tools in `adk-server/agents/marketlogic/tools.py`:
  - Document tools: `IndexRegistry`, `IndexNavigator`, `TargetedFetcher`, `SufficiencyChecker`.
  - DB tools: box office by genre/territory, actor Q-score/social reach, theatrical windows, exchange rates, VOD benchmarks, comparable films.
  - Utility/validation tools: `source_citation_tool`, `financial_sanity_check`, `hallucination_check`, `confidence_threshold_check`, `format_scorecard`.
- Enforce architecture rule: only `DataAgent` can access DB/doc tools directly.

4. Session State + Follow-up Behavior
- Persist in ADK session state:
  - resolved context
  - evidence bundle
  - valuation/risk/strategy outputs
  - last scorecard.
- Follow-up queries should detect scenario deltas, reuse prior artifacts, and fetch only missing evidence.

5. API/Runtime Compatibility
- Keep `adk-server/app/main.py` interface and `X-ADK-API-Key` auth unchanged.
- Preserve no-provider fallback behavior with session create/reuse.

## Assumptions and Defaults
- Scope: phased MVP (not full production-depth quant logic in one pass).
- ADK pattern: workflow agents.
- Output contract: strict JSON string in `reply`.
- No backend API/schema changes in this phase.

## Evals (Added At End)
1. Add eval suite last, after orchestrator + tools are stable.
- File: `adk-server/agents/eval/test_eval.py`.

2. Required eval scenarios
- Valuation query: “How will Interstellar perform in India?” returns valid scorecard JSON with citations.
- Censorship query: “Does Deadpool face censorship issues in Saudi Arabia?” returns non-empty `RiskFlag[]` with source refs.
- Strategy what-if follow-up: “If we skip theatrical in Germany, how does ROI change?” reuses session context and updates strategy outputs.
- Low-sufficiency path: missing/weak evidence triggers warnings and lower confidence.

3. Validation assertions in evals
- Scorecard schema compliance.
- Citation presence for each major claim bucket.
- Confidence threshold behavior.
- Deterministic fallback behavior when provider credentials are absent.
