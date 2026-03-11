# 01 MarketLogicOrchestrator Spec

## Objective
Specify top-level orchestration for intent resolution, routing, validation, and final scorecard assembly.

## Scope
In: message classification, entity resolution, invocation order, state delta writes.
Out: low-level retrieval SQL and per-tool internals.

## Inputs/Outputs
- Input: `message`, `session_state`, `provider_enabled`.
- Output: `Scorecard`, `state_delta`.

## Interfaces/Contracts
- Input contract: `OrchestratorInput` with `movie`, `territory`, `intent`, `scenario_override`.
- Downstream contracts: `EvidenceBundle`, `ValuationResult`, `RiskFlag[]`, `StrategyResult`, `ValidationReport`.
- Output contract: `Scorecard` per canonical schema.

## Control Flow
1. Resolve movie/territory using known index inventory and prior session context fallback.
2. Classify intent (`valuation`, `risk`, `strategy`, `full_scorecard`).
3. Build `EvidenceRequest` and invoke DataAgent.
4. Run Risk + Valuation (parallel-capable), then Strategy synthesis.
5. Execute validation checks and warning merge.
6. Format final scorecard and persist state delta.

## Failure Modes / Fallbacks
- Unknown movie/territory: fallback to prior session context defaults.
- Low evidence sufficiency: continue with warning and reduced confidence.
- Missing provider key: deterministic path warning added.

## Acceptance Criteria
- Deterministic orchestration order.
- Always returns scorecard JSON-ready object.
- State delta includes resolved context and intermediate outputs.

## Test Cases
- Valuation prompt with explicit movie/territory.
- Risk prompt with implicit territory from prior turn.
- Follow-up scenario override (`streaming_first`).
