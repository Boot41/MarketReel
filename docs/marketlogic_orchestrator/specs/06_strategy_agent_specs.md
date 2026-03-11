# 06 StrategyAgent Spec

## Objective
Specify release strategy synthesis from valuation, risks, sentiment cues, and scenario overrides.

## Scope
In: release mode, window, marketing spend, platform priority, ROI scenarios.
Out: scorecard formatting and API serialization.

## Inputs/Outputs
- Input: `OrchestratorInput`, `EvidenceBundle`, `ValuationResult`, `RiskFlag[]`.
- Output: `StrategyResult`.

## Interfaces/Contracts
- Supports scenario overrides: `streaming_first`, `theatrical_first`.
- Returns `release_mode`, `release_window_days`, `marketing_spend_usd`, `platform_priority`, `roi_scenarios`.

## Control Flow
1. Resolve release window from territory trends.
2. Determine base release mode from risk profile unless overridden.
3. Compute mode-adjusted projection behavior.
4. Estimate marketing spend and ROI scenarios.
5. Return typed strategy recommendation.

## Failure Modes / Fallbacks
- No window trends: use default window.
- Missing strategy evidence: conservative spend + explicit warning route.
- ROI instability: clamp denominator and return bounded values.

## Acceptance Criteria
- Override logic is deterministic and respected.
- ROI scenarios always include base and streaming-first keys.
- Output is directly consumable by scorecard formatter.

## Test Cases
- High-risk auto streaming-first behavior.
- Explicit theatrical override behavior.
- Sparse-data strategy fallback.
