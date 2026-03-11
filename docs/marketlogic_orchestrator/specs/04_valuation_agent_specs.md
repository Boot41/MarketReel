# 04 ValuationAgent Spec

## Objective
Define MG and revenue estimation logic from structured evidence + risk adjustment.

## Scope
In: MG estimate, confidence interval, theatrical/VOD projections, comparable film usage.
Out: strategy decisioning and final formatting.

## Inputs/Outputs
- Input: `EvidenceBundle`, `RiskFlag[]`.
- Output: `ValuationResult`.

## Interfaces/Contracts
- Uses `mg_calculator_tool` and `exchange_rate_tool`.
- Reads box office, actor signals, comparables, FX, VOD benchmarks from `db_evidence`.
- Returns confidence and bounds in USD.

## Control Flow
1. Aggregate comparables and key numeric inputs.
2. Compute risk penalty from risk flags.
3. Compute MG estimate and revenue projections.
4. Derive confidence from evidence sufficiency.
5. Return typed result object.

## Failure Modes / Fallbacks
- Missing comparables/box office: use conservative baseline estimate floor.
- Invalid numeric data: sanitize and clamp values.
- Low sufficiency: wider confidence interval + warning propagation.

## Acceptance Criteria
- Result always includes all required fields.
- Calculations are deterministic for same input bundle.
- Risk severity meaningfully affects MG.

## Test Cases
- High-data coverage valuation.
- Sparse-data fallback valuation.
- High-risk penalties reducing MG.
