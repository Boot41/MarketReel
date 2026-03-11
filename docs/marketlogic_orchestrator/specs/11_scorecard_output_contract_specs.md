# 11 Scorecard Output Contract Spec

## Objective
Define canonical output contract for strategic distribution scorecard.

## Scope
In: JSON schema, required fields, provenance expectations.
Out: frontend visualization specifics.

## Inputs/Outputs
- Input: synthesized valuation/risk/strategy outputs + citations + warnings.
- Output: scorecard object serialized to JSON reply.

## Interfaces/Contracts
Required top-level fields:
- `projected_revenue_by_territory: { [territory]: number }`
- `risk_flags: RiskFlag[]`
- `recommended_acquisition_price: number`
- `release_timeline: { release_mode, theatrical_window_days }`
- `citations: Citation[]`
- `confidence: number`
- `warnings: string[]`

Field rules:
- Monetary fields in USD unless explicitly converted and labeled.
- `risk_flags` must include severity and mitigation.
- `citations` should reference source path and local reference id/page when available.

## Control Flow
1. Format raw outputs into canonical field names.
2. Apply rounding and normalization.
3. Attach warnings and confidence.
4. Serialize as JSON string response payload.

## Failure Modes / Fallbacks
- Partial upstream results: return full schema with conservative defaults and warnings.
- Empty risk list prohibited: fallback market risk flag.

## Acceptance Criteria
- Schema is stable across intents.
- Response is JSON-serializable every run.
- Required fields are never omitted.

## Test Cases
- Full-scorecard schema validation.
- Sparse-data schema validation.
- Risk-only query still returns canonical contract.
