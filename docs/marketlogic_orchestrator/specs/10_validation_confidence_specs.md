# 10 Validation And Confidence Spec

## Objective
Define validation gates and confidence signaling before scorecard output.

## Scope
In: financial sanity, citation sufficiency, confidence threshold, warning composition.
Out: model safety policy beyond current runtime checks.

## Inputs/Outputs
- Input: evidence, valuation, confidence score, provider status.
- Output: `ValidationReport` and consolidated warning list.

## Interfaces/Contracts
- `financial_sanity_check(mg, theatrical, vod) -> bool`
- `hallucination_check(citations) -> bool`
- `confidence_threshold_check(confidence) -> bool`
- `combine_validation_warnings(report) -> list[str]`

## Control Flow
1. Build base warning list (provider disabled, low sufficiency).
2. Run three validation gates.
3. Translate failed gates into user-facing warnings.
4. Deduplicate warnings and attach to scorecard.

## Failure Modes / Fallbacks
- Missing citations: hallucination gate fails; warning required.
- Low confidence: threshold gate fails; warning required.
- Invalid valuations: financial sanity fails; warning required.

## Acceptance Criteria
- All failed gates emit warnings.
- Warning output is deduplicated and stable.
- Confidence value is always present in scorecard.

## Test Cases
- Financial sanity fail path.
- Low-citation fail path.
- Low-confidence fail path.
