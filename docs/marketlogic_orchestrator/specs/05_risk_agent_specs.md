# 05 RiskAgent Spec

## Objective
Define typed risk extraction for censorship, cultural sensitivity, and market-fit risks.

## Scope
In: detection heuristics, severity mapping, mitigation generation, confidence assignment.
Out: strategy or valuation math details.

## Inputs/Outputs
- Input: `EvidenceBundle`.
- Output: `RiskFlag[]`.

## Interfaces/Contracts
- `RiskFlag` fields: `category`, `severity`, `scene_ref`, `source_ref`, `mitigation`, `confidence`.
- Required categories for v1: `CENSORSHIP`, `CULTURAL_SENSITIVITY`, `MARKET`.

## Control Flow
1. Parse doc evidence by source type.
2. Apply territory-aware risk detection.
3. Generate typed risk flags with mitigation text.
4. If no explicit risks found, emit low-severity market baseline flag.

## Failure Modes / Fallbacks
- Missing compliance docs: downgrade confidence, include market baseline flag.
- Ambiguous signals: prefer `MEDIUM` with explicit mitigation.
- No evidence: non-empty fallback risk output required.

## Acceptance Criteria
- Non-empty `RiskFlag[]` for every run.
- Every flag has source reference or explicit derived marker.
- Severity domain is constrained to LOW/MEDIUM/HIGH.

## Test Cases
- High-severity censorship detection for strict territory.
- Cultural-sensitivity detection from report text.
- Empty-doc fallback behavior.
