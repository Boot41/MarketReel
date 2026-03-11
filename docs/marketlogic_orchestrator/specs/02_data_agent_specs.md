# 02 DataAgent Spec

## Objective
Define DataAgent as the single gateway for all document and DB evidence access.

## Scope
In: evidence request handling, retrieval orchestration, DB queries, citation packaging, sufficiency score.
Out: downstream financial/risk strategy reasoning.

## Inputs/Outputs
- Input: `EvidenceRequest`.
- Output: `EvidenceBundle` with `document_evidence`, `db_evidence`, `citations`, `data_sufficiency_score`.

## Interfaces/Contracts
- Only DataAgent may call document retrieval tools and DB tools directly.
- `needs_docs` / `needs_db` flags control retrieval breadth.
- Citation bundle must include source path + reference id.

## Control Flow
1. Build retrieval plan from movie, territory, intent.
2. Fetch targeted docs/scenes and compute sufficiency.
3. Conditionally execute DB evidence tools.
4. Merge evidence and attach citations.
5. Return unified typed bundle.

## Failure Modes / Fallbacks
- DB query failure: return safe zero-value structures; log warning.
- No doc hits: low sufficiency score, continue with warning path.
- Partial evidence: include what exists, never return malformed contract.

## Acceptance Criteria
- DataAgent output schema is stable across all intents.
- No other agent depends on DB/doc tools directly.
- Citations are present when document evidence exists.

## Test Cases
- Docs-only risk request.
- Full-scorecard request with docs + DB.
- DB unavailable scenario producing resilient fallback bundle.
