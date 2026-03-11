# 03 DocumentRetrievalAgent Spec

## Objective
Specify deterministic retrieval from indexed corpus for scripts, synopses, reviews, marketing, and compliance docs.

## Scope
In: index inspection, retrieval plan generation, targeted fetch, sufficiency decision.
Out: semantic ranking model changes.

## Inputs/Outputs
- Input: retrieval intent (`movie`, `territory`, `intent`, requested doc types).
- Output: selected docs/scenes + sufficiency status (`PASS`/`EXPAND`) and score.

## Interfaces/Contracts
- Tools: `IndexRegistry`, `IndexNavigator`, `TargetedFetcher`, `SufficiencyChecker`.
- Plan contract includes `doc_types`, `max_docs`, `max_scenes`.
- Fetch output includes normalized metadata for citation generation.

## Control Flow
1. Load manifests and known entities.
2. Build deterministic retrieval plan by intent.
3. Fetch bounded docs and scene chunks.
4. Evaluate sufficiency and trigger expansion policy if needed.

## Failure Modes / Fallbacks
- Corrupt index entries: skip invalid lines, continue.
- No territory-specific censorship doc: include best available compliance docs and warn.
- Insufficient items: set `EXPAND` and downstream low-confidence handling.

## Acceptance Criteria
- Retrieval is explainable and reproducible.
- Fetch is bounded by configured limits.
- Sufficiency score is always returned.

## Test Cases
- Territory-specific censorship retrieval.
- Movie-scene retrieval from `scripts_indexed`.
- Low-hit corpus case.
