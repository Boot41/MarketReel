# 07 Tools And Contracts Spec

## Objective
Define exact signatures and return contracts for all document, DB, valuation, and utility tools.

## Scope
In: tools listed in `docs/SOLUTION.md` and used in agent flow.
Out: implementation internals of each tool.

## Inputs/Outputs
- Input: typed tool arguments.
- Output: deterministic typed payloads suitable for orchestration.

## Interfaces/Contracts
- Document tools:
  - `IndexRegistry() -> dict`
  - `IndexNavigator(movie, territory, intent) -> plan`
  - `TargetedFetcher(plan) -> {documents, scenes}`
  - `SufficiencyChecker(fetched) -> {status, score, total_items}`
- DB tools:
  - `get_box_office_by_genre_territory(movie, territory) -> metrics`
  - `get_actor_qscore(movie) -> metrics`
  - `get_theatrical_window_trends(territory) -> list`
  - `get_exchange_rates(territory) -> currency`
  - `get_vod_price_benchmarks(territory) -> metrics`
  - `get_comparable_films(movie, territory) -> list`
- Valuation/utility tools:
  - `mg_calculator_tool(...) -> float`
  - `exchange_rate_tool(amount_usd, rate_to_usd) -> float`
  - `source_citation_tool(items) -> Citation[]`
  - `financial_sanity_check(...) -> bool`
  - `hallucination_check(citations) -> bool`
  - `confidence_threshold_check(confidence) -> bool`
  - `format_scorecard(...) -> Scorecard`

## Control Flow
- Tools are side-effect free except DB reads and logs.
- DataAgent invokes document and DB toolsets.
- Orchestrator applies utility/validation tools before output.

## Failure Modes / Fallbacks
- Tool exceptions should degrade to safe contract defaults where feasible.
- Contract violations are treated as blocking defects.

## Acceptance Criteria
- Signatures and keys are stable and documented.
- All tools have explicit output schemas and null/default semantics.

## Test Cases
- Contract tests per tool.
- Schema-key presence tests.
- Default-return tests on upstream failure.
