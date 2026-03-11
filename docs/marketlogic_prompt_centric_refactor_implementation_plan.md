## MarketLogic Hybrid Refactor Plan (Prompt-Centric Specialists + Deterministic Control Plane)

### Summary
Adopt a hybrid architecture: keep the orchestrator, routing, data gateway, and safety enforcement deterministic; move specialist reasoning (valuation/risk/strategy/explainability) to prompt-centric ADK agents. Replace aggressive loop-based retries with limited schema-only retries to preserve production predictability and observability.

### Key Implementation Changes
- Keep deterministic control plane in `run_marketlogic_orchestrator`:
  - deterministic turn classification, entity resolution, workflow routing, follow-up reuse logic, and response shaping.
  - deterministic evidence request construction and stage execution order.
- Keep deterministic data plane in `DataAgent` + `DocumentRetrievalAgent`:
  - all docs/DB access remains tool-driven and non-LLM.
  - evidence bundle, citation extraction, and sufficiency scoring remain deterministic.
- Introduce prompt-centric specialist reasoning agents only for analytical synthesis:
  - `valuation_reasoner`, `risk_reasoner`, `strategy_reasoner`, `explainability_reasoner`.
  - each gets strict input/output prompt contracts (required state keys, JSON schema, citation rules, no fabrication).
- Use **limited schema-only retries** (max 1 retry per specialist):
  - retry only when JSON/schema validation fails or required keys are missing.
  - no semantic “improvement loops”; if retry still fails, deterministic clarification/diagnostic response.
- Preserve deterministic safety gates before and after specialist calls:
  - pre-gates: backend/data/tool readiness and evidence sufficiency thresholds.
  - post-gates: schema validity, confidence threshold checks, citation sufficiency, financial sanity checks.
- Keep existing response contracts; add minimal metadata extensions:
  - continue `conversation_response`, `clarification_response`, `scorecard_response`.
  - add `evidence_basis` and `degraded_mode` fields for grounded vs benchmark-derived outputs.
  - keep comparative-territory output as `scorecard_response` + `comparative_scorecards`.

### Public Interfaces / Type Contract Updates
- Extend scorecard/output typed contract to include:
  - `evidence_basis: "grounded" | "benchmark_derived"`
  - `degraded_mode: { enabled: bool, reason_code: str | null }`
- Add internal specialist I/O schema types (input context + required JSON output keys) for valuation/risk/strategy/explainability validation.
- No changes to ADK HTTP boundary (`POST /v1/run`) or backend internal API endpoints.

### Test Plan
- Deterministic orchestrator/data gateway tests:
  - conversational gating, route selection, follow-up reuse, and evidence failure diagnostics unchanged and passing.
- Specialist schema-retry tests:
  - valid first-pass output, invalid schema then successful retry, invalid schema after retry -> deterministic clarification.
- Safety gate regression tests:
  - backend unavailable/auth failure, low evidence, low citations, confidence below threshold, failed financial sanity.
- Use-case matrix tests from `docs/chat.md`:
  - full scorecard, partial workflows, scenario follow-up, comparative territory, explainability, and degraded-mode behavior.
- Integration/eval:
  - `tests/test_run.py` unchanged contract validation.
  - expand evals to assert no fabricated outputs and deterministic fallback on specialist schema failures.

### Assumptions and Defaults
- Hybrid model is the target architecture; deterministic orchestrator/data gateway are retained as production source of control.
- Retries are strictly schema-level and capped at one retry per specialist to bound latency and variance.
- Existing auth/session persistence/logging flow remains unchanged.
- Rollout remains direct replacement of specialist internals, with deterministic guardrails preserving current operational behavior.
