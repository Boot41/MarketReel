# Plan: Eliminate Silent Tool Failures and Non-Deterministic-Looking Scorecards

## Summary
Make evidence/tool failures explicit, block full scorecard generation when minimum evidence is unavailable, and add observability so prompts like “evaluate interstellar for india” either return a valid evidence-backed scorecard or a clear error/clarification response (never hidden fallback output).

## Implementation Changes
- **Harden internal tool client behavior**
  - Replace silent `{}` returns in ADK internal API helpers with typed failure results carrying `status_code`, `error_type`, and endpoint context.
  - Distinguish retryable transport failures vs terminal auth/config failures.
  - Add structured warning/error logs for each failed tool call (without secrets).

- **Add evidence integrity gate before valuation/strategy**
  - Define minimum required evidence per workflow intent:
    - `valuation`: box office/comparables or explicit “insufficient evidence” response
    - `risk`: at least one risk-relevant citation class or explicit insufficiency
    - `strategy/full_scorecard`: both market + citation sufficiency thresholds
  - If gate fails, return `clarification_response` with actionable reason (`backend_unavailable`, `internal_auth_failed`, `insufficient_market_data`, etc.) instead of synthetic numeric projections.

- **Remove misleading numeric fallback paths**
  - Disable baseline MG fallback when upstream evidence retrieval failed due to tool error.
  - Keep deterministic computations only when evidence was actually fetched and validated.
  - Preserve deterministic behavior for valid evidence; block only error-derived defaults.

- **Strengthen response contract for failure transparency**
  - Extend non-scorecard payloads with machine-readable diagnostics:
    - `response_type`
    - `reason_code`
    - `missing_requirements`
    - `next_action`
  - Keep scorecard schema unchanged for successful runs.

- **Operational guardrails**
  - Add startup self-check endpoint/path in ADK runtime to verify backend connectivity and internal auth before first workflow run.
  - Add runtime metric/log counters for:
    - tool failures by endpoint
    - gated (blocked) scorecards
    - fallback invocations (should trend to near-zero).

## Public APIs / Interfaces
- No endpoint changes (`POST /v1/run` remains).
- Response payload additions for non-scorecard paths:
  - `reason_code` and `missing_requirements` fields in clarification/error responses.
- Internal behavior change:
  - full scorecard no longer emitted on failed tool/evidence preconditions.

## Test Plan
- **Unit tests (ADK tools/orchestrator)**
  - Internal API 401/403 -> `clarification_response` with `reason_code=internal_auth_failed`.
  - Backend unavailable/timeouts -> `clarification_response` with `reason_code=backend_unavailable`.
  - Partial evidence below thresholds -> blocked scorecard with explicit missing requirements.
  - Healthy evidence -> normal `scorecard_response` with citations/confidence.

- **Integration tests**
  - Run with backend down: verify no synthetic scorecard output.
  - Run with wrong internal key: verify deterministic failure response and logs.
  - Run healthy backend+ADK: verify interstellar/india returns evidence-backed scorecard.

- **Regression/eval tests**
  - Keep conversational gate behavior (`hi`) unchanged.
  - Keep explainability session-read behavior unchanged.
  - Add eval asserting no hidden fallback scorecard on tool failure.

## Assumptions
- Preferred behavior is **fail transparent, not fail silent**.
- It is acceptable to return clarification/error responses instead of best-effort numeric estimates when required tools fail.
- Existing frontend can render added diagnostic fields for non-scorecard responses (or safely ignore unknown keys).
