# 13 Operations, Logging, Security Spec

## Objective
Define operational guardrails and security requirements around ADK runtime execution.

## Scope
In: auth, logging events, secret handling, health endpoints, deployment readiness signals.
Out: cloud-specific IaC implementation.

## Inputs/Outputs
- Input: inbound ADK API requests and runtime execution events.
- Output: secure request handling + structured operational logs.

## Interfaces/Contracts
Security:
- Enforce `X-ADK-API-Key` before runtime invocation.
- Reject missing/invalid key with 401.
- Never log raw keys, secrets, or full prompt payloads.

Logging events:
- API boundary logs: run start, success, failure, auth failure.
- Runtime logs: runner/session-service init, session create/reuse, provider-disabled fallback, run complete.

Operations:
- `/health` endpoint required.
- Env-driven config (`ADK_MODEL`, `DATABASE_URL`, `ADK_API_KEY`, provider keys).

## Control Flow
1. Validate service auth.
2. Emit start log with safe metadata.
3. Run runtime flow.
4. Emit success/failure logs.

## Failure Modes / Fallbacks
- Invalid auth header: immediate reject.
- Runtime exception: log with stack trace and return sanitized 500.
- Missing provider key: deterministic fallback warning path.

## Acceptance Criteria
- Auth failures do not reach model/tool execution.
- Required log events exist with safe metadata only.
- Health endpoint and env config are documented and verifiable.

## Test Cases
- `/v1/run` unauthorized request returns 401.
- Authorized run emits success path behavior.
- Exception path emits failure and returns 500.
