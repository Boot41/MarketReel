# 00 System Context Spec

## Objective
Define system boundaries, ownership, and data movement for MarketLogic.

## Scope
In: service boundaries, data stores, ownership model, trust boundaries.
Out: algorithm internals for valuation/risk/strategy.

## Inputs/Outputs
- Input: user message from frontend via backend APIs.
- Output: structured scorecard JSON reply returned from ADK service through backend.

## Interfaces/Contracts
- Frontend only calls backend.
- Backend calls ADK service `POST /v1/run` with service API key.
- ADK service returns `{ reply, session_id }` where `reply` is scorecard JSON string.
- Backend stores business chat/messages; ADK persists runtime session state.

## Control Flow
1. User sends message.
2. Backend authenticates user JWT, persists user message.
3. Backend calls ADK with `message`, `user_id`, and optional `session_id`.
4. ADK orchestrates agent flow and returns scorecard.
5. Backend persists assistant message and returns to frontend.

## Failure Modes / Fallbacks
- ADK auth failure: reject before model/tool run.
- Session missing: create new ADK session.
- Provider key absent: deterministic fallback path still returns scorecard with warning.

## Acceptance Criteria
- Ownership boundaries are explicit and non-overlapping.
- Chat history and ADK session state are clearly separated.
- Trust boundary for API key auth is documented.

## Test Cases
- New chat path with null session id.
- Existing chat path with reused session id.
- Missing/invalid ADK API key is rejected.
