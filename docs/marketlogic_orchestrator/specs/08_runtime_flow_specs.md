# 08 Runtime Flow Spec

## Objective
Define end-to-end runtime flow for fresh and follow-up queries.

## Scope
In: sequence from `/v1/run` request to scorecard response.
Out: frontend rendering behavior.

## Inputs/Outputs
- Input: `{ message, user_id, session_id? }`.
- Output: `{ reply, session_id }` where `reply` is scorecard JSON.

## Interfaces/Contracts
- API boundary in `adk-server/app/main.py` validates `X-ADK-API-Key` before invoking runtime.
- Runtime entrypoint `run_agent` creates/reuses ADK session and invokes orchestrator.

## Control Flow
Fresh query:
1. Auth check.
2. Session create/reuse.
3. Orchestrator resolves context + intent.
4. DataAgent retrieves evidence.
5. Risk and valuation run; strategy synthesizes.
6. Validation and warning merge.
7. Scorecard format + JSON serialization.
8. Response return.

Follow-up query:
1. Reuse existing session state.
2. Detect scenario delta.
3. Reuse prior artifacts where valid.
4. Recompute affected downstream outputs.
5. Return updated scorecard.

## Failure Modes / Fallbacks
- Session fetch miss with provided id: create session and continue.
- Any non-fatal evidence gap: return scorecard with warnings.
- Fatal runtime exception: 500 from API boundary.

## Acceptance Criteria
- Fresh and follow-up paths are both fully specified.
- Session lifecycle is explicit.
- Validation always runs before response.

## Test Cases
- New session path.
- Existing session follow-up path.
- Runtime failure path with proper API error.
