# 09 Session State And Follow-Ups Spec

## Objective
Define ADK session state keys, lifecycle, and follow-up recomputation policy.

## Scope
In: state persistence, cache behavior, scenario override handling, artifact reuse.
Out: backend chat record schema.

## Inputs/Outputs
- Input: current turn message + prior session state.
- Output: updated state delta merged into session state.

## Interfaces/Contracts
State keys:
- `resolved_context`
- `evidence_bundle`
- `valuation`
- `risk`
- `strategy`
- `last_scorecard`
- `recommended_acquisition_local`

## Control Flow
1. Load session state.
2. Resolve new intent/entities.
3. Determine if message is scenario delta or topic shift.
4. Reuse prior artifacts when entity scope unchanged.
5. Refresh only invalidated layers.
6. Persist merged state.

## Failure Modes / Fallbacks
- Corrupt state object: reset to minimal safe state.
- Entity mismatch in follow-up: force full retrieval recompute.
- Missing previous artifacts: treat as fresh query.

## Acceptance Criteria
- State keys are stable and documented.
- Follow-up behavior is deterministic.
- State persistence does not leak secrets.

## Test Cases
- Streaming-first follow-up in same territory.
- Territory switch follow-up forcing recompute.
- Missing prior state fallback to fresh run.
