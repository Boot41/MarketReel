# 12 Eval And Test Matrix Spec

## Objective
Define required eval and test coverage for regression-prone MarketLogic behaviors.

## Scope
In: ADK evals, runtime API tests, contract tests.
Out: load/perf testing.

## Inputs/Outputs
- Input: representative prompts, fixture data, deterministic fallbacks.
- Output: pass/fail checks for behavior and output contracts.

## Interfaces/Contracts
- Primary suites:
  - `adk-server/tests/test_run.py`
  - `adk-server/agents/eval/test_eval.py`
- Core assertions:
  - scorecard schema validity
  - citation presence
  - confidence/warning behavior
  - session reuse behavior

## Control Flow
1. Execute API boundary tests (auth + run behavior).
2. Execute agent eval scenarios.
3. Verify schema and warning expectations.
4. Validate deterministic behavior in no-provider mode.

## Failure Modes / Fallbacks
- Provider unavailable: evals use deterministic path assertions.
- Missing corpus rows: assert warning behavior rather than hard fail on numeric values.

## Acceptance Criteria
Required scenarios pass:
1. Valuation prompt for territory revenue/MG behavior.
2. Censorship risk prompt returns non-empty typed risks.
3. Follow-up scenario prompt updates strategy output.
4. Low-sufficiency prompt returns lowered confidence and warning.

## Test Cases
- “How will Interstellar perform in India?”
- “Does Deadpool face censorship issues in Saudi Arabia?”
- “If we skip theatrical in Germany, how does ROI change?”
- Sparse-evidence case for warning path.
