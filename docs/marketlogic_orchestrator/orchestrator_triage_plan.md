# Orchestrator Triage Plan (Direct / Clarify / Workflow)

## Summary
Add a deterministic triage step at the start of orchestration that classifies user input and routes to exactly one action:
- `respond_directly`
- `ask_clarification`
- `run_workflow`

This prevents greetings/casual turns from triggering valuation and prevents fake analysis when entities are missing.

## Core Decision Logic
1. Detect conversational intent
- Detect `greeting`, `thanks`, `help`, and `casual` signals from message text.
- If conversational and no explicit analytic request, route to `respond_directly`.

2. Resolve analysis context
- Extract `movie` and `territory` from current message.
- If missing, try prior session context.
- Compute context status:
  - `sufficient`: movie + territory available
  - `insufficient`: either missing

3. Route action
- `respond_directly`:
  - For greeting/thanks/casual/help where user is not requesting analysis.
  - Return conversational guidance + examples of supported queries.
- `ask_clarification`:
  - For analytic/risk/strategy/valuation intent with insufficient context.
  - Ask specifically for missing field(s): movie, territory, or both.
- `run_workflow`:
  - For analytic intent with sufficient context.
  - Execute Data -> Risk -> Valuation -> Strategy -> Validation/Scorecard.

## Implementation Changes
- Add a pre-routing function in orchestrator, e.g. `decide_action(message, session_state)`.
- Return a typed routing object:
  - `action: Literal["respond_directly","ask_clarification","run_workflow"]`
  - `intent`
  - `movie | None`
  - `territory | None`
  - `missing_fields: list[str]`
- Remove hardcoded fallback movie/territory defaults.
- Ensure ADK finalize stage supports non-scorecard text responses for first two actions.
- Keep existing workflow path unchanged when routed to `run_workflow`.

## Test Plan
- `"hi"` -> `respond_directly` (no scorecard fields).
- `"thanks"` -> `respond_directly`.
- `"help"` -> `respond_directly` with examples.
- `"Should we acquire this?"` with no context -> `ask_clarification` (asks movie + territory).
- `"Risk for Interstellar"` with no territory and no prior -> `ask_clarification` (asks territory).
- Follow-up with prior context -> `run_workflow`.
- Analytic query with both entities -> `run_workflow`.

## Assumptions
- Casual/greeting/thanks/help should never trigger valuation workflow.
- Analytic workflow requires both movie and territory (current turn or prior context).
- Clarification prompts should be precise and minimal (ask only missing fields).
