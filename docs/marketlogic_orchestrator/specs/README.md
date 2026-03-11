# MarketLogic Specs Package

This folder contains implementation specs for all agents, tools, contracts, and flows defined in `docs/SOLUTION.md`.

## Canonical Plan
- `plan.md` is the top-level implementation plan.
- `implementation_sequence.md` is the execution order and phase gates.

## Specs Index
- `00_system_context/specs.md`
- `01_marketlogic_orchestrator/specs.md`
- `02_data_agent/specs.md`
- `03_document_retrieval_agent/specs.md`
- `04_valuation_agent/specs.md`
- `05_risk_agent/specs.md`
- `06_strategy_agent/specs.md`
- `07_tools_contracts/specs.md`
- `08_runtime_flow/specs.md`
- `09_session_state_followups/specs.md`
- `10_validation_confidence/specs.md`
- `11_scorecard_output_contract/specs.md`
- `12_eval_and_test_matrix/specs.md`
- `13_ops_logging_security/specs.md`

## Working Rules
- `docs/SOLUTION.md` is source architecture intent and remains unchanged.
- All interfaces and type names in specs must stay consistent with `adk-server/agents/marketlogic/types.py`.
- Backend owns business chat history; ADK session state is runtime context only.
