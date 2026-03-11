# Implementation Sequence

## Build Order
1. `00_system_context`
2. `07_tools_contracts`
3. `01_marketlogic_orchestrator`
4. `02_data_agent`
5. `03_document_retrieval_agent`
6. `04_valuation_agent`
7. `05_risk_agent`
8. `06_strategy_agent`
9. `08_runtime_flow`
10. `09_session_state_followups`
11. `10_validation_confidence`
12. `11_scorecard_output_contract`
13. `13_ops_logging_security`
14. `12_eval_and_test_matrix`

## Phase Gates
- Gate A: contracts freeze (`07_tools_contracts`) before downstream implementation.
- Gate B: orchestration flow freeze (`01`, `08`, `09`) before optimization.
- Gate C: scorecard schema freeze (`11`) before frontend/output integration.
- Gate D: eval/security/ops pass (`12`, `13`) before release.

## Definition Of Done
- Types and tool signatures are implemented and tested.
- Orchestrator routing and session behavior are deterministic for baseline queries.
- Validation and confidence warnings are attached to output consistently.
- Evals cover valuation, risk, strategy what-if, and low-sufficiency fallback.
