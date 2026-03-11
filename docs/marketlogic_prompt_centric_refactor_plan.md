### MarketReel Prompt-Centric Agent Refactor Plan (Using blog-writer as Reference)

### Summary

Refactor MarketReel from mostly deterministic orchestration into a prompt-led multi-agent ADK design that still preserves strict routing/safety gates. Mirror blog-writer
patterns: explicit specialist prompts, robust looped sub-agents with validation checkers, typed state keys, and a root interactive orchestrator. Goal: pass all use cases in
docs/chat.md with deterministic fallbacks and transparent diagnostics.

### Key Implementation Changes

- Adopt the blog-writer structure pattern in MarketReel agent package
    - Keep one user-facing root agent (MarketLogicOrchestrator) but define specialist ADK sub-agents with explicit instruction prompts:
        - data_agent (evidence planning + retrieval orchestration)
        - valuation_agent
        - risk_agent
        - strategy_agent
        - explainability_agent (why weak/strong market, evidence-backed justification)
    - Add a small config module for model split (worker_model, critic_model) similar to blog-writer/config.py.
- Introduce robust sub-agent loops + validation checkers
    - For each specialist with critical outputs, wrap in LoopAgent(max_iterations=2-3):
        - robust_valuation_agent + ValuationValidationChecker
        - robust_risk_agent + RiskValidationChecker
        - robust_strategy_agent + StrategyValidationChecker
    - Validation checkers escalate only when required output keys are present and schema-valid; otherwise force retry.
    - Keep hard stop to deterministic clarification if retries exhaust.
- Define explicit prompt contracts per specialist
    - Add prompt instructions that encode:
        - required input state keys
        - required output JSON schema
        - allowed tools only
        - evidence/citation obligations
        - no fabricated numbers when evidence missing
    - Root orchestrator prompt handles: turn type classification, workflow intent mapping, follow-up reuse intent, and invocation order.
- Retain current hardening and integrate into prompt workflow
    - Keep existing evidence integrity gate and diagnostic payloads (reason_code, missing_requirements, next_action).
    - Gate runs before/after specialist loops:
        - before: tool/data readiness
        - after: output schema + confidence + citation sufficiency.
    - Preserve single ADK entrypoint and service auth flow.
- Use-case matrix implementation (docs/chat.md)
    - Map each use case to route + required specialists + output type:
        - conversational: greeting/help/clarification
        - partial workflows: valuation-only, risk-only, strategy-only
        - full scorecard workflow
        - scenario follow-up (reuse prior artifacts; rerun only required specialists)
        - comparative territory analysis
        - explanation/justification requests
    - Add deterministic fallback policy for missing market rows:
        - benchmark-derived mode with downgraded confidence when allowed
        - otherwise structured clarification block.
- State and interface contracts
    - Standardize state keys (input/output) across all specialists (like blog_outline/blog_post pattern).
    - Keep response types:
        - conversation_response
        - clarification_response
        - scorecard_response
    - Add explicit evidence_basis + degraded_mode fields for non-fully-grounded analytical outputs.

### Test Plan

- Unit tests (agent orchestration)
    - turn-type routing correctness for all conversational/support cases.
    - specialist loop success/failure behavior with validation checker escalation.
    - tool failure -> diagnostic clarification (no synthetic scorecard).
- Use-case matrix tests (from docs/chat.md)
    - one test per listed core business and conversational use case.
    - verify expected specialists invoked, expected output type, and required fields.
- Eval suite expansion
    - add eval datasets for:
        - full scorecard
        - partial workflows
        - scenario follow-up reuse
        - comparative territory analysis
        - explainability/justification
        - backend unavailable/auth failure behavior.
- Integration tests
    - internal API + ADK end-to-end on seeded DB.
    - readiness checks verify both registry and market endpoints.
    - confirm Interstellar + India behavior follows policy (scorecard/degraded/clarification depending on evidence mode).

### Assumptions

- Keep one ADK root entrypoint; add prompt-driven sub-agents under it.
- Keep current backend/internal API contracts; only extend response metadata fields where needed.
- “Pass all use cases” allows either full output or structured degraded/clarification response when evidence is genuinely unavailable, with no silent fabrication.