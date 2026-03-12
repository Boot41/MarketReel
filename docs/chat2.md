# MarketReel — Global Film Distribution & Acquisition Agent
## Architecture & Implementation Guide
### Google Agent Development Kit (ADK) — v1.0

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Agent Architecture](#2-agent-architecture)
3. [Agent Reference](#3-agent-reference)
   - 3.1 [MarketLogicOrchestrator](#31-marketlogicorchestrator)
     - 3.1.1 [Conversational Gate](#311-conversational-gate)
     - 3.1.2 [Routing Policy](#312-routing-policy)
     - 3.1.3 [Output Policy](#313-output-policy)
     - 3.1.4 [Intent Classification](#314-intent-classification)
     - 3.1.5 [Execution Dependency](#315-execution-dependency)
   - 3.2 [DataAgent](#32-dataagent)
   - 3.3 [ValuationAgent](#33-valuationagent)
   - 3.4 [RiskAgent](#34-riskagent)
   - 3.5 [StrategyAgent](#35-strategyagent)
   - 3.6 [DocumentRetrievalAgent (Sub-agent)](#36-documentretrievalagent-sub-agent)
4. [Tool Reference](#4-tool-reference)
5. [Document Corpus Mapping](#5-document-corpus-mapping)
6. [Runtime Flows — All Use Cases](#6-runtime-flows--all-use-cases)
   - 6.1 [Territory Valuation & MG Estimation](#61-territory-valuation--mg-estimation)
   - 6.2 [Censorship & Cultural Risk Flagging](#62-censorship--cultural-risk-flagging)
   - 6.3 [Festival Sentiment → Revenue Split](#63-festival-sentiment--revenue-split-projection)
   - 6.4 [Release Window & Marketing Strategy](#64-release-window--marketing-strategy)
   - 6.5 [Contextual Follow-up: Skip Theatrical](#65-contextual-follow-up-skip-theatrical)
   - 6.6 [Contextual Follow-up: Underperformance Explanation](#66-contextual-follow-up-underperformance-explanation)
   - 6.7 [Multi-Territory Comparison](#67-multi-territory-comparison)
   - 6.8 [Full Distribution Scorecard](#68-full-distribution-scorecard)
   - 6.9 [Acquisition Bid Support](#69-acquisition-bid-support)
   - 6.10 [VOD vs. Theatrical ROI Modelling](#610-vod-vs-theatrical-roi-modelling)
   - 6.11 [Greeting](#611-greeting)
   - 6.12 [Help Request](#612-help-request)
   - 6.13 [Clarification Flow](#613-clarification-flow)
   - 6.14 [Low-Confidence Response](#614-low-confidence-response)
   - 6.15 [Insufficient-Data Handling](#615-insufficient-data-handling)
   - 6.16 [Explainability Request](#616-explainability-request)
   - 6.17 [Evidence Inspection](#617-evidence-inspection)
7. [Session State & Follow-up Handling](#7-session-state--follow-up-handling)
8. [Validation Pipeline](#8-validation-pipeline)
9. [Structured JSON Scorecard Schema](#9-structured-json-scorecard-schema)
10. [Google ADK Implementation Notes](#10-google-adk-implementation-notes)

---

## 1. System Overview

**MarketReel** is a multi-agent AI system built on **Google ADK** that helps film distribution executives evaluate independent films for global acquisition and theatrical release strategy. It combines structured database queries with unstructured document understanding to produce financially grounded, risk-aware distribution scorecards.

### Core Capabilities at a Glance

| Capability | Description |
|---|---|
| Territory Valuation | Estimate Minimum Guarantees (MG) per region by cross-referencing script themes with historical box office data |
| Censorship Flagging | Detect plot points and imagery that trigger regulatory cuts in specific markets using regional PDFs |
| Festival Sentiment | Translate critic reviews and awards buzz into projected theatrical vs. digital revenue splits |
| Release Strategy | Recommend optimal release window, marketing spend, and platform priority for each film |
| Scenario Modelling | Answer counterfactual questions: skip theatrical, change territory, vary release timing |
| Full Scorecard | Generate structured JSON output: revenue by territory, risk flags, acquisition price, release timeline |

### Architecture Pattern

MarketReel uses a **hierarchical multi-agent architecture** orchestrated via Google ADK. A top-level Orchestrator passes every input through a Conversational Gate before routing, delegates to specialist agents, enforces a shared data gateway, and runs a validation pipeline before every analytical response.

| Property | Value |
|---|---|
| Agents | 5 agents (1 orchestrator + 4 specialists) + 1 sub-agent |
| Tools | 16 tools across document retrieval, database, valuation, and validation layers |
| Data sources | Local PostgreSQL database + local unstructured document corpus |
| Interface | Natural language queries → structured JSON scorecard or natural-language response |
| Framework | Google ADK (Python), multi-turn session state |

---

## 2. Agent Architecture

```
User Natural Language Query
          │
          ▼
MarketLogicOrchestrator
  ┌───────────────────────────────────┐
  │  1. Conversational Gate           │
  │     classify turn type            │
  │     check session context         │
  ├───────────────────────────────────┤
  │  2. Routing Policy                │
  │     apply routing rules           │
  ├───────────────────────────────────┤
  │  3. Output Policy                 │
  │     select response format        │
  └───────────────────────────────────┘
          │
          ├─── conversational / help / clarification
          │         │
          │         ▼
          │    Natural-language response
          │    (no agents called)
          │
          └─── workflow intent
                    │
                    ▼
              DataAgent
              · single data gateway · DB + Docs + Citations
                    │                       │
                    ▼                       ▼
        DocumentRetrievalAgent        DB Tools (PostgreSQL)
                    │
                    ▼  unified evidence + sufficiency score
          ┌─────────────────────┐
          │  ValuationAgent     │  ┐
          │  RiskAgent          │  ├─ parallel
          └─────────────────────┘  ┘
                    │
                    ▼  (after Valuation + Risk complete, when strategy_needs_risk=true)
          ┌─────────────────────┐
          │  StrategyAgent      │
          └─────────────────────┘
                    │
                    ▼
          Validation Pipeline
          · financial_sanity_check
          · hallucination_check
          · confidence_threshold_check
                    │
                    ▼
          Structured JSON Scorecard
```

---

## 3. Agent Reference

### 3.1 MarketLogicOrchestrator

**Role:** Top-level controller. Every user input is processed here first. The Orchestrator passes input through the Conversational Gate before any routing decision is made. Conversational turns are handled directly with a natural-language response. Workflow turns are routed to specialist agents, validated, and returned as a structured scorecard. No other agent communicates with the user directly.

**Owns:**
- Conversational Gate (turn type classification)
- Routing policy enforcement
- Output policy enforcement
- Session state storage and retrieval
- Intent classification for workflow turns
- Execution dependency resolution (`strategy_needs_risk` flag)
- Pre-response validation pipeline
- Scorecard formatting via `format_scorecard()`
- Citation assembly via `source_citation_tool()`

**Inputs:**
- User natural language query
- Session ID
- Movie ID or movie name (resolved from query or session state)
- Optional scenario override (e.g., skip theatrical)

**Outputs:**
- Structured JSON scorecard — for analytical turns
- Natural-language response — for conversational turns
- Clarification prompt — for ambiguous or incomplete turns
- Confidence warning — if `confidence_threshold_check` fails

---

### 3.1.1 Conversational Gate

The Conversational Gate is the **first processing step** inside the Orchestrator. It runs before intent classification and before any routing decision. Its job is to classify the turn type, check session context for ambiguous inputs, and decide whether the turn should be handled conversationally or routed to the workflow.

**Step 1 — Session context check (runs first, before turn-type classification):**

Before classifying the turn, the Orchestrator checks whether an active session exists with a resolved movie and territory. This resolves the key ambiguity between a `workflow_followup` and a `clarification` for inputs like "What about Germany?" or "Show me the risk."

| Condition | Resolution |
|---|---|
| Active session exists, movie + territory resolved | Treat as `workflow_followup` — load state and continue |
| Active session exists, movie resolved but territory missing | Treat as `clarification` — ask for territory |
| No active session, query references a film or territory | Proceed to turn-type classification |
| No active session, query has no workflow signal | Treat as `greeting`, `help`, or `clarification` |

**Step 2 — Turn type classification:**

| Turn Type | Description | Examples |
|---|---|---|
| `greeting` | Casual opening, no workflow signal | "Hi", "Hello", "Good morning" |
| `acknowledgement` | User confirms, thanks, or reacts | "Got it", "Thanks", "Perfect" |
| `help` | User asks what the system can do | "What can you help with?", "What do you do?" |
| `clarification` | User's input is ambiguous or incomplete; system needs more info | "Tell me about this film" (no film named), "What about Germany?" (no session) |
| `workflow_request` | Clear analytical request with enough context to proceed | "What MG should we pay for Dune in Brazil?" |
| `workflow_followup` | Follow-up on an active session — scenario update, drill-down, or question about prior output | "If we skip theatrical, how does ROI change?" |

---

### 3.1.2 Routing Policy

The Orchestrator applies the following routing rules in order after the Conversational Gate classifies the turn. **Rules are evaluated top to bottom; the first match wins.**

| Priority | Condition | Action |
|---|---|---|
| 1 | Turn type is `greeting` or `acknowledgement` | Respond directly — no agents called |
| 2 | Turn type is `help` | Respond with capability summary — no agents called |
| 3 | Turn type is `clarification` | Respond with clarification prompt — no agents called |
| 4 | Turn type is `workflow_followup` | Load session state → route to StrategyAgent only (or targeted specialist); skip DataAgent if evidence already cached |
| 5 | Turn type is `workflow_request` and movie + territory are both resolved | Classify analytical intent → route to workflow |
| 6 | Turn type is `workflow_request` but movie is missing | Respond with clarification prompt asking for movie name |
| 7 | Turn type is `workflow_request` but territory is missing | Respond with clarification prompt asking for target territory |
| 8 | Analytical signal is weak and no session context exists | Respond with clarification prompt — **never default to `full_scorecard`** |

> **Rule 8 is absolute.** Unknown or ambiguous intent never triggers a full scorecard run. The cost and latency of a full scorecard run means it is only invoked when the user explicitly requests it or when all required context is present and the intent is unambiguously `full_scorecard`.

---

### 3.1.3 Output Policy

The Orchestrator selects the response format based on turn type. The format is determined after routing, before any agents are called.

| Turn Type | Response Format | Notes |
|---|---|---|
| `greeting` | Natural-language response | Short, friendly, no scorecard |
| `acknowledgement` | Natural-language response | Short confirmation or next-step prompt |
| `help` | Natural-language response | Structured capability list in plain text |
| `clarification` | Clarification prompt | Single focused question; never multiple questions at once |
| `workflow_request` / `workflow_followup` (analytical) | Structured JSON scorecard | Full schema; citations included |
| `workflow_request` / `workflow_followup` (low-confidence) | JSON scorecard + `confidence_warning` field populated | Scorecard still returned; warning surfaced prominently |
| `workflow_request` / `workflow_followup` (insufficient data) | Natural-language response | Explains what data is missing and what the user can do |
| Explainability request | Natural-language response | References session-state citations; no re-computation |
| Evidence inspection | Natural-language response | Surfaces raw citations and source references from session state |

---

### 3.1.4 Intent Classification

Intent classification only runs after the Conversational Gate confirms the turn is a `workflow_request` or `workflow_followup`. The Orchestrator classifies the request into one of four workflow intents:

| Intent | Trigger |
|---|---|
| `valuation` | User wants MG estimate, revenue projection, or comparable film analysis |
| `risk` | User wants censorship flags, cultural sensitivity issues, or market fit concerns |
| `strategy` | User wants release window, marketing spend, platform priority, or scenario comparison |
| `full_scorecard` | User explicitly requests a complete distribution scorecard |

> `full_scorecard` is **only triggered by explicit user request**. It is never inferred from ambiguous input.

---

### 3.1.5 Execution Dependency

The Orchestrator sets a `strategy_needs_risk` flag at classification time to control whether StrategyAgent must wait for RiskAgent or can run immediately after ValuationAgent.

**Default execution order:**

```
DataAgent
    │
    ├── ValuationAgent ──┐
    │                    ├── both complete → StrategyAgent (when strategy_needs_risk=true)
    └── RiskAgent ───────┘

ValuationAgent and RiskAgent always run in parallel.
StrategyAgent waits for both when strategy_needs_risk=true.
StrategyAgent runs immediately after ValuationAgent when strategy_needs_risk=false.
```

**`strategy_needs_risk` flag decision table:**

| Condition | `strategy_needs_risk` |
|---|---|
| Intent is `full_scorecard` | `true` |
| Intent is `strategy` in a new territory with no prior risk output in session | `true` |
| Intent is `strategy` and risk output already in session state for this territory | `false` — reuse cached risk output |
| Intent is `strategy` and query is VOD-only or streaming-only scenario | `false` |
| Intent is `valuation` only | StrategyAgent not called |
| Intent is `risk` only | StrategyAgent not called |

---

### 3.2 DataAgent

**Role:** Single data gateway. No specialist agent (Valuation, Risk, Strategy) touches the database or documents directly. All evidence flows through DataAgent.

**Owns:**
- All document retrieval requests (delegated to DocumentRetrievalAgent)
- All DB queries
- Evidence packaging and deduplication
- Citation attachment per evidence item
- Data sufficiency scoring

**Inputs:**
- Typed evidence request from Orchestrator, ValuationAgent, RiskAgent, or StrategyAgent
- Movie ID + target territory + retrieval intent

**Outputs:**
- Unified context object with document evidence + DB evidence
- Citations per evidence item (source, page/chunk)
- `data_sufficiency_score` (0–1)
- Flags for missing or low-confidence data

---

### 3.3 ValuationAgent

**Role:** Quantitative financial reasoning agent. Uses historical box office data, Q-scores, comparable films, exchange rates, and script themes to estimate MG and project revenue.

**Owns:**
- MG reasoning (`mg_calculator_tool`)
- Revenue projection by release mode
- Currency conversion (`exchange_rate_tool`)
- Confidence interval calculation
- Comparable film selection and weighting

**Inputs (from DataAgent):**
- Box office history by genre and territory
- Actor Q-scores and social reach
- Comparable films and their MGs
- Exchange rates
- Script themes and narrative hooks
- Known release history for the film (if any)

**Outputs:**
- MG estimate with confidence interval (low / mid / high)
- Theatrical revenue projection
- VOD/streaming revenue projection
- Comparable film list with similarity score
- Sufficiency score

---

### 3.4 RiskAgent

**Role:** Qualitative risk reasoning agent. Produces a typed `RiskFlag[]` array by cross-referencing script scenes against censorship guidelines and cultural sensitivity reports.

**Owns:**
- Censorship risk detection (`CENSORSHIP`)
- Cultural sensitivity conflict detection (`CULTURAL_SENSITIVITY`)
- Market-fit risk assessment (`MARKET`)
- Scene-level page reference annotation
- Mitigation suggestion generation

**Inputs (from DataAgent):**
- Script scenes with themes and page references
- Censorship guidelines by country
- Cultural sensitivity report pages
- Theatrical window trends (for market risk)

**Outputs — `RiskFlag[]`, one flag per issue:**
- `category` — CENSORSHIP | CULTURAL_SENSITIVITY | MARKET
- `severity` — HIGH | MEDIUM | LOW
- `scene` — scene identifier
- `page` — page reference in source document
- `description` — what the conflict is
- `mitigation` — suggested edit or workaround
- `confidence` — float 0–1

---

### 3.5 StrategyAgent

**Role:** Prescriptive recommendation agent. Consumes ValuationAgent and RiskAgent outputs (from session state or live run) plus festival sentiment, marketing briefs, and VOD benchmarks to produce a complete release plan.

**Owns:**
- Release mode recommendation (theatrical / streaming-first / day-and-date / VOD-only)
- Release window timing
- Marketing spend range and allocation
- Platform priority ranking
- ROI scenario comparison (up to 3 scenarios)

**Inputs:**
- ValuationAgent output (from session state or current run)
- RiskAgent output (from session state or current run, when `strategy_needs_risk=true`)
- Festival sentiment and critic reviews
- Marketing brief
- VOD price benchmarks
- Theatrical window trends
- Actor Q-scores and narrative hooks from script

**Outputs:**
- Release mode recommendation with rationale
- Recommended release window (month + territory sequence)
- Marketing spend range (USD)
- Platform priority ranked list
- ROI scenario comparison table

---

### 3.6 DocumentRetrievalAgent (Sub-agent)

**Parent:** DataAgent. The only sub-agent in v1. Turns unstructured documents into structured evidence packages using **index-guided retrieval — not blind similarity search**.

**Owns:**
- Index planning (`IndexNavigator` → `RetrievalPlan`)
- Exact page/chunk fetching (`TargetedFetcher`)
- Sufficiency checking and expansion loops (`SufficiencyChecker`)
- Source reference annotation

**Inputs:**
- Retrieval intent (e.g., "censorship risk for India")
- Movie ID + document type + territory + themes/keywords

**Outputs:**
- Exact page/chunk content
- Source references (document name, page number, scene ID)
- Sufficiency status: `PASS` or `EXPAND`

---

## 4. Tool Reference

| Tool | Owner | Purpose | Called By |
|---|---|---|---|
| `IndexRegistry` | DocumentRetrievalAgent | Static catalog mapping Act→Scene→Page, Country→Rule→Page, Film→Reviewer→Page | DocumentRetrievalAgent |
| `IndexNavigator` | DocumentRetrievalAgent | Builds a RetrievalPlan — which document and which pages/chunks to fetch | DocumentRetrievalAgent |
| `TargetedFetcher` | DocumentRetrievalAgent | Executes RetrievalPlan, fetching only specified pages/chunks (no blind similarity search) | DocumentRetrievalAgent |
| `SufficiencyChecker` | DocumentRetrievalAgent | Checks if fetched content meets evidence threshold; expands and refetches if EXPAND | DocumentRetrievalAgent |
| `get_box_office_by_genre_territory()` | DataAgent | Historical global box office performance by genre and territory from PostgreSQL | DataAgent |
| `get_actor_qscore()` | DataAgent | Actor Q-scores and social media reach for talent power estimation | DataAgent |
| `get_theatrical_window_trends()` | DataAgent | Regional theatrical release-window trends and timing data | DataAgent |
| `get_exchange_rates()` | DataAgent | Current and historical currency exchange rates for MG normalization | DataAgent |
| `get_vod_price_benchmarks()` | DataAgent | Streaming/digital licensing price benchmarks by genre and territory | DataAgent |
| `get_comparable_films()` | DataAgent | Comparable film titles, MGs, and performance data for benchmarking | DataAgent |
| `mg_calculator_tool` | ValuationAgent | Converts comparable evidence, Q-scores, and genre multipliers into an MG estimate | ValuationAgent |
| `exchange_rate_tool` | ValuationAgent | Converts MG estimates to target currency using rates already fetched by DataAgent | ValuationAgent |
| `format_scorecard()` | Orchestrator | Builds the final structured JSON scorecard with all required output fields | Orchestrator |
| `source_citation_tool()` | Orchestrator | Attaches every claim to its supporting document name and page/chunk reference | Orchestrator |
| `financial_sanity_check()` | Orchestrator | Validates that MG estimates and revenue projections are within realistic bounds | Orchestrator (validation) |
| `hallucination_check()` | Orchestrator | Cross-checks all stated facts against retrieved evidence; flags unsupported claims | Orchestrator (validation) |
| `confidence_threshold_check()` | Orchestrator | Ensures data_sufficiency_score meets minimum threshold; warns user if not | Orchestrator (validation) |

---

## 5. Document Corpus Mapping

The document corpus covers **10 films** and **10 regions**.

| Document Type | Format | Retrieval Intent | Consumed By |
|---|---|---|---|
| synopses (×10) | unstructured text | Theme, narrative hooks, audience shape | ValuationAgent, StrategyAgent |
| scripts (×10) | .md + .pdf | Scene-level analysis, risky plot points, tone | RiskAgent, ValuationAgent |
| scripts_indexed/scenes.jsonl | JSON Lines | Precise scene lookup by Act/Scene ID | DocumentRetrievalAgent |
| scripts_indexed/pages.jsonl | JSON Lines | Page-level script content lookup | DocumentRetrievalAgent |
| reviews (×10) | unstructured text | Awards buzz, critic sentiment, prestige signal | StrategyAgent, ValuationAgent |
| cultural_sensitivity (×10) | unstructured text | Local taboo mapping, edit risk | RiskAgent |
| censorship_guidelines (×10 regions) | PDF | Regulatory conflict detection by country | RiskAgent |
| marketing (×10) | unstructured text | Campaign hooks, star power vs. story hook | StrategyAgent |
| docs/page_index/pages.jsonl | JSON Lines | Broad retrieval across all doc types | DocumentRetrievalAgent |
| docs/page_index/manifest.json | JSON | IndexRegistry — what exists and where | DocumentRetrievalAgent |

---

## 6. Runtime Flows — All Use Cases

---

### 6.1 Territory Valuation & MG Estimation

> **Query:** "What MG should we pay for Interstellar in Japan?"
>
> **Gate classification:** `workflow_request` — movie and territory both present
>
> **Routing:** analytical intent → `valuation`
>
> **Output format:** JSON scorecard
>
> **Agents:** Orchestrator → DataAgent → ValuationAgent → Validation → Scorecard

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify intent as `valuation` | movie=interstellar, territory=japan; `strategy_needs_risk=false` — StrategyAgent not called |
| 2 | DataAgent | Request valuation evidence | Triggers DocumentRetrievalAgent and DB tools simultaneously |
| 3 | DocumentRetrievalAgent | Retrieve: synopsis, script themes, marketing brief | IndexRegistry → IndexNavigator → TargetedFetcher |
| 4 | DataAgent (DB) | Fetch: Japan box office by genre, Q-scores, exchange rates JPY, VOD benchmarks, comparable films | 5 DB tool calls in parallel |
| 5 | DataAgent | Return unified evidence + citations + sufficiency score | Passed to ValuationAgent |
| 6 | ValuationAgent | Run `mg_calculator_tool` → `exchange_rate_tool` | Produces MG (low/mid/high) + theatrical + VOD projections |
| 7 | Orchestrator | Run validation: `financial_sanity_check`, `hallucination_check`, `confidence_threshold_check` | All three must pass before formatting |
| 8 | Orchestrator | `format_scorecard()` + `source_citation_tool()` | Returns structured JSON scorecard with citations |

---

### 6.2 Censorship & Cultural Risk Flagging

> **Query:** "Does Deadpool face censorship issues in Saudi Arabia?"
>
> **Gate classification:** `workflow_request` — movie and territory both present
>
> **Routing:** analytical intent → `risk`
>
> **Output format:** JSON scorecard
>
> **Agents:** Orchestrator → DataAgent (DocumentRetrievalAgent) → RiskAgent → Validation → Scorecard

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify intent as `risk` | movie=deadpool, territory=saudi_arabia; `strategy_needs_risk=false` — StrategyAgent not called |
| 2 | DataAgent | Request risk evidence | Document-heavy request; no DB calls needed for risk-only query |
| 3 | DocumentRetrievalAgent | Retrieve: script scenes, censorship_guidelines/saudi_arabia, cultural_sensitivity/deadpool | IndexNavigator builds scene-level RetrievalPlan |
| 4 | SufficiencyChecker | Check coverage of retrieved content | If EXPAND: DocumentRetrievalAgent widens page range and refetches |
| 5 | DataAgent | Return unified document evidence + citations | Page references included per scene |
| 6 | RiskAgent | Cross-reference script scenes with censorship rules | Produces typed `RiskFlag[]` with CENSORSHIP and CULTURAL_SENSITIVITY flags |
| 7 | Orchestrator | Run validation | `hallucination_check` verifies every flagged scene against retrieved source |
| 8 | Orchestrator | `format_scorecard()` with `risk_flags` section | Structured output includes scene/page ref, severity, mitigation per flag |

---

### 6.3 Festival Sentiment → Revenue Split Projection

> **Query:** "How will the Cannes buzz for this film affect its theatrical vs. streaming revenue?"
>
> **Gate classification:** `workflow_request` — movie resolved from session, intent is compound
>
> **Routing:** analytical intent → `valuation + strategy`
>
> **Output format:** JSON scorecard
>
> **Agents:** Orchestrator → DataAgent → ValuationAgent + StrategyAgent → Scorecard

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify as `valuation + strategy` compound | `strategy_needs_risk=false` — no new territory, risk not relevant to sentiment split |
| 2 | DataAgent | Retrieve: reviews (critic sentiment, festival scores), VOD price benchmarks, theatrical window trends | Mixed doc + DB request |
| 3 | DocumentRetrievalAgent | Retrieve: full review corpus for the film | IndexRegistry Film→Reviewer→Page index |
| 4 | ValuationAgent | Adjust revenue projection weights based on sentiment signal | High award buzz → higher theatrical multiplier |
| 5 | StrategyAgent | Model theatrical vs. streaming split using sentiment signal | Runs immediately after ValuationAgent — no RiskAgent dependency |
| 6 | Validation | `financial_sanity_check` + `confidence_threshold_check` | Sentiment adjustment must stay within historical bounds |
| 7 | Orchestrator | `format_scorecard()` with `sentiment_impact` field | Returns theatrical %, VOD %, and revenue delta vs. baseline |

---

### 6.4 Release Window & Marketing Strategy

> **Query:** "What is the optimal release window and marketing spend for La La Land in the UK?"
>
> **Gate classification:** `workflow_request` — movie and territory both present
>
> **Routing:** analytical intent → `full_scorecard` (all signals present)
>
> **Output format:** JSON scorecard
>
> **Agents:** Orchestrator → DataAgent → ValuationAgent ∥ RiskAgent → StrategyAgent → Scorecard

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify as `full_scorecard`; set `strategy_needs_risk=true` | All three specialist agents will run |
| 2 | DataAgent | Full evidence pull: all document types + all relevant DB tables | synopsis, script, reviews, marketing brief, censorship, cultural sensitivity + DB box office, Q-scores, window trends, VOD benchmarks, comparables |
| 3 | ValuationAgent ∥ RiskAgent | Run in parallel | ValuationAgent: MG + revenue estimates. RiskAgent: `RiskFlag[]` for UK |
| 4 | StrategyAgent | Wait for both to complete; consume outputs + marketing brief + Q-scores | Outputs: release mode, window timing, marketing spend range, platform priority list |
| 5 | Validation | All three validation tools run | Validates financial figures, citations, and confidence |
| 6 | Orchestrator | `format_scorecard()` — full schema | Returns all required scorecard fields |

---

### 6.5 Contextual Follow-up: Skip Theatrical

> **Query:** "If we skip theatrical in France and go straight to streaming, how does the ROI change?"
>
> **Gate classification:** `workflow_followup` — active session, movie + territory resolved
>
> **Routing:** reuse session state; StrategyAgent only
>
> **Output format:** JSON scorecard with `scenario_comparison` field
>
> **Agents:** Orchestrator (loads session) → StrategyAgent → Validation → Updated Scorecard

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_followup`; detect scenario override `theatrical=FALSE` | Session context check confirms movie + France territory already resolved |
| 2 | Orchestrator | Set `scenario_overrides: {theatrical: false, streaming: true}` in session state | DataAgent call only if France-specific VOD benchmarks not yet cached |
| 3 | StrategyAgent | Recalculate with constraint applied | Reuses existing ValuationAgent output from session state; applies VOD-only revenue model |
| 4 | StrategyAgent | Generate ROI comparison: Scenario A (theatrical) vs. Scenario B (streaming-first) | Includes revenue delta, window difference, marketing spend delta |
| 5 | Validation | `financial_sanity_check` + `confidence_threshold_check` on new scenario | Validates the scenario delta is within realistic bounds |
| 6 | Orchestrator | `format_scorecard()` with `scenario_comparison` field | Returns updated scorecard with both scenarios and recommendation |

---

### 6.6 Contextual Follow-up: Underperformance Explanation

> **Query:** "Why is this film projected to underperform in Germany despite high festival scores?"
>
> **Gate classification:** `workflow_followup` — active session, contradiction between prior outputs detected
>
> **Routing:** targeted specialist run; DataAgent for Germany-specific evidence only
>
> **Output format:** JSON scorecard + natural-language explanation
>
> **Agents:** Orchestrator (loads session) → DataAgent → RiskAgent ∥ ValuationAgent → Explanation

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_followup`; detect explanation intent for Germany | Recognizes mismatch between high sentiment score and low revenue projection in session state |
| 2 | DataAgent | Targeted request: Germany box office genre patterns, cultural_sensitivity/Germany, Germany censorship guidelines | Focused call — only Germany-specific evidence not already in session |
| 3 | RiskAgent ∥ ValuationAgent | Run in parallel for Germany specifically | RiskAgent: MARKET and CULTURAL_SENSITIVITY flags. ValuationAgent: comparable film divergence analysis |
| 4 | Orchestrator | Synthesize contradiction across both outputs + DataAgent evidence | Combines `RiskFlag[]` + ValuationAgent notes into a coherent explanation |
| 5 | Orchestrator | Return explanation + updated scorecard with Germany risk detail | `source_citation_tool` attaches page refs to each claim in explanation |

---

### 6.7 Multi-Territory Comparison

> **Query:** "Compare MG estimates for Latin America vs. South Korea for this film."
>
> **Gate classification:** `workflow_request` — movie resolved, two territories explicit
>
> **Routing:** analytical intent → `valuation` across two territories in parallel
>
> **Output format:** JSON scorecard with `multi_territory_comparison` section

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; split into two parallel valuation requests | territory_A=latin_america, territory_B=south_korea; `strategy_needs_risk=false` for each |
| 2 | DataAgent | Two parallel evidence pulls — one per territory | Separate DB calls per territory; shared document retrieval for the film |
| 3 | ValuationAgent ∥ ValuationAgent | Run twice in parallel — once per territory | `exchange_rate_tool` converts both to USD for comparison |
| 4 | RiskAgent ∥ RiskAgent | Run once per territory in parallel | Different censorship guidelines; cultural sensitivity may differ significantly |
| 5 | Orchestrator | Merge outputs into comparison scorecard | Builds `territory_a` and `territory_b` fields side-by-side |
| 6 | Validation | Run validation on each territory independently | `confidence_threshold_check` must pass for both |
| 7 | Orchestrator | `format_scorecard()` with `multi_territory_comparison` section | Returns prioritization recommendation based on MG delta and risk profile |

---

### 6.8 Full Distribution Scorecard

> **Query:** "Give me a complete distribution scorecard for this film."
>
> **Gate classification:** `workflow_request` — explicit full scorecard request
>
> **Routing:** `full_scorecard` intent; all specialist agents run
>
> **Output format:** JSON scorecard — all fields populated

`full_scorecard` is only triggered by an explicit user request. The Orchestrator never infers this intent from ambiguous input.

**Flow summary:**
1. Orchestrator classifies as `full_scorecard`; sets `strategy_needs_risk=true`
2. DataAgent pulls the full corpus: all document types for the film + all relevant DB tables
3. ValuationAgent and RiskAgent run in parallel once evidence is ready
4. StrategyAgent runs after both complete
5. Full validation pipeline runs (all three tools)
6. `format_scorecard()` builds the complete schema with all fields populated

---

### 6.9 Acquisition Bid Support

> **Query:** "What is the maximum we should bid to acquire this film for global distribution?"
>
> **Gate classification:** `workflow_request` — movie resolved, global scope implied
>
> **Routing:** `valuation + risk` across all available territories
>
> **Output format:** JSON scorecard with `acquisition_recommendation` section

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify as `valuation + risk` across all territories | `strategy_needs_risk=false` — no release strategy needed |
| 2 | DataAgent | Full evidence pull: all territories in DB, script themes, all risk documents | Broadest DataAgent call — full territory scope |
| 3 | ValuationAgent ∥ RiskAgent | Run in parallel | ValuationAgent: MG per territory. RiskAgent: HIGH severity flags that trigger bid discount |
| 4 | ValuationAgent | Apply risk discount to global MG aggregate | HIGH severity flags reduce bid ceiling by a configurable discount factor |
| 5 | Validation | Full validation pipeline | `financial_sanity_check` critical — acquisition price must be within historical precedent |
| 6 | Orchestrator | `format_scorecard()` with `acquisition_recommendation` section | Includes bid ceiling, risk-adjusted ceiling, territory breakdown, comparables |

---

### 6.10 VOD vs. Theatrical ROI Modelling

> **Query:** "Model the ROI for a VOD-only release versus a wide theatrical release for this film in Asia."
>
> **Gate classification:** `workflow_request` — movie resolved, two explicit scenarios
>
> **Routing:** `strategy + valuation` compound; `strategy_needs_risk=false`
>
> **Output format:** JSON scorecard with `roi_scenarios` field

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request`; classify as `strategy + valuation`; set `strategy_needs_risk=false` | VOD-only scenario — risk flags do not materially affect the model |
| 2 | DataAgent | Retrieve: Asia box office trends, VOD benchmarks for Asia, theatrical window trends Asia | Both DB tables consulted |
| 3 | ValuationAgent | Run revenue model for Scenario A (theatrical): MG + P&A cost estimate + theatrical revenue | P&A cost approximated from marketing brief and historical comparables |
| 4 | ValuationAgent | Run revenue model for Scenario B (VOD-only): licensing benchmarks, no P&A cost | Uses `get_vod_price_benchmarks()` output |
| 5 | StrategyAgent | Compute ROI for both scenarios, break-even point, NPV comparison | Runs immediately after ValuationAgent — no RiskAgent dependency |
| 6 | Validation | `financial_sanity_check` on both scenario models | Validates P&A estimates and VOD licensing numbers against benchmarks |
| 7 | Orchestrator | `format_scorecard()` with `roi_scenarios` field | Returns scenario_comparison table with ROI %, break-even month, and recommendation |

---

### 6.11 Greeting

> **Query:** "Hi", "Hello", "Good morning"
>
> **Gate classification:** `greeting` — no workflow signal, no session context
>
> **Routing:** direct response — no agents called
>
> **Output format:** natural-language response

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `greeting` | Routing policy rule 1: respond directly |
| 2 | Orchestrator | Generate natural-language welcome response | Introduces MarketReel capabilities at a high level; invites the user to name a film or ask a question |

**Example response:**
> "Hello! I'm MarketReel, your global film distribution and acquisition assistant. I can help you estimate Minimum Guarantees, flag censorship risks, model theatrical vs. streaming ROI, and build full distribution scorecards. Which film are you evaluating today?"

---

### 6.12 Help Request

> **Query:** "What can you help with?", "What do you do?", "Show me what you can analyze"
>
> **Gate classification:** `help` — no workflow signal
>
> **Routing:** direct response — no agents called
>
> **Output format:** natural-language capability summary

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `help` | Routing policy rule 2: respond with capability summary |
| 2 | Orchestrator | Generate natural-language capability summary | Lists the six core capability areas with brief examples; no scorecard returned |

**Example response:**
> "Here's what I can help you with:
> - **Territory Valuation** — MG estimates for any region (e.g., 'What should we pay for Dune in Japan?')
> - **Censorship & Risk Flagging** — scene-level flags with mitigations (e.g., 'Does this film have issues in Saudi Arabia?')
> - **Festival Sentiment** — how awards buzz affects your theatrical vs. streaming split
> - **Release Strategy** — window timing, marketing spend, and platform priority
> - **Scenario Modelling** — skip theatrical, compare territories, VOD vs. wide release
> - **Full Scorecard** — complete distribution analysis in one go
>
> Just tell me which film and territory you're working on."

---

### 6.13 Clarification Flow

> **Query examples:** "Tell me about this film" (no film named) · "What about Germany?" (no session) · "Analyze it" (no context)
>
> **Gate classification:** `clarification` — workflow signal present but required context missing
>
> **Routing:** clarification prompt — no agents called
>
> **Output format:** single focused clarification question

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → checks session context first | No active session or missing movie/territory |
| 2 | Orchestrator | Routing policy rules 6–8: detect what is missing | Missing movie name? Missing territory? Weak analytical signal with no context? |
| 3 | Orchestrator | Generate a single clarification question | Never asks multiple questions at once; asks for the most critical missing piece first |

**Decision table for clarification questions:**

| Missing | Clarification Prompt |
|---|---|
| Movie name | "Which film are you looking to evaluate?" |
| Territory | "Which territory or region would you like to analyze?" |
| Both | "Which film and territory are you working on?" |
| Weak signal, no context | "I can help with valuation, censorship risk, or release strategy — which would you like to start with, and for which film?" |

---

### 6.14 Low-Confidence Response

> **Trigger:** `confidence_threshold_check()` returns score between 0.3 and 0.59
>
> **Gate classification:** `workflow_request` or `workflow_followup` — analytical turn that completes but with thin data
>
> **Output format:** JSON scorecard with `confidence_warning` field populated

**This is distinct from the insufficient-data case.** Low confidence means data exists and was retrieved, but the volume or coverage of supporting evidence is below the ideal threshold. The scorecard is still returned.

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1–N | Normal workflow | All specialist agents run and produce outputs | Scorecard is generated as usual |
| N+1 | Orchestrator | `confidence_threshold_check()` returns score 0.3–0.59 | Below the 0.6 threshold but above the 0.3 floor |
| N+2 | Orchestrator | Populate `confidence_warning` field in scorecard | Warning explains which data was thin and what the user should verify manually |
| N+3 | Orchestrator | Return scorecard with warning surfaced prominently | Scorecard is usable but flagged; user can choose to proceed or provide more context |

**Example `confidence_warning`:**
> "Confidence is moderate (score: 0.52). The MG estimate relies on only 2 comparable films for this territory. The theatrical revenue projection would benefit from more recent genre performance data. Treat the high-end estimate with caution."

---

### 6.15 Insufficient-Data Handling

> **Trigger:** `confidence_threshold_check()` returns score below 0.3, or a required document or DB table is entirely missing
>
> **Gate classification:** `workflow_request` — analytical turn that cannot be completed reliably
>
> **Output format:** natural-language response explaining the gap; no scorecard returned

**This is distinct from the low-confidence case.** Insufficient data means a required input is absent — a film has no script in the corpus, a territory has no box office history, or a censorship guideline document is missing entirely.

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_request` | Routing proceeds normally |
| 2 | DataAgent | Returns evidence package with `data_sufficiency_score < 0.3` or hard missing-data flags | Identifies specifically which document type or DB table is absent |
| 3 | Orchestrator | `confidence_threshold_check()` fails at floor level | Orchestrator does not proceed to specialist agents |
| 4 | Orchestrator | Generate natural-language response | Explains exactly what is missing, which analysis is affected, and what the user can do to resolve it |

**Low confidence vs. insufficient data — summary:**

| Condition | Score Range | Response |
|---|---|---|
| Sufficient data | ≥ 0.6 | Scorecard returned, no warning |
| Low confidence | 0.3 – 0.59 | Scorecard returned + `confidence_warning` populated |
| Insufficient data | < 0.3 or hard missing | No scorecard; natural-language explanation of the gap |

**Example natural-language response:**
> "I can't generate a reliable valuation for this film in Nigeria. The corpus doesn't include a script or synopsis for this title, and Nigeria box office history isn't available in the database. I can still run a censorship risk assessment using the regional guidelines, or give you a rough estimate based on comparable African market data if you'd like to proceed with caveats."

---

### 6.16 Explainability Request

> **Query:** "Why did you estimate that MG?", "How did you arrive at that risk flag?", "Walk me through your reasoning"
>
> **Gate classification:** `workflow_followup` — references prior output in session state
>
> **Routing:** session-state retrieval only; no agents re-run; no DataAgent call
>
> **Output format:** natural-language explanation with inline citations

Explainability requests reference prior outputs already in session state. The Orchestrator does not re-run any agents. It retrieves the relevant agent output and citation package from session state and constructs a natural-language explanation.

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_followup`; detect explainability intent | Keywords: "why", "how did you", "explain", "walk me through", "what data" |
| 2 | Orchestrator | Load relevant output from session state | Identifies which agent output is being questioned (ValuationAgent, RiskAgent, or StrategyAgent) |
| 3 | Orchestrator | Load citation package for that output from session state | Citations already attached by `source_citation_tool()` in prior turn |
| 4 | Orchestrator | Generate natural-language explanation | Walks through the reasoning step by step; cites source documents and DB rows inline |

**Example natural-language response:**
> "The MG estimate of $1.85M for Japan was based on three factors:
> 1. **Comparable films**: Arrival ($1.6M, similarity 0.84) and Gravity ($2.1M, similarity 0.71) set the range. Source: `comparable_films` DB table.
> 2. **Genre multiplier**: Sci-fi performs at 1.4x the Japan market baseline. Source: `box_office_history` DB, genre=sci-fi, territory=japan.
> 3. **Talent signal**: The lead actor's Japan Q-score is 72, above the 65 threshold for a positive multiplier. Source: `actor_qscores` DB.
>
> These three inputs produced a mid estimate of $1.85M with a confidence of 0.78."

---

### 6.17 Evidence Inspection

> **Query:** "Show me the sources you used", "What documents did you pull?", "Which pages did you reference for the censorship flags?"
>
> **Gate classification:** `workflow_followup` — references prior session state
>
> **Routing:** session-state retrieval only; no agents re-run; no DataAgent call
>
> **Output format:** natural-language response surfacing raw citations from session state

Evidence inspection is a read from session state, not a re-computation. The Orchestrator surfaces the raw citation package from `source_citation_tool()` in the prior turn, formatted for readability.

**This is distinct from explainability.** Explainability explains the *reasoning*. Evidence inspection shows the *raw sources* — document names, page numbers, chunk references, and DB rows.

| Step | Agent | Action | Detail |
|---|---|---|---|
| 1 | Orchestrator | Conversational Gate → `workflow_followup`; detect evidence inspection intent | Keywords: "show me sources", "what documents", "which pages", "where did you get" |
| 2 | Orchestrator | Load citation package from session state | `citations[]` array from the most recent scorecard in session |
| 3 | Orchestrator | Filter citations by agent or topic if specified | e.g., "censorship sources only" → filter where `agent=RiskAgent` |
| 4 | Orchestrator | Format and return citations as natural-language response | Groups by source document; lists page/chunk refs and the claim each supports |

**Example natural-language response:**
> "Here are the sources used for the censorship risk assessment for Saudi Arabia:
>
> **censorship_guidelines/saudi_arabia.pdf**
> - Page 4: prohibition on depictions of alcohol consumption → flags Scene 12, Act 2
> - Page 7: restrictions on sexual content → flags Scene 18, Act 3
> - Page 11: prohibition on content undermining social order → flags Scene 24, Act 4
>
> **cultural_sensitivity/deadpool.md**
> - Section 2: notes on graphic violence reception in GCC markets → flagged as MARKET risk, severity MEDIUM
>
> **scripts_indexed/scenes.jsonl**
> - Scene 12 (page 34): retrieved for cross-reference with censorship rule on page 4
> - Scene 18 (page 51): retrieved for cross-reference with censorship rule on page 7"

---

## 7. Session State & Follow-up Handling

MarketReel maintains session state across turns using Google ADK's built-in session management.

| State Field | Description |
|---|---|
| `movie_id` | Resolved film identifier — persists across all turns in a session |
| `territory` | Current territory context — updated when user changes territory |
| `turn_type_history` | List of prior turn types — used by Conversational Gate on ambiguous inputs |
| `valuation_output` | Last ValuationAgent result — reused by StrategyAgent in follow-ups |
| `risk_output` | Last RiskAgent result — reused when `strategy_needs_risk=false` |
| `strategy_output` | Last StrategyAgent result — reused in scenario follow-ups |
| `data_evidence` | Full evidence package from DataAgent — cached to avoid redundant retrieval |
| `citations` | Full citation package from `source_citation_tool()` — used by explainability and evidence inspection flows |
| `scenario_overrides` | Active scenario constraints (e.g., `theatrical=false`) — set on follow-up, cleared on new film |
| `strategy_needs_risk` | Boolean flag set by Orchestrator at classification time — not persisted between turns |
| `last_scorecard` | Most recent scorecard output — referenced by follow-up and explainability turns |

### Follow-up Resolution Logic

- **Movie and territory unchanged:** load existing session state; skip DataAgent unless scenario requires new evidence
- **Territory changes:** request only territory-specific evidence from DataAgent; reuse document evidence for the film
- **Scenario override added:** set `scenario_overrides` in state; call StrategyAgent only; reuse cached ValuationAgent and RiskAgent outputs
- **Movie changes:** clear all session state except user preferences; start fresh
- **Ambiguous follow-up with no session context:** Conversational Gate routes to `clarification`, not `workflow_followup`

---

## 8. Validation Pipeline

The Orchestrator runs three validation tools in sequence before calling `format_scorecard()`. Validation only runs for analytical turns that produce a scorecard. Conversational, clarification, explainability, and evidence inspection turns bypass validation entirely.

| Tool | What It Checks | Failure Behaviour |
|---|---|---|
| `financial_sanity_check()` | MG estimates and revenue projections must be within ±3 standard deviations of comparable film data | Outliers flagged as HIGH severity; scorecard still returned with flag |
| `hallucination_check()` | Cross-references every stated fact (scene references, box office figures, Q-scores) against the retrieved evidence package | Unsupported claims removed from scorecard before formatting |
| `confidence_threshold_check()` | Checks `data_sufficiency_score` from DataAgent. Score < 0.6 → populate `confidence_warning`. Score < 0.3 → abort scorecard; return insufficient-data response instead | See UC-6.14 and UC-6.15 for response formats |

---

## 9. Structured JSON Scorecard Schema

Structured JSON scorecards are returned only for analytical turns. Conversational, clarification, explainability, and evidence inspection turns return natural-language responses.

| Field | Type | Description |
|---|---|---|
| `movie_id` | string | Resolved film identifier |
| `territory` | string \| string[] | Target territory or list of territories |
| `intent` | enum | `valuation` \| `risk` \| `strategy` \| `full_scorecard` |
| `revenue_by_territory` | object[] | Array of `{territory, theatrical_usd, vod_usd, total_usd, currency, exchange_rate}` |
| `mg_estimate` | object | `{low: number, mid: number, high: number, currency: string, confidence: float}` |
| `comparable_films` | object[] | Array of `{title, territory, mg, similarity_score}` |
| `risk_flags` | RiskFlag[] | Array of `{category, severity, scene, page, description, mitigation, confidence}` |
| `release_recommendation` | object | `{mode, window_start, window_end, platform_priority: string[], marketing_spend_usd: {low, high}}` |
| `scenario_comparison` | object[] | Array of `{scenario_name, roi_pct, revenue_usd, cost_usd, recommendation}` |
| `acquisition_recommendation` | object | `{recommended_price_usd: {low, mid, high}, risk_adjusted_ceiling_usd: number}` |
| `sentiment_impact` | object | `{festival_score: float, sentiment_label: string, theatrical_uplift_pct: float}` |
| `data_sufficiency_score` | float | 0.0–1.0 score from DataAgent; triggers warning if < 0.6 |
| `citations` | object[] | Array of `{claim, source_document, page_or_chunk, agent}` |
| `confidence_warning` | string \| null | Populated when score is 0.3–0.59; null when score ≥ 0.6 |
| `generated_at` | ISO8601 | Timestamp of scorecard generation |

### Example Scorecard (abbreviated)

```json
{
  "movie_id": "interstellar",
  "territory": "japan",
  "intent": "valuation",
  "mg_estimate": {
    "low": 1200000,
    "mid": 1850000,
    "high": 2400000,
    "currency": "USD",
    "confidence": 0.78
  },
  "revenue_by_territory": [
    {
      "territory": "japan",
      "theatrical_usd": 4200000,
      "vod_usd": 800000,
      "total_usd": 5000000,
      "currency": "JPY",
      "exchange_rate": 149.5
    }
  ],
  "comparable_films": [
    { "title": "Arrival", "territory": "japan", "mg": 1600000, "similarity_score": 0.84 },
    { "title": "Gravity", "territory": "japan", "mg": 2100000, "similarity_score": 0.71 }
  ],
  "risk_flags": [],
  "release_recommendation": {
    "mode": "theatrical",
    "window_start": "2025-03",
    "window_end": "2025-05",
    "platform_priority": ["Netflix JP", "Amazon Prime JP", "local VOD"],
    "marketing_spend_usd": { "low": 300000, "high": 550000 }
  },
  "data_sufficiency_score": 0.87,
  "citations": [
    {
      "claim": "Japan sci-fi genre multiplier 1.4x",
      "source_document": "box_office_history",
      "page_or_chunk": "DB row genre=sci-fi territory=japan",
      "agent": "ValuationAgent"
    }
  ],
  "confidence_warning": null,
  "generated_at": "2026-03-11T10:00:00Z"
}
```

---

## 10. Google ADK Implementation Notes

### Agent Registration

- Register `MarketLogicOrchestrator` as the **root agent** in your ADK application
- Register `DataAgent`, `ValuationAgent`, `RiskAgent`, `StrategyAgent` as sub-agents of the root
- Register `DocumentRetrievalAgent` as a sub-agent of `DataAgent` only
- All tools are registered on their owning agent as ADK `FunctionTool` instances
- The Conversational Gate logic lives inside the Orchestrator's instruction prompt and pre-routing step — it is not a separate agent

### Tool Implementation

- DB tools map directly to PostgreSQL queries wrapped in ADK `FunctionTool` with typed schemas
- Document tools (`IndexRegistry`, `IndexNavigator`, `TargetedFetcher`, `SufficiencyChecker`) operate on local filesystem JSON Lines indexes and PDF/MD files
- `mg_calculator_tool` and `exchange_rate_tool` are pure Python computation functions — no external API calls
- Validation tools run synchronously in the Orchestrator before `format_scorecard()`
- Validation tools are **skipped entirely** for conversational, clarification, explainability, and evidence inspection turns

### Session State

- Use ADK's built-in `Session` object to store all session state fields defined in Section 7
- Pass session ID with every user query; Orchestrator reads and updates state via ADK session API
- Scenario overrides are stored as a dict in session state and injected into agent prompts
- `strategy_needs_risk` is computed fresh at classification time each turn — it is not persisted

### Parallelism

- `ValuationAgent` and `RiskAgent` always run in **parallel** when both are needed
- `StrategyAgent` waits for both only when `strategy_needs_risk=true`; otherwise it runs immediately after `ValuationAgent`
- DataAgent's DB tool calls can be parallelized using `asyncio` within the DataAgent implementation
- For `full_scorecard` queries: DataAgent → (ValuationAgent ∥ RiskAgent) → StrategyAgent is the canonical execution order

### Model Configuration

| Agent | Recommended Model | Reason |
|---|---|---|
| MarketLogicOrchestrator | Gemini 1.5 Pro | Long context for full session state; Conversational Gate requires nuanced turn classification |
| ValuationAgent | Gemini 1.5 Pro | Complex financial reasoning |
| RiskAgent | Gemini 1.5 Pro | Long script + guidelines cross-reference |
| StrategyAgent | Gemini 1.5 Pro | Multi-signal synthesis |
| DocumentRetrievalAgent | Gemini 1.5 Flash | Fast, lower-cost for index operations |
| DataAgent | Gemini 1.5 Flash | Routing and packaging — no heavy reasoning |

- Set `temperature=0` for all validation tools to ensure deterministic sanity checks
- Set `temperature=0` for the Conversational Gate classification step inside the Orchestrator
- Set `temperature=0.2` for specialist agents to allow flexibility in reasoning while staying grounded

### Suggested Directory Structure

```
marketreel/
├── agents/
│   ├── orchestrator.py              # MarketLogicOrchestrator
│   ├── conversational_gate.py       # Gate logic (turn classification + session context check)
│   ├── routing_policy.py            # Routing rules
│   ├── output_policy.py             # Response format selection
│   ├── data_agent.py                # DataAgent
│   ├── valuation_agent.py           # ValuationAgent
│   ├── risk_agent.py                # RiskAgent
│   ├── strategy_agent.py            # StrategyAgent
│   └── document_retrieval_agent.py  # DocumentRetrievalAgent (sub-agent)
├── tools/
│   ├── document/
│   │   ├── index_registry.py
│   │   ├── index_navigator.py
│   │   ├── targeted_fetcher.py
│   │   └── sufficiency_checker.py
│   ├── db/
│   │   ├── box_office.py
│   │   ├── actor_qscore.py
│   │   ├── theatrical_windows.py
│   │   ├── exchange_rates.py
│   │   ├── vod_benchmarks.py
│   │   └── comparable_films.py
│   ├── valuation/
│   │   ├── mg_calculator.py
│   │   └── exchange_rate_tool.py
│   └── validation/
│       ├── financial_sanity_check.py
│       ├── hallucination_check.py
│       └── confidence_threshold_check.py
├── schemas/
│   ├── scorecard.py                 # Pydantic model for JSON scorecard
│   ├── risk_flag.py                 # Pydantic model for RiskFlag
│   └── session_state.py             # Pydantic model for session state
├── docs/                            # Document corpus
│   ├── synopses/
│   ├── scripts/
│   ├── reviews/
│   ├── cultural_sensitivity/
│   ├── censorship_guidelines/
│   ├── marketing/
│   ├── scripts_indexed/
│   │   ├── scenes.jsonl
│   │   └── pages.jsonl
│   └── page_index/
│       ├── pages.jsonl
│       └── manifest.json
├── session/
│   └── state.py                     # Session state management
└── main.py                          # ADK app entry point
```

