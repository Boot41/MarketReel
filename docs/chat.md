## plan

Build the system around **one ADK root orchestrator** and **four specialist agents**, with **one retrieval sub-agent** under the data layer:

* `MarketLogicOrchestrator`
* `DataAgent`
* `ValuationAgent`
* `RiskAgent`
* `StrategyAgent`
* sub-agent: `DocumentRetrievalAgent` 

This matches your planned architecture and fits the requirement to support natural-language film acquisition analysis over both PostgreSQL data and local docs.  

## 1) ADK architecture to build

### Root agent: `MarketLogicOrchestrator`

This should be the only user-facing ADK entrypoint. It should:

* receive the user query
* detect whether the turn is casual/help/clarification/workflow
* resolve movie, territory, and scenario
* decide which specialist agents to run
* load and save session state
* run validation
* return the final scorecard or clarification response 

### Data layer: `DataAgent`

This should be the **single gateway** for all external data access. No other agent should touch docs or DB directly. It should:

* accept typed evidence requests
* decide whether docs, DB, or both are needed
* call `DocumentRetrievalAgent`
* call DB tools
* merge evidence, citations, and sufficiency score into one context object 

### Retrieval sub-agent: `DocumentRetrievalAgent`

Keep this under `DataAgent`. It should:

* build retrieval plans using your document indexes
* fetch exact pages/chunks
* run sufficiency checks
* expand/refetch if evidence is weak 

### Specialist agents

`ValuationAgent` should estimate MG, theatrical revenue, VOD/streaming revenue, confidence interval, and comp list. `RiskAgent` should produce typed risk flags for censorship, cultural sensitivity, and market risk. `StrategyAgent` should recommend release mode, release window, marketing spend, platform priority, and scenario comparisons. 

## 2) Tools to keep where they belong

### Under `DocumentRetrievalAgent`

* `IndexRegistry`
* `IndexNavigator`
* `TargetedFetcher`
* `SufficiencyChecker` 

### Under `DataAgent`

* `get_box_office_by_genre_territory()`
* `get_actor_qscore()`
* `get_theatrical_window_trends()`
* `get_exchange_rates()`
* `get_vod_price_benchmarks()`
* `get_comparable_films()` 

### Under `ValuationAgent`

* `mg_calculator_tool`
* `exchange_rate_tool` 

### Under `MarketLogicOrchestrator`

* `format_scorecard()`
* `source_citation_tool()`
* `financial_sanity_check()`
* `hallucination_check()`
* `confidence_threshold_check()` 

## 3) Runtime flow to implement

### Fresh analytical request

For a query like “How will Interstellar perform in India?” the system should:

1. `MarketLogicOrchestrator` resolves movie, territory, and intent.
2. It sends a typed evidence request to `DataAgent`.
3. `DataAgent` calls `DocumentRetrievalAgent` for scripts, reviews, marketing, cultural sensitivity, censorship docs.
4. `DataAgent` calls the DB tools for box office patterns, Q-scores, window trends, FX, VOD benchmarks, comps.
5. `DataAgent` returns one evidence bundle with citations and sufficiency score.
6. Orchestrator invokes `ValuationAgent`, `RiskAgent`, and `StrategyAgent` as needed.
7. Orchestrator runs validation.
8. Orchestrator formats the structured scorecard. 

### Follow-up scenario request

For a follow-up like “If we skip theatrical in India and go straight to streaming, how does ROI change?” the system should:

1. load session state
2. recognize scenario follow-up
3. reuse prior valuation/risk where possible
4. ask `DataAgent` only for missing evidence
5. rerun `StrategyAgent`
6. rerun validation
7. return updated scorecard 

## 4) Very important production change

Do **not** let unknown text fall back to `full_scorecard`.

Your orchestrator should first classify the turn into:

* greeting / casual
* help
* clarification
* workflow request
* workflow follow-up

Only workflow requests should trigger the full MarketLogic pipeline. This prevents `"hi"` or `"thanks"` from invoking all agents.

So the safe order is:

1. conversational gate
2. entity resolution
3. session reuse
4. workflow routing
5. validation
6. output
7. persistence

## 5) All supported use cases

Your system should support all of these, not just the main scorecard flow.

### Core business use cases

* full distribution scorecard
* territory valuation / MG pricing
* theatrical revenue forecast
* VOD / streaming revenue forecast
* censorship risk review
* cultural sensitivity risk review
* market-fit risk review
* festival sentiment impact on digital vs theatrical split
* release mode recommendation: theatrical-first vs streaming-first
* release window recommendation
* marketing spend recommendation
* platform priority recommendation
* scenario comparison, such as skip theatrical
* comparative territory analysis
* explanation of why a market is weak or strong
* evidence-backed justification for the recommendation

These are all directly supported by the requirement and your agent definitions.  

### Conversational/support use cases

* greeting
* help / “what can you do?”
* clarification when movie or territory is missing
* contextual follow-up using prior session state
* partial workflow requests such as “only valuation” or “only risk”
* scenario override like `streaming_first` vs `theatrical_first`

These should be handled by the orchestrator without always invoking every agent.

## 6) Data and retrieval strategy

Use your corpus in a targeted way:

* `synopses`, `scripts`, and indexed script pages/scenes for themes, tone, risky scenes, and hooks
* `reviews` for awards buzz and sentiment
* `cultural_sensitivity` and `censorship_guidelines_countries` for risk analysis
* `marketing` plus `get_actor_qscore()` for campaign strategy
* page indexes for broad retrieval
* scene/page indexes for precise script analysis 

This is the right fit for the requirement’s dependence on scripts, reviews, censorship guidelines, cultural reports, and marketing briefs. 

## 7) Ownership boundaries

Keep the boundaries strict:

* `MarketLogicOrchestrator`: routing, state, validation, formatting
* `DataAgent`: all retrieval, evidence packaging, citations, sufficiency
* `DocumentRetrievalAgent`: document plan/fetch/refine
* `ValuationAgent`: MG and revenue reasoning
* `RiskAgent`: censorship/cultural/market risk reasoning
* `StrategyAgent`: release and marketing recommendations, ROI comparison 

This is the most important design rule after the conversational gate.

## 8) Output contract

Every analytical response should end in a structured JSON scorecard with:

* projected revenue by territory
* risk flags
* recommended acquisition price
* release timeline

And in practice also:

* citations
* confidence
* warnings from validation  

## 9) Recommended implementation phases

### Phase 1

Build:

* orchestrator
* data agent
* document retrieval sub-agent
* valuation path
* scorecard formatter
* session state

### Phase 2

Add:

* risk agent
* censorship/cultural risk output
* validation layer
* source citation attachment

### Phase 3

Add:

* strategy agent
* release mode/window/marketing recommendations
* scenario comparison and follow-up optimization

### Phase 4

Harden:

* confidence thresholds
* hallucination checks
* partial reruns
* better clarification logic
* production telemetry

## one line

**Use one ADK orchestrator with a conversational gate, one centralized `DataAgent`, one retrieval sub-agent, and three specialist reasoning agents; support both analytical and conversational use cases, and reserve full scorecard generation only for clearly valid workflow requests.**  
