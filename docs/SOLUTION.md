# 1. Agent architecture

## Real agents

* `MarketLogicOrchestrator`
* `DataAgent`
* `ValuationAgent`
* `RiskAgent`
* `StrategyAgent`

## Sub-agent

* `DocumentRetrievalAgent`

## Tools

### Document tools

* `IndexRegistry`
* `IndexNavigator`
* `TargetedFetcher`
* `SufficiencyChecker`

### DB tools

* `get_box_office_by_genre_territory()`
* `get_actor_qscore()`
* `get_theatrical_window_trends()`
* `get_exchange_rates()`
* `get_vod_price_benchmarks()`
* `get_comparable_films()`

### Valuation tools

* `mg_calculator_tool`
* `exchange_rate_tool`

### Utility / validation tools

* `format_scorecard()`
* `source_citation_tool()`
* `financial_sanity_check()`
* `hallucination_check()`
* `confidence_threshold_check()`

---

# 2. What each agent does

## A. `MarketLogicOrchestrator`

This is the **top-level controller**.

### Job

* receives the user’s natural-language question
* resolves the movie and target territory
* classifies the request
* decides which agents to run
* stores session state
* triggers validation
* returns the final scorecard

### Why it exists

The architecture doc explicitly defines it as the top-level controller that routes queries, manages session state, and calls validation before every response. 

### Typical user questions it should handle

* “How will Interstellar perform in India?”
* “What MG should we pay for La La Land in Japan?”
* “Does Deadpool face censorship issues in Saudi Arabia?”
* “If we skip theatrical in Germany, how does ROI change?”

These question types are consistent with the requirement’s valuation, censorship, sentiment, release strategy, and contextual follow-up sections. 

### Inputs

* user query
* session id
* movie id or movie name
* optional scenario override

### Outputs

* final structured scorecard
* explanation
* citations
* warning if confidence is low

---

## B. `DataAgent`

This is the **single data gateway**.

### Job

* receives a typed evidence request from another agent
* decides whether the answer needs docs, DB data, or both
* calls `DocumentRetrievalAgent`
* calls DB tools
* combines everything into one evidence package
* attaches citations
* returns a `data_sufficiency_score`

The architecture doc explicitly says all external data access flows through `DataAgent`, and no other agent should touch the database or documents directly. 

### Why it exists

Without `DataAgent`, `ValuationAgent`, `RiskAgent`, and `StrategyAgent` would each duplicate retrieval logic. This centralization is one of the main design principles in your architecture doc. 

### Inputs

* typed request from Valuation, Risk, or Strategy
* movie id
* target territory
* retrieval intent

### Outputs

* unified context object
* document evidence
* DB evidence
* citations per item
* sufficiency score

---

## C. `ValuationAgent`

This is the **financial reasoning agent**.

### Job

* estimate MG for a target territory
* project theatrical revenue
* project VOD / streaming revenue
* produce a confidence interval
* list comparable films used

The architecture doc defines it as the quantitative reasoning agent and says it uses box office history, Q-scores, comparable films, exchange rates, and script themes from document retrieval. 

### Why it exists

The requirement explicitly calls for territory valuation by cross-referencing **script themes** with **historical regional box office data**. 

### Inputs

From `DataAgent`:

* box office history by genre and territory
* actor Q-scores and social reach
* comparable films and MGs
* exchange rates
* script themes
* optionally known release history for the movie

### Outputs

* MG estimate
* confidence interval
* theatrical projection
* VOD projection
* comparable film list
* sufficiency score

---

## D. `RiskAgent`

This is the **qualitative risk reasoning agent**.

### Job

* identify censorship risks
* identify cultural sensitivity risks
* identify market-fit risks
* return typed `RiskFlag[]`

The architecture doc already defines it as the agent that produces `RiskFlag[]` with category, severity, scene, page, mitigation, and confidence. 

### Why it exists

The requirement explicitly asks to detect plot points or imagery in the script that may trigger censorship or heavy editing using regional regulatory PDFs and cultural sensitivity reports. 

### Inputs

From `DataAgent`:

* script scenes with themes and page references
* censorship guidelines by country
* cultural sensitivity report pages
* theatrical window trends if needed

### Outputs

* `RiskFlag[]`
* category
* severity
* scene/page reference
* mitigation
* confidence

### Risk types for v1

Use:

* `CENSORSHIP`
* `CULTURAL_SENSITIVITY`
* `MARKET`

Those are already in your architecture doc. 

---

## E. `StrategyAgent`

This is the **prescriptive reasoning agent**.

### Job

* recommend theatrical vs streaming-first
* recommend release window
* recommend marketing spend
* rank platform priorities
* compare scenarios like “skip theatrical”

The architecture doc defines it as the agent that consumes Valuation and Risk outputs from session state, plus festival sentiment, marketing briefs, VOD benchmarks, and theatrical window trends. 

### Why it exists

The requirement explicitly asks for:

* release window and marketing spend recommendations
* comparison of talent power with narrative hooks in the script
* sentiment impact on digital vs theatrical split
* contextual follow-up analysis. 

### Inputs

* valuation output
* risk output
* festival sentiment / reviews
* marketing brief
* VOD benchmarks
* theatrical window trends
* talent signals
* narrative hooks from script/synopsis

### Outputs

* release mode recommendation
* release window
* marketing spend range
* platform priority
* ROI scenario comparison

---

# 3. Sub-agent

## `DocumentRetrievalAgent`

This is the only sub-agent in the current architecture.

### Job

* accept retrieval intent from `DataAgent`
* navigate the document indexes
* build a retrieval plan
* fetch exact pages/chunks
* check if evidence is sufficient
* expand and refetch if needed

The architecture doc describes it exactly this way and positions it under `DataAgent`. 

### Why it exists

Your requirement depends heavily on unstructured docs:

* scripts and synopses
* reviews and sentiment reports
* censorship guidelines
* cultural sensitivity reports
* marketing briefs. 

This sub-agent is what turns those into usable evidence.

### Inputs

* retrieval intent
* movie id
* document type
* territory
* themes / keywords

### Outputs

* exact page/chunk content
* source references
* sufficiency status: `PASS` or `EXPAND`

---

# 4. Tools and what each one is for

## Document tools under `DocumentRetrievalAgent`

### `IndexRegistry`

Static catalog of what exists in the indexes.
Your architecture doc says it maps structures like `Act → Scene → Page`, `Country → Rule → Page`, and `Film → Reviewer → Page`. 

### `IndexNavigator`

Builds a `RetrievalPlan`, for example which document and which pages/chunks to fetch. 

### `TargetedFetcher`

Executes the retrieval plan and fetches only the specified pages/chunks. The architecture doc explicitly says this system is **not** doing blind similarity search. 

### `SufficiencyChecker`

Checks whether the fetched content is enough. If not, expands the page range and refetches. 

---

## DB tools under `DataAgent`

### `get_box_office_by_genre_territory()`

Historical box office performance by genre and territory. Required for valuation.

### `get_actor_qscore()`

Talent power and social reach. Required for release strategy and MG estimation.

### `get_theatrical_window_trends()`

Regional release-window and timing trends. Required for Strategy and sometimes Risk.

### `get_exchange_rates()`

Currency normalization for MG calculations.

### `get_vod_price_benchmarks()`

Streaming / digital licensing benchmarks. Needed for Strategy and revenue split modeling.

### `get_comparable_films()`

Comparable movies for MG estimation. 

---

## Valuation tools under `ValuationAgent`

### `mg_calculator_tool`

Turns comparable evidence, Q-scores, and genre multipliers into an MG estimate. 

### `exchange_rate_tool`

Converts MG estimates to the target currency using rates already fetched by `DataAgent`. 

---

## Utility tools under `MarketLogicOrchestrator`

### `format_scorecard()`

Builds the final structured JSON scorecard.

### `source_citation_tool()`

Attaches every claim to its supporting document and page/chunk reference. 

---

# 5. retrieval mapping for document inventory

You said you currently have:

* synopses for 10 films
* marketing docs for the same 10 films
* cultural_sensitivity docs for the same 10 films
* reviews for the same 10 films
* censorship guidelines for 10 regions
* scripts in `.md`
* scripts in `.pdf`
* page index artifacts
* script index artifacts with pages and scenes

Given that inventory, here is how the sub-agent should use the corpus.

## For movie understanding

Use:

* `synopses`
* `scripts`
* `scripts_indexed/pages.jsonl`
* `scripts_indexed/scenes.jsonl`

Purpose:

* themes
* narrative hooks
* risky plot points
* tone
* audience shape

## For sentiment

Use:

* `reviews`

Purpose:

* awards buzz
* critic positivity / negativity
* prestige vs mainstream appeal
* likely theatrical vs streaming skew

## For censorship and cultural risk

Use:

* `cultural_sensitivity`
* `censorship_guidelines_countries`
* `scripts` / indexed scenes

Purpose:

* identify content conflicts
* local taboo/sensitivity mapping
* likely cut/edit risk

## For marketing strategy

Use:

* `marketing`
* `reviews`
* `synopses`
* `get_actor_qscore()`

Purpose:

* campaign hooks
* star power vs story hook comparison
* audience messaging
* positioning

## For general broad retrieval

Use:

* `docs/page_index/pages.jsonl`
* `docs/page_index/manifest.json`

## For script-scene analysis

Prefer:

* `docs/scripts_indexed/scenes.jsonl`
* `docs/scripts_indexed/pages.jsonl`

That setup matches the requirement’s emphasis on scripts, reviews, censorship guidance, cultural sensitivity, and marketing briefs. 

---

# 6. Full runtime flow

## Flow for a fresh question

Example:
**“How will Interstellar perform in India?”**

### Step 1 — Orchestrator

* resolve `movie = interstellar`
* resolve `territory = india`
* classify the intent:

  * valuation
  * risk
  * strategy
  * or full scorecard

### Step 2 — Orchestrator calls `DataAgent`

It asks for all evidence needed for the selected intent.

### Step 3 — `DataAgent` calls `DocumentRetrievalAgent`

It retrieves:

* Interstellar synopsis/script content
* Interstellar reviews
* Interstellar marketing brief
* Interstellar cultural sensitivity doc
* India censorship guidelines

### Step 4 — `DataAgent` calls DB tools

It fetches:

* India genre/territory box office patterns
* actor Q-scores
* India theatrical window trends
* exchange rates
* VOD benchmarks
* comparable films

### Step 5 — `DataAgent` returns unified context

This context should include:

* raw evidence
* citations
* sufficiency score

### Step 6 — `ValuationAgent` runs

It estimates:

* MG
* revenue by release mode
* confidence band

### Step 7 — `RiskAgent` runs

It flags:

* censorship
* cultural sensitivity
* market risk

### Step 8 — `StrategyAgent` runs

It decides:

* theatrical vs streaming-first
* release window
* marketing spend
* platform priority
* scenario recommendation

### Step 9 — Orchestrator validates

The architecture doc says validation is inline and includes:

* `financial_sanity_check()`
* `hallucination_check()`
* `confidence_threshold_check()` 

### Step 10 — Orchestrator formats output

`format_scorecard()` returns:

* projected revenue by territory
* risk flags
* recommended acquisition price
* release timeline

Those output categories come directly from the requirement. 

---

## Flow for a follow-up question

Example:
**“If we skip theatrical in India and go straight to streaming, how does ROI change?”**

### Step 1

Orchestrator loads session state.

### Step 2

It recognizes this as a scenario update.

### Step 3

It reuses previous valuation/risk outputs where possible.

### Step 4

It asks `DataAgent` only for any missing evidence.

### Step 5

`StrategyAgent` recalculates release strategy and ROI scenario comparison.

### Step 6

Validation runs again.

### Step 7

Updated scorecard is returned.

This behavior is required because the requirement explicitly calls for contextual follow-up questions like skipping theatrical in a territory and recomputing ROI. 

---

# 7. ownership table

## Orchestrator owns

* routing
* state
* validation
* scorecard formatting

## DataAgent owns

* document retrieval requests
* DB requests
* evidence packaging
* citations
* sufficiency score

## DocumentRetrievalAgent owns

* index planning
* page/chunk fetches
* retrieval refinement

## ValuationAgent owns

* MG reasoning
* revenue estimation
* currency conversion

## RiskAgent owns

* censorship and cultural risk reasoning
* market risk reasoning

## StrategyAgent owns

* release plan
* marketing recommendation
* ROI scenario comparison

---

# 8. structure

## Agents

* `MarketLogicOrchestrator`
* `DataAgent`
* `ValuationAgent`
* `RiskAgent`
* `StrategyAgent`

## Sub-agent

* `DocumentRetrievalAgent`

## Tools

* `IndexRegistry`
* `IndexNavigator`
* `TargetedFetcher`
* `SufficiencyChecker`
* `get_box_office_by_genre_territory()`
* `get_actor_qscore()`
* `get_theatrical_window_trends()`
* `get_exchange_rates()`
* `get_vod_price_benchmarks()`
* `get_comparable_films()`
* `mg_calculator_tool`
* `exchange_rate_tool`
* `format_scorecard()`
* `source_citation_tool()`
* `financial_sanity_check()`
* `hallucination_check()`
* `confidence_threshold_check()`
---
