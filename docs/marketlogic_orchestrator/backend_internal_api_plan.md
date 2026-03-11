# Backend API Plan For Agent Tool Data Access

## Summary
Your approach is good for MVP: keep ADK as orchestration/runtime and make backend the single data-access layer.
Use one shared service key now (`X-Internal-API-Key`) between backend and adk-server, then harden later with key rotation and optional signed requests.

## API Design (Tool-Aligned)
- `GET /internal/v1/market/box-office`
  - Query: `movie`, `territory`
  - Returns: `avg_gross_usd`, `total_gross_usd`, `samples`
- `GET /internal/v1/market/actor-signals`
  - Query: `movie`
  - Returns: `avg_qscore`, `total_social_reach`
- `GET /internal/v1/market/theatrical-windows`
  - Query: `territory`
  - Returns: list of `{window_type, days}`
- `GET /internal/v1/market/exchange-rate`
  - Query: `territory`
  - Returns: `{currency_code, rate_to_usd, rate_date}`
- `GET /internal/v1/market/vod-benchmarks`
  - Query: `territory`
  - Returns: `{avg_price_min_usd, avg_price_max_usd}`
- `GET /internal/v1/market/comparables`
  - Query: `movie`, `territory`, `limit`
  - Returns: comparable film list with territory gross
- `POST /internal/v1/docs/search`
  - Body: `movie`, `territory`, `intent`, `max_docs`, `max_scenes`
  - Returns: matched document chunks + scene chunks + citations
- `POST /internal/v1/evidence/bundle`
  - Body: `movie`, `territory`, `intent`, `needs_db`, `needs_docs`
  - Returns: single bundled payload for all tool needs (recommended to reduce N+1 calls)
- `GET /internal/v1/meta/registry`
  - Returns: canonical movies, territories, and available doc/index coverage

## Security & Communication
- Keep existing user auth (`Bearer`) for frontend -> backend.
- Add service auth for adk-server -> backend on `/internal/v1/*`:
  - Header: `X-Internal-API-Key`
  - Validate with constant-time compare against shared secret in backend env.
- Keep current backend -> adk-server key (`X-ADK-API-Key`) for agent run endpoint.
- Add minimum hardening now:
  - key rotation support (`current` + `next` keys)
  - request ID logging
  - timeout + retry policy for service-to-service calls
  - strict CORS bypass for internal endpoints (no browser exposure)

## Implementation Changes
- Backend:
  - Add new internal router group `/internal/v1/*` with service-key dependency.
  - Move/centralize query logic into backend services (reuse existing SQL logic shape from ADK tools).
  - Add Pydantic response models matching tool contracts.
- ADK server:
  - Refactor tools to call backend internal endpoints instead of direct DB/docs access.
  - Keep tool function signatures stable so agent behavior does not change.
- Config:
  - Reuse one shared key variable or add dedicated `INTERNAL_API_KEY` (preferred long-term).

## Test Plan
- Unit tests for each internal endpoint response contract.
- Integration tests for auth failures:
  - missing key -> `401`
  - wrong key -> `401`
- Integration tests for happy path:
  - endpoints return non-empty seeded data for known movie/country.
- ADK tool integration:
  - mocked backend responses
  - timeout/retry behavior
- End-to-end:
  - chat flow still works and tool results map to seeded DB/docs correctly.

## Assumptions
- ADK should not read DB/docs directly in the target design.
- Internal endpoints are only service-to-service, not exposed to frontend.
- Shared API key is acceptable for current stage; stronger auth can be phased in later.
