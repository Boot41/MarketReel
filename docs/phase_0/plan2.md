# Add ADK Service API Key Auth

## Summary
Implement a simple service-to-service API key between `server` and `adk-server` using `X-ADK-API-Key`, with env-configured shared secret. Backend remains the only user-auth boundary.

## Key Changes
- Config
  - Add `ADK_API_KEY` to root `.env.example` (and any service docs that list env vars).
  - Add `adk_api_key` setting in `server/app/core/config.py` and `adk-server/app/core/config.py`.
- ADK server auth
  - Add FastAPI dependency that reads `X-ADK-API-Key` and rejects missing/invalid keys with `401 Unauthorized`.
  - Apply dependency to `/v1/run` (and any future ADK routes).
  - Use `secrets.compare_digest` for constant-time comparison.
- Backend client
  - Update `server/app/services/adk_client.py` to send `X-ADK-API-Key` header from settings.
  - Keep request payload unchanged (`message`, `user_id`, `session_id`).
- Docs
  - Update `README.md` (or relevant setup section) with the new env var and how to run locally.

## Tests
- ADK server
  - `/v1/run` without header returns `401`.
  - `/v1/run` with correct header returns `200` (monkeypatch `run_agent` as in current test).
- Backend
  - If any HTTP client tests exist, add one to assert header is set (or add a small unit test for `adk_client.run_adk` using `httpx.MockTransport`).

## Assumptions
- One shared API key is sufficient for now (no key rotation or multi-tenant keys).
- ADK server is not publicly exposed; only backend calls it.
- We keep using `X-ADK-API-Key` as the header name.
