# MarketLogic AI Agent Guide

This file defines the working conventions for agents in this repository.

## Project Overview

MarketReel is a MarketLogic AI system for film distribution executives. It evaluates independent films for global acquisition and theatrical release strategy using:

- Structured data from PostgreSQL (box office, talent signals, window trends, licensing benchmarks, FX).
- Unstructured documents (scripts, synopses, reviews, censorship and cultural guidance, marketing briefs).
- Conversational analysis that returns strategic recommendations and structured scorecard outputs.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, PostgreSQL 16 |
| ADK Server | FastAPI, Google ADK, Google GenAI |
| Frontend | React 18, TypeScript, Vite 5, Material-UI v6 |
| Package Managers | uv (Python), npm (Node) |
| Quality | Ruff, MyPy (strict), import-linter, ESLint, TypeScript strict |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions |

## Folder And File Structure

```text
.
├── README.md                   # Setup, runbook, API notes
├── AGENTS.md                   # Agent runtime/document conventions
├── docs/                       # Product/implementation docs
├── server/                     # Backend API service
│   ├── app/
│   │   ├── api/routes/         # Backend routes
│   │   ├── auth/               # JWT auth
│   │   ├── db/                 # SQLAlchemy models + sessions
│   │   ├── services/           # Business logic + ADK orchestration
│   │   └── core/               # Config + logging
│   ├── alembic/                # DB migrations
│   └── tests/                  # Backend tests
├── adk-server/                 # ADK runtime service
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint for /v1/run
│   │   └── core/               # ADK server config
│   ├── docs/                   # Shared documents for all agents
│   ├── agents/
│   │   ├── marketlogic/        # Runtime MarketLogic agent
│   │   │   ├── agent.py        # root_agent and run_agent
│   │   │   └── docs/           # Docs scoped only to MarketLogic agent
│   │   ├── eval/               # Agent eval harness/data
│   │   └── tests/              # Agent-level tests
│   └── tests/                  # ADK server API tests
└── client/                     # React frontend
    └── src/                    # UI pages, state, API client
```

## Runtime Source Of Truth

- ADK runtime entrypoint is `adk-server/app/main.py`.
- Runtime agent module is `adk-server/agents/marketlogic/agent.py`.
- Do not reintroduce `adk-server/app/agent.py`.

## Google ADK Python Best Practices

These practices are aligned with the local `google-adk-python` skill references (`agents`, `models`, `tools`, `runtime-arch`, `callbacks`, `deploy-ops`).

- Keep one clear `root_agent` per runtime entrypoint and keep orchestration logic in one module.
- Keep model selection configurable via environment (`ADK_MODEL`), not hardcoded in agent code.
- Use explicit, narrowly scoped tool functions with typed signatures and deterministic returns.
- Keep tools side-effect free by default; isolate side-effectful operations behind service boundaries.
- Use `DatabaseSessionService` for persistent sessions in non-trivial apps; avoid in-memory sessions for runtime services.
- Reuse a lazily initialized `Runner` and session service per process instead of recreating per request.
- Validate service auth (`X-ADK-API-Key`) before invoking `run_agent` so unauthorized calls never reach model execution.
- Store business chat/message history in backend; use ADK session state for agent runtime context/state only.
- Add callbacks for logging/guardrails/metrics at agent, model, and tool boundaries as complexity grows.
- Favor workflow agents (sequential/parallel/loop) when tasks are multi-step, instead of overloading one prompt.
- Keep shared knowledge in `adk-server/docs/` and agent-specific knowledge in `adk-server/agents/<agent>/docs/`.
- Add eval coverage for critical behaviors and regression-prone prompts in `adk-server/agents/eval/`.
- Keep deployment/runtime concerns explicit: health endpoint, structured logs, and reproducible env-driven config.
- Treat safety checks and policy enforcement as first-class runtime concerns before/around model/tool calls.

## ADK Logging Standards

- Log at ADK API boundary (`adk-server/app/main.py`) for:
  - run start (`user_id`, `session_id`, message length)
  - run success (`session_id`, reply length)
  - run failure (`exception` with stack trace)
  - auth failures (`X-ADK-API-Key` invalid/missing)
- Log at agent runtime (`adk-server/agents/marketlogic/agent.py`) for:
  - runner/session-service initialization
  - session create vs session reuse
  - model-disabled fallback path (no provider key)
  - run completion (`session_id`, reply length)
- Never log secrets, raw API keys, or full prompt content by default.

## ADK Evals Standards

- Keep agent eval assets in `adk-server/agents/eval/`.
- Keep reusable eval fixtures under `adk-server/agents/eval/data/`.
- Maintain regression checks for:
  - instruction/prompt intent coverage
  - tool output schema/contract
  - fallback behavior when provider credentials are absent
- Run evals with:

```bash
cd adk-server
uv run pytest agents/eval/test_eval.py -q
```
- Run API and eval suites together before merging ADK changes:

```bash
cd adk-server
uv run pytest tests/test_run.py agents/eval/test_eval.py -q
```

## Document Structure

- Shared/common docs for all agents go in `adk-server/docs/`.
- Agent-specific docs go in `adk-server/agents/<agent_name>/docs/`.
- Current runtime agent-specific docs path:
  - `adk-server/agents/marketlogic/docs/`

## Service Boundaries

- Frontend: UI only (`client/`).
- Backend: owns auth, chats, messages, and orchestration (`server/`).
- ADK server: runs LLM agent logic only (`adk-server/`).
- Backend calls ADK server over HTTP (`POST /v1/run`).

## Auth Rules

- User JWT is validated in backend only.
- Backend authenticates to ADK server with `X-ADK-API-Key`.
- ADK server must reject missing/invalid API key before invoking the agent runtime.
- API keys are service-to-service secrets and must not be sent to LLM prompts/tools.

## Persistence Rules

- Backend owns business chat/message records (`chats`, `messages`).
- ADK session state persists through ADK `DatabaseSessionService` in PostgreSQL.
- Preferred local DB name: `marketreeldb`.

## Environment

Single root `.env` is used by all services. Important keys:

- `DATABASE_URL`
- `SECRET_KEY`
- `ADK_BASE_URL`
- `ADK_API_KEY`
- `GOOGLE_API_KEY`
- `ADK_MODEL`
- `VITE_API_BASE_URL`

## Local Run Commands

Backend:
```bash
cd server
uv sync --all-extras
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8010
```

ADK server:
```bash
cd adk-server
uv sync --all-extras
uv run uvicorn app.main:app --reload --port 8011
```

Frontend:
```bash
cd client
npm install
npm run dev
```
