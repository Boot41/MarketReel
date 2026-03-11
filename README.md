# MarketLogic AI Scaffold

Full-stack scaffold with FastAPI + React + PostgreSQL + Google ADK. Three services (backend, frontend, ADK server) with auth, chat history, and tooling pre-configured.

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

## Project Structure

```
.
├── .github/workflows/ci.yml    # CI pipeline
├── docker-compose.yml           # PostgreSQL + API + Client + ADK server
├── Dockerfile                   # Multi-stage production build
├── server/                      # FastAPI backend
│   ├── app/
│   │   ├── agent/               # Backend API (no LLMs here)
│   │   ├── api/routes/          # HTTP endpoints
│   │   ├── auth/                # JWT auth + bcrypt
│   │   ├── core/                # Config + logging
│   │   ├── db/                  # SQLAlchemy models + session
│   │   ├── middleware/          # Error handler + request logging
│   │   ├── services/            # Business logic layer
│   │   └── main.py              # App factory
│   ├── alembic/                 # Database migrations
│   ├── tests/                   # Unit + integration tests
│   └── pyproject.toml           # Dependencies + tool config
├── adk-server/                  # Google ADK service
│   ├── app/
│   │   ├── core/                # Config
│   │   └── main.py              # ADK FastAPI app
│   ├── docs/                    # Shared/common documents for all agents
│   ├── agents/                  # Agent code + agent-scoped assets
│   │   ├── marketlogic/          # Runtime MarketLogic agent
│   │   │   ├── docs/             # Docs used only by the MarketLogic agent
│   │   ├── eval/                 # Evaluation harness
│   │   └── tests/                # Agent-specific tests
│   ├── tests/                   # ADK server tests
│   └── pyproject.toml
└── client/                      # React frontend
    ├── src/
    │   ├── App.tsx              # Router + pages
    │   ├── main.tsx             # Entry point + providers
    │   └── lib/api.ts           # Axios instance
    ├── package.json             # Dependencies + scripts
    └── Dockerfile               # Node build → Nginx
```

## Module Boundaries

- `app.api` depends on `app.services`, `app.auth`, `app.agent`, `app.core` only
- `app.services` depends on `app.db`, `app.core` only (must NOT import from `app.api`)
- `app.db` depends on `app.core` only (must NOT import from `app.services` or `app.api`)
- `app.agent` depends on `app.core` only (must NOT import from `app.api` or `app.services`)
- Enforced by `import-linter` in CI

## Quick Start (Local)

```bash
# Start PostgreSQL
docker-compose up -d db

# Backend (terminal 1)
cd server
uv sync --all-extras
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8010

# ADK server (terminal 2)
cd adk-server
uv sync --all-extras
uv run uvicorn app.main:app --reload --port 8011

# Frontend (terminal 3)
cd client
npm install
npm run dev
```

App runs at http://localhost:5173 with API proxied to :8010.
ADK server runs at http://localhost:8011 and is called by the backend.

## Seeding MarketLogic Data

From `server/`, run:

```bash
uv run python scripts/seed_marketlogic.py
```

This loads MarketLogic datasets from `seed.txt` (repo root), prints inserted counts,
and performs a full reset of MarketLogic domain tables before inserting fresh data.

## Environment Setup

1. Copy `.env.example` to `.env` in the repo root.
2. Set the required secrets:
   - `SECRET_KEY` for JWT signing
   - `GOOGLE_API_KEY` for Google GenAI access (ADK server)
   - `ADK_API_KEY` shared between backend and ADK server
   - `INTERNAL_API_KEY` shared between adk-server and backend `/internal/v1/*`

Example:
```
ENV=development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/marketreeldb
SECRET_KEY=app-scaffold-dev-secret
GOOGLE_API_KEY=your-google-key
GOOGLE_GENAI_USE_VERTEXAI=false
ADK_BASE_URL=http://localhost:8011
ADK_API_KEY=your-shared-service-key
BACKEND_BASE_URL=http://localhost:8010
INTERNAL_API_KEY=your-internal-shared-key
INTERNAL_API_NEXT_KEY=
APP_NAME=marketlogic_adk
ADK_MODEL=gemini-2.5-flash
VITE_API_BASE_URL=http://localhost:8010
```

## Quality Gates

Backend (from `server/`):
```bash
uv run ruff format --check .    # Formatting
uv run ruff check .             # Linting
uv run mypy .                   # Type checking (strict)
uv run lint-imports             # Module boundaries
uv run pytest tests/unit -q     # Unit tests
uv run pytest tests/integration -q  # Integration tests
```

Frontend (from `client/`):
```bash
npm run lint    # ESLint
npm run build   # TypeScript + Vite build
```

## API Endpoints (Backend)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/auth/login` | No | Returns JWT token |
| POST | `/api/v1/agent/run` | Bearer | Run the ADK agent via backend |
| * | `/api/v1/*` | Bearer | Protected routes (add yours here) |
| * | `/internal/v1/*` | `X-Internal-API-Key` | Internal-only service endpoints for ADK tools |

## API Endpoints (ADK Server)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/v1/run` | `X-ADK-API-Key` | Run the ADK agent |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | development / test / production |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5433/marketreeldb` | Database connection |
| `SECRET_KEY` | `app-scaffold-dev-secret` | JWT signing secret |
| `GOOGLE_API_KEY` | *(empty)* | Google AI Studio API key for ADK |
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` | Set `true` to use Vertex AI instead |
| `ADK_API_KEY` | *(empty)* | Service key for backend → ADK server calls |
| `BACKEND_BASE_URL` | `http://localhost:8010` | adk-server target backend URL for internal tools |
| `INTERNAL_API_KEY` | *(empty)* | Service key for adk-server → backend internal calls |
| `INTERNAL_API_NEXT_KEY` | *(empty)* | Optional key rotation slot accepted by backend |

## Docker

```bash
# Development (3 services)
docker-compose up

# Production (single image)
docker build -t app-scaffold .
```

## Google ADK Agent

A sample agent is provided at `adk-server/agents/marketlogic/agent.py`. It uses the `google-adk` SDK with Gemini models.

Document convention:
- `adk-server/docs/` contains shared documents reusable by multiple agents.
- `adk-server/agents/<agent_name>/docs/` contains documents scoped to that specific agent only.

To customize:
1. Edit `adk-server/agents/marketlogic/agent.py` — define tools and the `root_agent`
2. Set `GOOGLE_API_KEY` in the root `.env`
3. Run via service: `cd adk-server && uv run uvicorn app.main:app --reload --port 8011`
4. Or call via backend: `POST /api/v1/agent/run` with `{"message": "Hello"}`

## Coding Style

- Python: 4-space indent, type hints, `snake_case` functions, `PascalCase` classes
- TypeScript: strict mode, `PascalCase` components, `camelCase` utilities
- Backend layering: Routes → Services → DB (enforced)
- Conventional Commits: `feat:`, `fix:`, `refactor:`
