# Project State — BUSFLOW

## Status: Phase 1 — Repository & Architecture Initialization
- **Backend**: `backend/` (Python 3.14, FastAPI, SQLAlchemy 2.x, Alembic, `uv`, `ruff` full, `pyrefly` full, `pytest` 100% coverage)
- **Frontend**: `frontend/` (React, TypeScript strict, Vite, Vitest 100% coverage, ESLint, Prettier, Telegram Mini Apps SDK)
- **Infra**: `docker-compose.yml`, `.env.example`, GitHub Actions CI (`.github/workflows/ci.yml`), docs (`README.md`)

## Completed Milestones
- [ ] Phase 1: Repository, architecture, tooling, Docker Compose, CI
- [ ] Phase 2: Database models, Alembic migrations, repositories, services, transactional booking logic
- [ ] Phase 3: Telegram integration (`initData` HMAC validation, session JWT/auth, bot client & outbox worker)
- [ ] Phase 4: Passenger Mini App UI & tests
- [ ] Phase 5: Dispatcher, Admin, and Driver interfaces & tests
- [ ] Phase 6: E2E, concurrency, security, 100% coverage verification
