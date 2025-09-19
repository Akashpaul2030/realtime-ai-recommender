# Repository Guidelines

## Project Structure & Module Organization
- `api/` hosts the FastAPI application (`app.py`), route handlers in `routes/`, and optional middleware.
- `services/` contains orchestration logic for ingestion, ranking, and vector updates invoked by the API.
- `adapters/` encapsulates Redis, Supabase, Pinecone, and other backend integrations behind clean interfaces.
- `models/` stores embedding utilities, similarity scoring helpers, and related ML components.
- `utils/` offers shared logging and metrics helpers; keep dependencies minimal to avoid import cycles.
- `tests/` mirrors the runtime packages with pytest suites; add fixtures near their consumers for clarity.
- `demos/`, `notebooks/`, and Streamlit entry points provide interactive flows; refresh them whenever API schemas evolve.
- Runtime configuration defaults live in `config.py`; override via `.env` files derived from `.env.example`.

## Build, Test & Run Commands
- `python -m venv venv` followed by `pip install -r requirements.txt` bootstraps the Python environment.
- `uvicorn api.app:app --reload` (or `python -m api.app`) starts the REST API using the current environment.
- `python demo_setup.py` seeds sample data and exercises the end-to-end recommendation loop.
- `python streamlit_app.py` launches the Streamlit dashboard at `http://localhost:8501`.
- `python simple_api_test.py` runs smoke checks; `python -m pytest` executes the full automated suite.
- `python playwright_api_test.py` performs browser-based verification once the API and Streamlit apps are active.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and annotate public interfaces with type hints.
- Order imports by standard library, third-party, then local modules; avoid relative paths that cross package boundaries.
- Keep FastAPI route modules thin; delegate business logic to `services/` and persistence to `adapters/`.
- Update docstrings and comments when contracts or payloads change, and reflect cross-cutting impacts in `INTEGRATION_GUIDE.md`.

## Testing Guidelines
- Mirror package structure with files such as `tests/test_api.py`; share fixtures locally or via a `tests/conftest.py` when reuse grows.
- Use pytest markers to flag slow or external-resource tests so the default run stays fast and deterministic.
- Run `pytest --cov=api --cov=services --cov=adapters --cov-report=term-missing` before submitting and mention meaningful coverage shifts in PR notes.
- Capture unhappy-path scenarios for new endpoints and adapters, especially when toggling backend flags in `config.py`.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) as seen in history; add scopes when they clarify impact (e.g., `feat(api):`).
- Name branches after the objective (`feature/hybrid-ranking`, `bugfix/redis-auth`) to aid tracking.
- PR checklist: concise summary, linked issue, test commands with results, and screenshots or recordings for UI or demo changes.
- Call out configuration or migration steps, particularly `.env` keys or `config.py` defaults, and request reviews from the relevant module owners.

## Configuration & Security Notes
- Copy `.env.example` to `.env` for local work; never commit actual credentials. Update the example file whenever new required variables appear.
- Adjust backend toggles (`BACKEND_TYPE`, `VECTOR_STORE_TYPE`, `EVENT_PROCESSOR_TYPE`, etc.) deliberately and document default changes in the integration guide.
- Store production secrets in the deployment platform's secret manager and rotate keys referenced by demo scripts on a regular cadence.
