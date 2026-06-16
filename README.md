# Health Insurance Claims Processing System

Automated claim adjudication: document verification, vision extraction,
deterministic policy rules, explainable decisions with a complete audit
trace, and graceful degradation under component failure.

Built for the Plum AI Automation Engineer assignment.

## Result

All 12 official test cases pass deterministically (no LLM involved in any
decision): `backend/tests/test_official_cases.py`. The generated eval
report with full decision output and trace per case is at
`docs/eval_report.md`.

## Architecture in one paragraph

An orchestrator coordinates seven single-responsibility agents (intake,
extraction, category resolution, document verification, derivation,
consistency, adjudication) over typed Pydantic contracts. Every policy
decision and every rupee of math is deterministic code reading
`policy_terms.json` at runtime; the LLM (Claude vision) does exactly one
job: turning messy document images into structured fields behind an
injectable, mockable interface. The documents are the source of truth:
extraction runs before the gate; the claim category (when not provided),
amount, hospital, and treatment date are derived from the documents on the
real-upload path — genuinely ambiguous documents ask the member to pick the
category rather than guessing or queueing a reviewer; and the consistency
agent cross-checks the patient against the roster and the filed category
against the document evidence, routing mismatches to manual review rather
than auto-rejecting. Document problems
stop the claim early with specific actionable messages. Every check appends
to a structured trace, component failures degrade the pipeline (lower
confidence, manual-review recommendation) instead of crashing it, and
fraud signals route payable claims to manual review. Full detail:
`docs/architecture.md`, `docs/component_contracts.md`,
`docs/assumptions.md`.

## Run it

Backend (Python 3.10+; tested on 3.10.11):

    cd backend
    # recommended: isolate dependencies in a virtual environment
    python -m venv .venv
    source .venv/bin/activate          # macOS/Linux
    # .venv\Scripts\Activate.ps1       # Windows PowerShell
    pip install -r requirements.lock      # exact tested versions (reproducible)
    # or: pip install -r requirements.txt  # bounded ranges (newest within major)

**Create your API-key file.** `backend/.env` is gitignored, so a fresh clone
does not include one — you must create it yourself. From `backend/`, copy the
template and fill in your own Anthropic key (get one at
https://console.anthropic.com/):

    cp .env.example .env       # then edit .env and set your key:
    #   ANTHROPIC_API_KEY=sk-ant-...

This key powers live vision extraction of uploaded document images. It is
optional for the deterministic path — the 12 official test cases, the eval
report, and the web UI all run without it (the server logs
`live_extractor=False`) — but extracting a real uploaded document needs it.

Start the server (the `--env-file` flag loads the `backend/.env` you created):

    python -m uvicorn app.api:app --port 8000 --env-file .env

Then open http://localhost:8000. The built frontend is committed under
`backend/static/` and served by the backend, so you do not need to build the
frontend just to run the app.

Prefer an environment variable to a file? Export the key and drop `--env-file`:

    export ANTHROPIC_API_KEY=...            # macOS/Linux
    # $env:ANTHROPIC_API_KEY = "..."        # Windows PowerShell
    python -m uvicorn app.api:app --port 8000

Tests and eval report:

    cd backend
    python -m pytest tests/ -v
    python -m scripts.generate_eval_report    # writes docs/eval_report.md

Frontend build (optional — the built UI is already committed under
`backend/static/`, so this is NOT needed to run the app; rebuild only if you
change `frontend/`, requires Node.js 18+):

    cd frontend
    npm install
    npm run build          # output is copied to backend/static for deploy
    cp -r dist/* ../backend/static/                        # macOS/Linux
    # Copy-Item -Recurse -Force dist/* ../backend/static/  # Windows PowerShell

## Layout

    backend/app/models/        Pydantic contracts (claim, decision, policy)
    backend/app/agents/        intake, extraction, category_resolution,
                               document_verification, derivation,
                               consistency, adjudication
    backend/app/orchestrator.py
    backend/app/condition_mapping.py   deterministic diagnosis -> policy mapping
    backend/app/confidence.py          deterministic confidence formula
    backend/app/llm/extractor.py       Claude vision extractor (the only LLM component)
    backend/app/storage.py             SQLite persistence + member history
    backend/app/api.py                 FastAPI layer, serves the frontend
    backend/tests/                     12 official cases + component + API tests
    backend/scripts/generate_eval_report.py
    frontend/                          React (Vite) submit + review UI
    docs/                              architecture, contracts, assumptions, eval report
