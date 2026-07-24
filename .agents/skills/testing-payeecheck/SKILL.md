---
name: testing-payeecheck
description: How to run and end-to-end test the PayeeCheck FastAPI UPI fraud-detection demo locally (server, dashboard, sandbox scenario playground, /score, unit suite).
---

# Testing PayeeCheck locally

## Environment

- A venv at `./venv` usually already exists with fastapi, uvicorn, rapidfuzz, metaphone,
  scikit-learn, pydantic, pandas, numpy, networkx, pytest, pytest-cov, httpx.
- `torch` / `torch-geometric` / `sentence-transformers` are deliberately NOT installed
  (~1 GB). Do not install them just to test.
- Consequence: `POST /match/4`, `POST /match/5` and `POST /compare` return HTTP 500 with
  `ModuleNotFoundError: No module named 'sentence_transformers'`, and `/score` reports
  `"gat": {"is_available": false, "note": "GAT import failed"}`. These are expected in an
  offline environment — report them as untested, not failed.

## Running

```bash
cd <repo> && ./venv/bin/python run.py          # http://localhost:8000, reload=True
# dashboard: GET /      swagger: /docs
```

The marketing/sandbox page in `docs/index.html` is NOT served by FastAPI. To exercise it:

```bash
./venv/bin/python -m http.server 8001 -d docs  # then http://localhost:8001/#sandbox
```

## What is real vs hardcoded (important for honest test claims)

- `static/index.html` (served at `/`) is an analyst alert-queue dashboard built from a
  hardcoded `ALERTS` JS array. Its only live network call is `fetch('/health')` every 12 s,
  which drives the "API online" / "API offline" indicator in the sidebar. Killing the server
  and waiting ~12 s is a good way to prove that indicator is real.
- `docs/index.html`'s 13-scenario "7-signal pipeline" playground is also hardcoded client-side
  (object `S`) and does not call `POST /sandbox/v1/payee-checks`. Verify UI↔API parity by
  comparing the rendered verdict/risk_score with a real API call for the same `scenario_id`.

## Useful endpoint checks (no auth needed)

```bash
curl -s localhost:8000/health
curl -s localhost:8000/sandbox/v1/scenarios          # expect 13 scenarios
curl -s -X POST localhost:8000/sandbox/v1/payee-checks -H 'content-type: application/json' \
     -d '{"scenario_id":"mule_consortium"}'          # deterministic block / 95
# unknown scenario_id falls back to clean (pass / 3)
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d '{...}'
curl -s localhost:8000/rings                          # ring_count 2 on seeded graph
```

`/score` verdict thresholds are weighted: a paste + brand-impersonating VPA + 4-day VPA
payload alone scores ~34 → `pass`. Add `"mule_flagged": true, "mule_flag_count": 2` to get
`block` (~90). The README's documented `/score` example may therefore not reproduce the
`block` / 91 response shown in the README's attribution section — check before asserting.

## Unit suite

```bash
./venv/bin/python -m pytest --cov=. --cov-report=term-missing   # 192 passed, ~66% total
```

`pytest.ini` restricts collection to `tests/unit`; legacy scripts in `tests/` (test_l4.py,
test_l5.py, benchmark*.py) are not pytest tests and need the ML deps.

## Devin Secrets Needed

None — the app, sandbox and `/score` routes are unauthenticated.
