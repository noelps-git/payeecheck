"""
api.py — PayeeCheck Name Matching API

Hosts all 6 matcher levels behind one local FastAPI server so you can
test them live in a browser or with curl/Postman, instead of only via
the command line.

Run:
    python run.py

Then open:
    http://localhost:8000/docs        <- interactive Swagger UI
    http://localhost:8000/health      <- health check

Example request:
    POST http://localhost:8000/match/4
    {"entered": "SBI", "actual": "State Bank of India"}
"""
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal, Optional
import hmac
import os

from api_v1.storage import init_db
from api_v1.gate import router as gate_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="PayeeCheck — Name Matching Intelligence API",
    description="L1 to L6 name matching, hosted locally. "
                 "See /docs for interactive testing.",
    version="1.0",
    lifespan=lifespan,
)

# Browser origins allowed to call this API. Defaults to local development
# origins; set PAYEECHECK_ALLOWED_ORIGINS to a comma-separated list when
# deploying. "*" is honoured only if explicitly configured.
_DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("PAYEECHECK_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_API_KEY = os.environ.get("PAYEECHECK_API_KEY", "")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Authenticates scoring/matching endpoints when PAYEECHECK_API_KEY is set.
    Unset (local development) leaves the endpoints open; the sandbox and
    health endpoints are always public.
    """
    if not _API_KEY:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── Gate — visitor identification + query flow ────────────────────────
app.include_router(gate_router)

# ── Sandbox — deterministic demo environment ──────────────────────────
from sandbox.router import router as sandbox_router
app.include_router(sandbox_router)

_NAME = Field(min_length=1, max_length=256)
_OPT_NAME = Field(default=None, max_length=256)


class MatchRequest(BaseModel):
    entered: str = _NAME
    actual: str = _NAME


class AttributionRequest(BaseModel):
    transaction: dict
    payee: dict
    temporal: dict


class ResolveRequest(BaseModel):
    name: Optional[str] = _OPT_NAME
    vpa: Optional[str] = _OPT_NAME
    account: Optional[str] = _OPT_NAME
    mobile: Optional[str] = _OPT_NAME


class ScoreRequest(BaseModel):
    entered_name: str = _NAME
    actual_name: str = _NAME
    payee_vpa: str = Field(min_length=1, max_length=256)
    amount: float = Field(ge=0, le=1e12)
    input_method: Literal["type", "paste"] = "type"
    vpa_age_days: int = Field(default=365, ge=0, le=100_000)
    prior_tx_count: int = Field(default=10, ge=0, le=10_000_000)
    unique_senders_7d: int = Field(default=1, ge=0, le=10_000_000)
    mule_flagged: bool = False
    mule_flag_count: int = Field(default=0, ge=0, le=1_000)


# ── Lazy-load matchers so the server starts instantly and only loads ─
# ── the heavy ML models (L4/L5) the first time they're actually hit ──
_matchers_cache = {}


def get_matcher(level: int):
    if level in _matchers_cache:
        return _matchers_cache[level]

    if level == 1:
        from matchers.l1_fuzzy import match
    elif level == 2:
        from matchers.l2_phonetic import match
    elif level == 3:
        from matchers.l3_tfidf import match
    elif level == 4:
        from matchers.l4_embeddings import match
    elif level == 5:
        from matchers.l5_siamese import match
    else:
        raise HTTPException(status_code=404, detail=f"No matcher for level {level}")

    _matchers_cache[level] = match
    return match


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the PayeeCheck B2B dashboard."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="UI file not found. Expected static/index.html in the "
                   "payeecheck/ folder."
        )
    return FileResponse(index_path, media_type="text/html")


@app.get("/sandbox", include_in_schema=False)
def serve_sandbox():
    """Serves the gated fraud intelligence sandbox UI."""
    sandbox_path = os.path.join(_STATIC_DIR, "sandbox.html")
    if not os.path.exists(sandbox_path):
        raise HTTPException(
            status_code=404,
            detail="sandbox.html not found in static/."
        )
    return FileResponse(sandbox_path, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "payeecheck-name-matching",
             "levels_available": [1, 2, 3, 4, 5, 6]}


@app.post("/match/{level}", dependencies=[Depends(require_api_key)])
def match_at_level(level: int, req: MatchRequest):
    """
    Run name matching at a specific level (1-5).
    Levels 4 and 5 load ML models on first call — expect a few seconds
    of one-time latency the first time you hit them.
    """
    if level not in (1, 2, 3, 4, 5):
        raise HTTPException(
            status_code=400,
            detail="Level must be 1-5. Use /resolve for Level 6 entity resolution."
        )
    matcher = get_matcher(level)
    return matcher(req.entered, req.actual)


@app.post("/compare", dependencies=[Depends(require_api_key)])
def compare_all_levels(req: MatchRequest):
    """
    Run the same name pair through every level (1-5) and return all
    results together — useful for seeing accuracy improve level by level
    on a single example.
    """
    results = {}
    for level in (1, 2, 3, 4, 5):
        matcher = get_matcher(level)
        results[f"level_{level}"] = matcher(req.entered, req.actual)
    return {"entered": req.entered, "actual": req.actual, "results": results}


@app.post("/resolve", dependencies=[Depends(require_api_key)])
def resolve_entity(req: ResolveRequest):
    """
    Level 6 — graph-augmented entity resolution.
    Pass any combination of name / vpa / account / mobile; the resolver
    finds known entities sharing any of those attributes.
    """
    from matchers.l6_graph import resolve
    query = {"name": req.name, "vpa": req.vpa,
             "account": req.account, "mobile": req.mobile}
    return resolve(query)


@app.get("/rings", dependencies=[Depends(require_api_key)])
def get_fraud_rings(min_size: int = Query(default=2, ge=2, le=1000)):
    """Level 6 — scan the seeded entity graph for fraud rings."""
    from matchers.l6_graph import find_rings
    rings = find_rings(min_ring_size=min_size)
    return {"ring_count": len(rings), "rings": rings}


@app.post("/attribution", dependencies=[Depends(require_api_key)])
def get_attribution(req: AttributionRequest):
    """
    Three-layer attribution record. Pass transaction / payee / temporal
    signal dicts — see attribution.py docstring for the expected shape.
    """
    from attribution import build_attribution
    return build_attribution(req.transaction, req.payee, req.temporal)


@app.post("/score", dependencies=[Depends(require_api_key)])
def score_transaction_endpoint(req: ScoreRequest):
    """
    Phase 2 — the real Risk Scorer. This is what the PayeeCheck prototype
    UI calls live: every field the phone screen captures (name, VPA,
    amount, paste-vs-type) is sent here, runs through all 6 signal
    modules (name match, look-alike, mule, velocity, sanctions, plus the
    attribution record), and returns one verdict.

    First call loads the TF-IDF name matcher (fast, no network needed).
    """
    from risk_engine.risk_scorer import score_transaction
    tx = {
        "tx_id": "live",
        "entered_name": req.entered_name,
        "actual_name": req.actual_name,
        "payee_vpa": req.payee_vpa,
        "amount": req.amount,
        "input_method": req.input_method,
        "vpa_age_days": req.vpa_age_days,
        "prior_tx_count": req.prior_tx_count,
        "unique_senders_7d": req.unique_senders_7d,
        "mule_flagged": req.mule_flagged,
        "mule_flag_count": req.mule_flag_count,
    }
    return score_transaction(tx)
