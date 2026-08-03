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

Security (env):
    ALLOWED_ORIGINS     comma-separated CORS origins (default: localhost)
    PAYEECHECK_API_KEY  if set, require X-API-Key / Bearer on API routes
    DISABLE_DOCS=1      hide /docs and /redoc
    RATE_LIMIT_PER_MIN  requests per IP per minute (default: 60)
"""
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
import os
import time

_DISABLE_DOCS = os.environ.get("DISABLE_DOCS", "").strip() in ("1", "true", "True", "yes")
_API_KEY = os.environ.get("PAYEECHECK_API_KEY", "").strip()
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]

_PUBLIC_PATHS = {"/", "/sandbox", "/health", "/og-image.png", "/docs", "/redoc", "/openapi.json"}
_PROTECTED_PREFIXES = ("/match", "/compare", "/resolve", "/rings", "/attribution")

app = FastAPI(
    title="PayeeCheck — Name Matching Intelligence API",
    description="L1 to L6 name matching, hosted locally. "
                 "See /docs for interactive testing.",
    version="1.0",
    docs_url=None if _DISABLE_DOCS else "/docs",
    redoc_url=None if _DISABLE_DOCS else "/redoc",
    openapi_url=None if _DISABLE_DOCS else "/openapi.json",
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-process per-IP sliding window (~RATE_LIMIT requests/min)."""

    def __init__(self, app, limit_per_min: int = 60):
        super().__init__(app)
        self.limit = max(1, limit_per_min)
        self.window = 60.0
        self._hits: dict[str, deque] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS or not any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()
        q = self._hits[ip]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
                headers={"Retry-After": "60"},
            )
        q.append(now)
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """When PAYEECHECK_API_KEY is set, require it on protected API routes."""

    async def dispatch(self, request: Request, call_next):
        if not _API_KEY:
            return await call_next(request)

        path = request.url.path
        if path in _PUBLIC_PATHS or not any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get("x-api-key", "").strip()
        if not provided:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                provided = auth[7:].strip()

        if provided != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)


# Middleware runs in reverse add order: CORS outermost, then API key, then rate limit.
app.add_middleware(RateLimitMiddleware, limit_per_min=_RATE_LIMIT)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

UI_FILE = os.path.join(os.path.dirname(__file__), "ui.html")
LANDING_FILE = os.path.join(os.path.dirname(__file__), "landing.html")
OG_IMAGE = os.path.join(os.path.dirname(__file__), "og-image.png")

_MAX_NAME = 256


@app.get("/", include_in_schema=False)
def serve_landing():
    return FileResponse(LANDING_FILE, media_type="text/html")


@app.get("/sandbox", include_in_schema=False)
def serve_ui():
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/og-image.png", include_in_schema=False)
def serve_og_image():
    return FileResponse(OG_IMAGE, media_type="image/png")


class MatchRequest(BaseModel):
    entered: str = Field(..., min_length=1, max_length=_MAX_NAME)
    actual: str = Field(..., min_length=1, max_length=_MAX_NAME)


class TransactionSignals(BaseModel):
    input_method: Literal["paste", "type"] = "type"
    paste_trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    amount: int = Field(default=0, ge=0, le=10_000_000_000)


class PayeeSignals(BaseModel):
    vpa_age_days: int = Field(default=999, ge=0, le=36500)
    mule_flagged: bool = False
    mule_flag_count: int = Field(default=0, ge=0, le=10000)
    name_score: float = Field(default=1.0, ge=0.0, le=1.0)


class TemporalSignals(BaseModel):
    first_time_payee: bool = False
    amount_vs_avg_ratio: float = Field(default=1.0, ge=0.0, le=1000.0)


class AttributionRequest(BaseModel):
    transaction: TransactionSignals
    payee: PayeeSignals
    temporal: TemporalSignals


class ResolveRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=_MAX_NAME)
    vpa: Optional[str] = Field(default=None, max_length=_MAX_NAME)
    account: Optional[str] = Field(default=None, max_length=_MAX_NAME)
    mobile: Optional[str] = Field(default=None, max_length=32)


# ── Lazy-load matchers so the server starts instantly and only loads ─
# ── the heavy ML models (L4/L5) the first time they're actually hit ──
_matchers_cache = {}


_ML_LEVELS = {4, 5}  # require torch / sentence-transformers — not on Vercel


def get_matcher(level: int):
    if level in _matchers_cache:
        return _matchers_cache[level]

    if level in _ML_LEVELS:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Level {level} uses PyTorch/sentence-transformers and is not "
                "available on this deployment. Run the full stack locally: "
                "`python run.py`"
            ),
        )

    if level == 1:
        from matchers.l1_fuzzy import match
    elif level == 2:
        from matchers.l2_phonetic import match
    elif level == 3:
        from matchers.l3_tfidf import match
    else:
        raise HTTPException(status_code=404, detail=f"No matcher for level {level}")

    _matchers_cache[level] = match
    return match


@app.get("/health")
def health():
    return {"status": "ok", "service": "payeecheck-name-matching",
             "levels_available": [1, 2, 3, 6],
             "levels_local_only": [4, 5]}


@app.post("/match/{level}")
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


@app.post("/compare")
def compare_all_levels(req: MatchRequest):
    """
    Run the same name pair through every level (1-5) and return all
    results together — useful for seeing accuracy improve level by level
    on a single example.
    """
    results = {}
    for level in (1, 2, 3):
        matcher = get_matcher(level)
        results[f"level_{level}"] = matcher(req.entered, req.actual)
    for level in (4, 5):
        results[f"level_{level}"] = {"available": False, "reason": "ML levels require local deployment"}
    return {"entered": req.entered, "actual": req.actual, "results": results}


@app.post("/resolve")
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


@app.get("/rings")
def get_fraud_rings(min_size: int = Query(default=2, ge=2, le=20)):
    """Level 6 — scan the seeded entity graph for fraud rings."""
    from matchers.l6_graph import find_rings
    rings = find_rings(min_ring_size=min_size)
    return {"ring_count": len(rings), "rings": rings}


@app.post("/attribution")
def get_attribution(req: AttributionRequest):
    """
    Three-layer attribution record. Pass transaction / payee / temporal
    signal dicts — see attribution.py docstring for the expected shape.
    """
    from attribution import build_attribution
    return build_attribution(
        req.transaction.model_dump(),
        req.payee.model_dump(),
        req.temporal.model_dump(),
    )
