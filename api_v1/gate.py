"""
api_v1/gate.py — Sandbox gate endpoints.

POST /gate/identify  — visitor submits name or email; gets a session token
POST /gate/query     — visitor sends a message/question to the owner

No auth required — these are the public entry points.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import uuid

from api_v1.storage import insert_visitor, insert_query

router = APIRouter(prefix="/gate", tags=["gate"])


class IdentifyRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class QueryRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    message: str


@router.post("/identify")
def identify(req: IdentifyRequest, request: Request):
    """
    Gate the sandbox. Visitor provides a name or email (or both).
    Returns a session_token stored in localStorage by the frontend.
    """
    name = (req.name or "").strip() or None
    email = (req.email or "").strip() or None

    if not name and not email:
        raise HTTPException(
            status_code=400,
            detail="Provide at least a name or email to access the sandbox.",
        )

    session_token = str(uuid.uuid4())
    visitor_id = str(uuid.uuid4())
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")

    insert_visitor(visitor_id, session_token, name, email, ip, ua)

    return {"ok": True, "session_token": session_token}


@router.post("/query")
def submit_query(req: QueryRequest):
    """
    Visitor sends a message or question. Stored in sandbox_queries table.
    """
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    query_id = str(uuid.uuid4())
    insert_query(
        query_id=query_id,
        name=(req.name or "").strip() or None,
        email=(req.email or "").strip() or None,
        message=message,
    )

    return {"ok": True, "id": query_id}
