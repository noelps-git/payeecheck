"""
api_v1/storage.py — SQLite persistence layer for PayeeCheck.

Tables:
  sandbox_visitors — who accessed the sandbox (name or email + session token)
  sandbox_queries  — messages sent via the in-sandbox query flow

Usage:
    from api_v1.storage import init_db, DB_PATH
    init_db()   # called once at server startup via api.py lifespan
"""
import sqlite3
import os
from pathlib import Path

# On Vercel the filesystem is read-only except /tmp.
# Locally the DB lives next to the repo root.
_DEFAULT_PATH = (
    "/tmp/payeecheck.db"
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV")
    else str(Path(__file__).parent.parent / "payeecheck.db")
)
DB_PATH = os.environ.get("SQLITE_PATH", _DEFAULT_PATH)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sandbox_visitors (
            id            TEXT PRIMARY KEY,
            name          TEXT,
            email         TEXT,
            ip            TEXT,
            user_agent    TEXT,
            session_token TEXT NOT NULL UNIQUE,
            visited_at    TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sandbox_queries (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            email      TEXT,
            message    TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def insert_visitor(
    visitor_id: str,
    session_token: str,
    name: str | None,
    email: str | None,
    ip: str,
    user_agent: str,
) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO sandbox_visitors
               (id, name, email, ip, user_agent, session_token)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (visitor_id, name, email, ip, user_agent, session_token),
    )
    conn.commit()
    conn.close()


def insert_query(
    query_id: str,
    name: str | None,
    email: str | None,
    message: str,
) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO sandbox_queries (id, name, email, message)
           VALUES (?, ?, ?, ?)""",
        (query_id, name, email, message),
    )
    conn.commit()
    conn.close()
