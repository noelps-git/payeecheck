"""
run.py — Start the PayeeCheck local API server.

Usage:
    python run.py

Then visit http://localhost:8000 in your browser — this serves the live
PayeeCheck phone-UI prototype directly, calling the real Risk Scorer.

Interactive API docs (Swagger UI) are at http://localhost:8000/docs
"""
import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("PAYEECHECK_HOST", "127.0.0.1")
    port = int(os.environ.get("PAYEECHECK_PORT", "8000"))
    reload = os.environ.get("PAYEECHECK_RELOAD", "").lower() in ("1", "true", "yes")
    print("\nStarting PayeeCheck...")
    print(f"UI:   http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs\n")
    uvicorn.run("api:app", host=host, port=port, reload=reload)
