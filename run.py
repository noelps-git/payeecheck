"""
run.py — Start the PayeeCheck local API server.

Usage:
    python run.py

Then visit http://127.0.0.1:8000/docs in your browser.

Env:
    HOST      bind address (default: 127.0.0.1)
    PORT      port (default: 8000)
    RELOAD=1  enable auto-reload for development
"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "").strip() in ("1", "true", "True", "yes")

    print("\nStarting PayeeCheck Name Matching API...")
    print(f"Once running, open: http://{host}:{port}/docs\n")
    uvicorn.run("api:app", host=host, port=port, reload=reload)
