"""
run.py — Start the PayeeCheck local API server.

Usage:
    python run.py

Then visit http://localhost:8000 in your browser — this serves the live
PayeeCheck phone-UI prototype directly, calling the real Risk Scorer.

Interactive API docs (Swagger UI) are at http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    print("\nStarting PayeeCheck...")
    print("UI:   http://localhost:8000")
    print("Docs: http://localhost:8000/docs\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
