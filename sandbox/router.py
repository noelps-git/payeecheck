"""
sandbox/router.py — Deterministic sandbox environment for PayeeCheck.

Every call to POST /sandbox/v1/payee-checks returns a fixed, pre-computed
response based on the scenario_id field in the request body. No live
scoring, no database writes, no API key required.

Purpose: bank integration teams write integration tests against deterministic
responses before touching production. Also serves as the always-on demo
environment pointed to by the launch post.

13 scenario IDs map to the 13 synthetic dataset types:
  clean, mule, mule_consortium, lookalike, clipboard_scam,
  biz_individual_mismatch, name_mismatch_abbr, fresh_vpa,
  ring_member, ring_satellite, sanction, seasonal_burst, session_anomaly

Any request without a scenario_id, or with an unrecognised one, returns
the clean scenario by default.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

router = APIRouter(prefix="/sandbox/v1", tags=["sandbox — deterministic demo"])

# ── Deterministic scenario library ───────────────────────────────────────────
SCENARIOS = {
    "clean": {
        "verdict": "pass", "risk_score": 3,
        "entered_name": "Amazon Pay", "actual_name": "Amazon Pay",
        "payee_vpa": "amazon.pay@apl",
        "signals": ["Exact name match", "VPA age: 5+ years", "Clean velocity"],
        "description": "All checks passed. Verified merchant, long-established VPA, zero flags.",
    },
    "mule": {
        "verdict": "block", "risk_score": 88,
        "entered_name": "Krishna Enterprises", "actual_name": "Ramesh Kumar",
        "payee_vpa": "ramesh.001@ybl",
        "signals": ["Mule flag: 1 bank", "Name mismatch (individual vs business)", "High velocity"],
        "description": "Account flagged as mule by one bank. Entity type mismatch detected.",
    },
    "mule_consortium": {
        "verdict": "block", "risk_score": 95,
        "entered_name": "Quick Transfer Services", "actual_name": "Pooja Pillai",
        "payee_vpa": "pooja.455@okaxis",
        "signals": ["Mule consortium: 3 banks", "Pass-through velocity 94th pct", "Name mismatch"],
        "description": "Hard override: 3 independent banks have reported this account as a mule.",
    },
    "lookalike": {
        "verdict": "block", "risk_score": 91,
        "entered_name": "Paytm Verify", "actual_name": "Vikram Nair",
        "payee_vpa": "paytm.verify@axl",
        "signals": ["Protected brand token: paytm", "VPA age: 4 days", "Unregistered PSP handle"],
        "description": "VPA impersonates Paytm. Brand token present in unregistered handle.",
    },
    "clipboard_scam": {
        "verdict": "warn", "risk_score": 62,
        "entered_name": "HDFC Support", "actual_name": "Mohammed Riaz",
        "payee_vpa": "hdfc.support@paytrn",
        "signals": ["Clipboard paste detected", "VPA age: 11 days", "43 unique senders 7d"],
        "description": "Pasted from clipboard. HDFC does not use @paytrn handle. High inflow velocity.",
    },
    "biz_individual_mismatch": {
        "verdict": "warn", "risk_score": 48,
        "entered_name": "Suresh Enterprises", "actual_name": "Suresh Kumar",
        "payee_vpa": "suresh.kumar@hdfcbank",
        "signals": ["Name score: 0.61", "Entity type: individual account, business name entered"],
        "description": "Close match but entity type mismatch. Established VPA, no other flags.",
    },
    "name_mismatch_abbr": {
        "verdict": "warn", "risk_score": 35,
        "entered_name": "SBI", "actual_name": "State Bank of India",
        "payee_vpa": "sbi.01@oksbi",
        "signals": ["Abbreviation match: 0.38", "Long-established VPA", "No other flags"],
        "description": "Abbreviation vs full name mismatch. Likely legitimate — low risk score.",
    },
    "fresh_vpa": {
        "verdict": "warn", "risk_score": 58,
        "entered_name": "Newbiz Solutions", "actual_name": "Newbiz Solutions",
        "payee_vpa": "newbiz.new@ybl",
        "signals": ["VPA age: 6 days", "Zero prior transactions", "Amount: Rs 2,30,000"],
        "description": "Exact name match but VPA is 6 days old with no transaction history.",
    },
    "ring_member": {
        "verdict": "block", "risk_score": 87,
        "entered_name": "Ananya Sharma", "actual_name": "Ananya Sharma",
        "payee_vpa": "ananya.123@paytm",
        "signals": ["Mule flag: 2 banks", "Ring coordinator", "Shared device: 4 accounts"],
        "description": "Account is the coordinator of a detected fraud ring. 4 accounts share device.",
    },
    "ring_satellite": {
        "verdict": "warn", "risk_score": 44,
        "entered_name": "Rohan Patel", "actual_name": "Rohan Patel",
        "payee_vpa": "rohan.456@okaxis",
        "signals": ["No individual flag", "2nd-degree ring proximity", "Shared mobile with ring coordinator"],
        "description": "Not individually flagged but shares mobile number with a confirmed mule ring coordinator.",
    },
    "sanction": {
        "verdict": "block", "risk_score": 97,
        "entered_name": "Global Trade Finance Ltd", "actual_name": "Global Trade Finance Ltd",
        "payee_vpa": "gtfl.pay@apl",
        "signals": ["Sanctions match: OFAC list", "Hard override applied"],
        "description": "Payee name matches a sanctioned entity. Payment blocked. Regulatory requirement.",
    },
    "seasonal_burst": {
        "verdict": "pass", "risk_score": 8,
        "entered_name": "Kavya Menon", "actual_name": "Kavya Menon",
        "payee_vpa": "kavya.789@ybl",
        "signals": ["High velocity: 47 senders 7d", "Seasonal pattern detected", "VPA age: 4 years"],
        "description": "High velocity but seasonal pattern recognised. Long-established VPA. Clean.",
    },
    "session_anomaly": {
        "verdict": "warn", "risk_score": 41,
        "entered_name": "Meera Das", "actual_name": "Meera Das",
        "payee_vpa": "meera.321@hdfcbank",
        "signals": ["Paste detected", "Session timing anomaly", "Established VPA, no fraud flags"],
        "description": "Legitimate payee but payer session shows anomalous timing — possible coercion.",
    },
}

class SandboxRequest(BaseModel):
    scenario_id: Optional[str] = "clean"
    # Any other fields accepted but ignored — makes it easy to send
    # a real PayeeCheck request body and just add scenario_id
    entered_name: Optional[str] = None
    actual_name: Optional[str] = None
    payee_vpa: Optional[str] = None
    amount: Optional[float] = None

def _build_response(scenario_id: str, req: SandboxRequest) -> dict:
    s = SCENARIOS.get(scenario_id, SCENARIOS["clean"])
    pchk_id = f"sandbox_pchk_{scenario_id}_{uuid.uuid4().hex[:8]}"
    rv_id   = f"sandbox_rv_{scenario_id}_{uuid.uuid4().hex[:8]}"
    now     = datetime.utcnow().isoformat() + "Z"

    verdict_meta = {
        "block": {"icon": "🚫", "label": "PAYMENT BLOCKED"},
        "warn":  {"icon": "⚠️",  "label": "FRICTION APPLIED"},
        "pass":  {"icon": "✅", "label": "PAYMENT PASSED"},
    }
    meta = verdict_meta[s["verdict"]]

    return {
        "sandbox": True,
        "scenario_id": scenario_id,
        "scenario_description": s["description"],
        "payee_check": {
            "id": pchk_id,
            "created_at": now,
            "bank_id": "sandbox_bank",
            "evidence": {
                "entered_name": req.entered_name or s["entered_name"],
                "actual_name":  req.actual_name  or s["actual_name"],
                "payee_vpa":    req.payee_vpa    or s["payee_vpa"],
                "signals_fired": s["signals"],
            },
        },
        "risk_verdict": {
            "id": rv_id,
            "payee_check_id": pchk_id,
            "verdict": s["verdict"],
            "verdict_label": meta["label"],
            "verdict_icon": meta["icon"],
            "risk_score": s["risk_score"],
        },
        "_note": "This is a sandbox response. All values are deterministic and fixed.",
    }

@router.get("/scenarios")
def list_scenarios():
    """List all available sandbox scenarios with their verdict and risk score."""
    return {
        "scenarios": [
            {
                "scenario_id": k,
                "verdict": v["verdict"],
                "risk_score": v["risk_score"],
                "description": v["description"],
            }
            for k, v in SCENARIOS.items()
        ]
    }

@router.post("/payee-checks")
def sandbox_payee_check(req: SandboxRequest):
    """
    Deterministic sandbox scoring. Pass scenario_id to select the scenario.
    No API key required. No database writes. Always returns the same result
    for the same scenario_id.
    """
    sid = req.scenario_id or "clean"
    return _build_response(sid, req)

@router.get("/payee-checks/{scenario_id}")
def sandbox_get_scenario(scenario_id: str):
    """Retrieve the fixed response for a named scenario."""
    class Req(BaseModel):
        scenario_id: Optional[str] = None
        entered_name: Optional[str] = None
        actual_name: Optional[str] = None
        payee_vpa: Optional[str] = None
        amount: Optional[float] = None
    return _build_response(scenario_id, Req())
