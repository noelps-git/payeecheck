"""
api_v1/agent.py — PayeeCheck AI + Agentic layer

Four agents, all powered by Groq (Llama 3.3 70B / 3.1 8B):

  POST /agent/analyze    — Fraud Analyst Agent (tool-use loop, Doc 10 §8)
  POST /agent/explain    — Plain-Language Explainer (Doc 10 §9)
  POST /agent/str-draft  — STR Draft Agent MVP (Doc 11 §2)
  POST /agent/triage     — Alert Triage Agent (Doc 11 §3)

Environment variable required:
  GROQ_API_KEY  — from console.groq.com (free tier)
"""
import json
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

router = APIRouter(prefix="/agent", tags=["AI agents"])

# ── Models ────────────────────────────────────────────────────────────
MODEL_BEST = "llama-3.3-70b-versatile"   # agents + STR draft
MODEL_FAST = "llama-3.1-8b-instant"      # explainer + triage notes


def _client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "GROQ_API_KEY not configured on this server.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


# ══════════════════════════════════════════════════════════════════════
# AGENT 1 — Fraud Analyst (tool-use loop)
# ══════════════════════════════════════════════════════════════════════

ANALYST_SYSTEM = """You are the PayeeCheck Fraud Intelligence Analyst — an AI \
that helps fintech professionals assess UPI payee risk in India.

You have access to PayeeCheck's signal pipeline as tools:
- check_name_match  : semantic name comparison (TF-IDF, L3)
- run_risk_score    : full 7-signal pipeline → verdict + score 0-100
- check_fraud_rings : graph entity resolution (L6) — ring membership
- screen_sanctions  : OFAC, UN, RBI sanctions and PEP screening

Strategy (Doc 10 §8 — signal gating):
1. If names are provided, call check_name_match first.
2. If you have name + VPA, call run_risk_score.
3. Only call check_fraud_rings / screen_sanctions when score > 40.
   Skip expensive signals when verdict is already clear.
4. Lead with verdict: PASS / WARN / BLOCK.
5. Cite specific signals and values. Maximum 4 sentences."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_name_match",
            "description": "Compare entered payee name vs actual KYC name. Returns similarity score 0–1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entered": {"type": "string"},
                    "actual":  {"type": "string"}
                },
                "required": ["entered", "actual"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_risk_score",
            "description": "Full 7-signal PayeeCheck pipeline. Returns verdict (pass/warn/block), risk_score 0–100, signals fired.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entered_name":  {"type": "string"},
                    "actual_name":   {"type": "string"},
                    "payee_vpa":     {"type": "string"},
                    "amount":        {"type": "number", "description": "INR amount"},
                    "input_method":  {"type": "string", "enum": ["type", "paste"]}
                },
                "required": ["entered_name", "actual_name", "payee_vpa"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_fraud_rings",
            "description": "Graph entity resolution — detect fraud ring membership via L6.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vpa":  {"type": "string"},
                    "name": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_sanctions",
            "description": "Screen payee name against OFAC, UN, and RBI sanctions/PEP lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    }
]


def _run_tool(name: str, args: dict) -> dict:
    try:
        if name == "check_name_match":
            from matchers.l3_tfidf import match
            return match(args["entered"], args["actual"])

        elif name == "run_risk_score":
            from risk_engine.risk_scorer import score_transaction
            return score_transaction({
                "tx_id":            "agent",
                "entered_name":     args["entered_name"],
                "actual_name":      args["actual_name"],
                "payee_vpa":        args["payee_vpa"],
                "amount":           args.get("amount", 10000),
                "input_method":     args.get("input_method", "type"),
                "vpa_age_days":     365,
                "prior_tx_count":   10,
                "unique_senders_7d": 1,
                "mule_flagged":     False,
                "mule_flag_count":  0,
            })

        elif name == "check_fraud_rings":
            from matchers.l6_graph import resolve
            return resolve({"vpa": args.get("vpa"), "name": args.get("name")})

        elif name == "screen_sanctions":
            from risk_engine.sanctions_screening import check_sanctions
            return check_sanctions(args["name"])

        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e), "tool": name}


class AnalyzeRequest(BaseModel):
    query: str


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Fraud Analyst Agent — tool-use loop over PayeeCheck signal pipeline."""
    client = _client()
    messages = [{"role": "user", "content": req.query}]
    tools_used = []

    for _ in range(6):   # max 6 iterations (3 tool pairs)
        resp = client.chat.completions.create(
            model=MODEL_BEST,
            messages=[{"role": "system", "content": ANALYST_SYSTEM}] + messages,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=800,
            temperature=0.1,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {"response": msg.content, "tools_used": tools_used}

        # Append assistant turn
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute tools + append results
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = _run_tool(tc.function.name, args)
            tools_used.append({"tool": tc.function.name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return {"response": "Analysis complete.", "tools_used": tools_used}


# ══════════════════════════════════════════════════════════════════════
# AGENT 2 — Plain-Language Explainer (Doc 10 §9)
# ══════════════════════════════════════════════════════════════════════

_EXPLAINER_SYSTEM = """You are a fraud detection analyst explaining a PayeeCheck \
verdict to a bank compliance officer.

Rules:
1. Reference only signals listed in the data. Never invent signals.
2. Plain English — no ML jargon.
3. Exactly 2-3 sentences.
4. Lead with the verdict and the single strongest signal.
5. Final sentence: one of "Allow payment.", "Apply friction screen.", or "Block payment." """


class ExplainRequest(BaseModel):
    scenario_id:  str
    verdict:      str
    risk_score:   int
    payee_vpa:    str
    entered_name: str
    actual_name:  str
    signals:      list[str]
    description:  str


@router.post("/explain")
def explain(req: ExplainRequest):
    """Plain-language verdict explanation for compliance officers."""
    client = _client()
    prompt = (
        f"Verdict: {req.verdict.upper()} (risk score: {req.risk_score}/100)\n"
        f"Scenario: {req.scenario_id}\n"
        f"Payee VPA: {req.payee_vpa}\n"
        f"Entered name: {req.entered_name} → Actual name: {req.actual_name}\n"
        f"Signals fired: {'; '.join(req.signals)}\n"
        f"Context: {req.description}\n\n"
        "Explain this verdict in plain English."
    )
    resp = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system",  "content": _EXPLAINER_SYSTEM},
            {"role": "user",    "content": prompt},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    return {"explanation": resp.choices[0].message.content}


# ══════════════════════════════════════════════════════════════════════
# AGENT 3 — STR Draft Agent (Doc 11 §2)
# ══════════════════════════════════════════════════════════════════════

_STR_SYSTEM = """You are a compliance analyst generating a Suspicious Transaction \
Report (STR) for FIU-IND under the Prevention of Money Laundering Act, 2002 \
and Rules notified thereunder.

PMLA 2002 defines a suspicious transaction as one which, to a person acting in \
good faith, (a) gives rise to reasonable grounds of suspicion that it may involve \
proceeds of crime; (b) appears to be made in circumstances of unusual or \
unjustified complexity; or (c) appears to have no economic rationale or bonafide purpose.

Generate the STR in the following exact structure:

SUSPICIOUS TRANSACTION REPORT
Under PMLA 2002 — Filed with Financial Intelligence Unit — India

1. REPORTING ENTITY DETAILS
   Name: PayeeCheck UPI Verification Service
   Registration type: Intermediary under PMLA 2002

2. SUBJECT / CLIENT DETAILS
   UPI VPA (Virtual Payment Address): [payee VPA]
   Name as entered by payer: [entered name]
   Name registered on account: [actual name]
   Identity concern: [note any name mismatch, non-face-to-face nature, or doubt over real beneficiary]

3. TRANSACTION DETAILS
   Transaction reference: [check_id]
   Transaction type: UPI payment (digital, non-cash)
   Risk score assigned: [score]/100
   Verdict: BLOCK

4. CATEGORY OF SUSPICION
   Select all applicable PMLA categories from the evidence provided:
   - Identity of Client (false/unverifiable identification, name mismatch, doubt over beneficiary)
   - Suspicious Background (links to known fraud rings, consortium flags)
   - Multiple Accounts (ring membership, shared device or mobile fingerprint)
   - Activity in Accounts (velocity anomalies, unusual sender fan-in, dormant-then-active pattern)
   - Nature of Transactions (no economic rationale, clipboard paste indicating coaching, look-alike VPA)
   - Value of Transactions (amount inconsistent with profile, structuring indicators)

5. BASIS FOR SUSPICION
   Write 150–250 words. Every sentence must cite a specific signal value from the \
   data provided. Use the PMLA category labels above to organise the narrative. \
   Example: "Under 'Nature of Transactions' — the VPA paytmsupport@ybl contained \
   the protected brand token 'paytm' but was not present in the registered VPA list, \
   indicating a look-alike VPA constructed to deceive the payer."

6. RECOMMENDED ACTION
   State one of: Freeze and investigate | Escalate to law enforcement | \
   Flag for enhanced monitoring | File STR with FIU-IND within 7 days

Formatting rules (non-negotiable):
- Third person, past tense, formal register throughout.
- Do not write 'AI detected', 'our model flagged', or 'automated system'.
- Cite exact numerical values wherever available: "The VPA was registered 11 days prior to the transaction."
- Do not invent any fact not present in the signal data provided.
- Final line must be exactly:
  "This report is a draft. Human review and approval required before submission to FIU-IND."
"""


class STRDraftRequest(BaseModel):
    check_id:     str
    scenario_id:  str
    payee_vpa:    str
    entered_name: str
    actual_name:  str
    risk_score:   int
    signals:      list[str]
    description:  str


@router.post("/str-draft")
def str_draft(req: STRDraftRequest):
    """STR Draft Agent — generates FIU-IND compliant draft. Never auto-files."""
    if req.risk_score < 85:
        raise HTTPException(400, "STR draft only triggered for risk_score >= 85.")

    client = _client()
    prompt = (
        f"Transaction reference : {req.check_id}\n"
        f"Payee VPA             : {req.payee_vpa}\n"
        f"Name entered by payer : {req.entered_name}\n"
        f"Name on account (KYC) : {req.actual_name}\n"
        f"Risk score            : {req.risk_score}/100  Verdict: BLOCK\n"
        f"Signals fired         : {'; '.join(req.signals)}\n"
        f"Context / description : {req.description}\n\n"
        "Generate the full PMLA 2002 STR using the 6-section structure in your instructions. "
        "Map each signal to the correct PMLA suspicion category. "
        "Cite specific values in Section 5. End with the mandatory disclaimer."
    )
    resp = client.chat.completions.create(
        model=MODEL_BEST,
        messages=[
            {"role": "system", "content": _STR_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.05,
    )
    return {
        "draft":        resp.choices[0].message.content,
        "draft_status": "draft — human review and approval required before filing",
        "check_id":     req.check_id,
        "model":        MODEL_BEST,
    }


# ══════════════════════════════════════════════════════════════════════
# AGENT 4 — Alert Triage Agent (Doc 11 §3)
# ══════════════════════════════════════════════════════════════════════

_TRIAGE_SYSTEM = """You are a fraud alert triage analyst.
Write exactly 2-3 sentences per alert — be specific: cite VPA, amounts, exact signals.
End every note with one recommendation:
  "Callback payer immediately." | "Monitor for 24h." | "Escalate to compliance team." """

_HIGH_WEIGHT = {"clipboard", "paste", "mule", "consortium", "brand", "impersonat", "ring"}


class TriageAlert(BaseModel):
    scenario_id:  str
    verdict:      str
    payee_vpa:    str
    risk_score:   int
    amount:       float = 50000.0
    signals:      list[str]
    description:  str


class TriageRequest(BaseModel):
    alerts: list[TriageAlert]


@router.post("/triage")
def triage(req: TriageRequest):
    """Alert Triage Agent — re-orders WARN queue by urgency, drafts investigation notes."""
    client = _client()

    # Score urgency per Doc 11 §3 formula
    scored = []
    for a in req.alerts:
        risk_c   = a.risk_score / 100
        amount_c = min(a.amount / 500_000, 1.0)
        sigs     = " ".join(a.signals).lower()
        hits     = sum(1 for hw in _HIGH_WEIGHT if hw in sigs)
        signal_c = min(hits / 3, 1.0)
        urgency  = round(0.40 * risk_c + 0.35 * amount_c + 0.25 * signal_c, 3)
        scored.append({**a.model_dump(), "urgency_score": urgency})

    scored.sort(key=lambda x: x["urgency_score"], reverse=True)

    # Draft investigation notes for top 5
    results = []
    for rank, alert in enumerate(scored[:5], 1):
        prompt = (
            f"Alert rank #{rank}\n"
            f"Payee VPA  : {alert['payee_vpa']}\n"
            f"Risk score : {alert['risk_score']}/100\n"
            f"Signals    : {'; '.join(alert['signals'])}\n"
            f"Context    : {alert['description']}\n\n"
            "Write a 2-3 sentence investigation note."
        )
        resp = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": _TRIAGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=160,
            temperature=0.1,
        )
        results.append({
            **alert,
            "urgency_rank":       rank,
            "investigation_note": resp.choices[0].message.content,
        })

    return {"triaged_alerts": results, "total_alerts": len(scored)}
