# PayeeCheck

**India processes 17 billion UPI transactions a month. None of them verify the payee's name.**

PayeeCheck is a fraud-detection API for Indian BFSI that addresses the structural gap between UPI's current architecture and the Confirmation of Payee (CoP) standard mandated in the UK (2020) and EU (2025). Seven signal modules, a trained edge-feature Graph Attention Network for mule ring detection, and a three-layer attribution record designed for the regulatory era ahead.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8+-blueviolet?style=flat-square)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/Live_Demo-Sandbox-1444FF?style=flat-square)](https://payeecheck.vercel.app)

---

## The problem

UPI routes money via account number and VPA. The name you see on a payment screen is a **display label** — it is never cross-verified against the account holder's KYC name at the point of payment. An attacker can register any display name on any VPA. The payer sees a familiar name; the money goes wherever the account number points.

This is the structural gap behind a large share of UPI social engineering fraud: fake bank-support VPAs, look-alike merchant handles, and mule accounts receiving "refund" transfers. The UK mandated Confirmation of Payee in 2020 and saw ~17% APP fraud reduction in year one. India's RBI has proposed an equivalent. PayeeCheck is a working proof-of-concept of what that implementation looks like.

---

## Live demo

**→ [payeecheck.vercel.app](https://payeecheck.vercel.app)**

13 deterministic scenarios — clean payment, mule account, consortium block, brand impersonation, clipboard scam, fraud ring coordinator, sanctions hit, session anomaly, and more. Select any scenario to watch the 7-signal pipeline animate in real time with full attribution output.

No API key required in sandbox mode.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            PayeeCheck API                │
                    │       POST /v1/payee-checks              │
                    └─────────────────┬────────────────────────┘
                                      │
           ┌──────────────────────────┼───────────────────────────┐
           │                          │                           │
  ┌────────▼────────┐     ┌───────────▼──────────┐     ┌─────────▼──────────┐
  │   LAYER 1       │     │   LAYER 2             │     │   LAYER 3          │
  │   Transaction   │     │   Payee Network       │     │   Temporal         │
  │   Signals       │     │   Signals             │     │   Behaviour        │
  └────────┬────────┘     └───────────┬──────────┘     └─────────┬──────────┘
           │                          │                           │
  ┌────────▼──────────┐   ┌───────────▼───────────────────────────▼──────────┐
  │  Behavioural SDK  │   │             Signal Modules (6 + GAT)              │
  │  · paste vs type  │   │                                                   │
  │  · app-switch     │   │  ① Name Matching      L1→L6  Jaro-Winkler→Graph  │
  │    cross-copy     │   │  ② Look-alike VPA     L1→L5  Brand Token Registry│
  │  · device FP      │   │  ③ Mule Database      L1→L4  Static→Consortium   │
  │  · per-user       │   │  ④ VPA Velocity       L1→L4  Rolling→Forecast    │
  │    z-score        │   │  ⑤ Sanctions Screen   Fuzzy  PEP / OFAC match    │
  └────────┬──────────┘   │  ⑥ GAT Ring Detector  PyG    Edge-feature GAT    │
           │              └───────────────────────────────────────┬───────────┘
           │                                                      │
           └──────────────────────────┬───────────────────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │    Risk Scorer       │
                           │  Weighted aggregation│
                           │  + attribution record│
                           └──────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │  BLOCK / WARN / PASS │
                           │  risk_score: 0–100   │
                           │  attribution: L1+L2+L3│
                           └─────────────────────┘
```

---

## Signal modules

| # | Module | Depth | Key technique | Attribution layer |
|---|--------|-------|---------------|-------------------|
| 1 | **Name Matching** | L1 → L6 | Jaro-Winkler → phonetic → TF-IDF char n-grams → sentence embeddings → Siamese fine-tune → graph entity resolution | L2 |
| 2 | **Look-alike VPA Detection** | L1 → L5 | Protected Brand Token Registry (NPCI-style, zero ML) — catches `paytm.support@axl` patterns that string distance cannot | L2 |
| 3 | **Mule Account Database** | L1 → L4 | Static flag lookup → velocity scoring (fan-in, pass-through speed, balance volatility) → consortium shared DB → graph ring propagation | L2 |
| 4 | **Behavioural Signal SDK** | L1 → L4 | Client-side paste vs type detection, app-switch cross-copy chain, device fingerprint, per-user z-score baseline | L1 |
| 5 | **VPA Age + Velocity Engine** | L1 → L4 | Rolling window (24h/7d/30d), amount-ratio scoring, salary-cycle-aware seasonal z-score, linear trend early-warning forecast | L2 |
| 6 | **Sanctions Screen** | Fuzzy | PEP and sanctions list fuzzy name match — hard block at threshold, weight 1.00 | L2 |
| 7 | **GAT Ring Detector** | PyTorch Geometric | `GATConv(edge_dim=…)` edge-feature attention — catches fraud rings via shared mobile/device attributes, not just individual flags | L2 |

---

## Benchmark results

> All numbers are from a synthetic 1,000-transaction dataset. See [honest framing](#honest-framing).

### Full pipeline (7 signal modules)

| Metric | Result | Notes |
|--------|--------|-------|
| Precision | **1.00** | Zero false positives across 700 clean transactions |
| Recall | **82.6%** | Up from 47% after fixing a real dataset generator bug |
| F1 | **0.905** | |
| AUPRC | **~0.996** | |
| Throughput | **2.2ms / tx** | CPU, full 7-signal pipeline |

### GAT ring detector ablation (636-node synthetic graph)

| Condition | Precision |
|-----------|-----------|
| Full model — with edge features | **95.6%** |
| NoEdge ablation — edge features zeroed | **88.9%** |
| **Edge feature contribution** | **+6.7pp** |

---

## GAT architecture

The ring detector uses PyTorch Geometric's `GATConv(edge_dim=…)` — edge features concatenated directly into the attention coefficient computation, informed by SCAFDS (Uddin, 2026) Stage 3:

```
α_vu = softmax(LeakyReLU(aᵀ[Wh_v ‖ Wh_u ‖ e_vu]))
```

Shared-attribute edges (mobile number, device fingerprint) carry fraud co-occurrence weights. A node flagged as a mule propagates risk to all co-attribute neighbours even if those accounts have zero individual flags — directly implementing the MAST (Mule Account Sharing Tool) pattern at graph level.

---

## Quick start

```bash
git clone https://github.com/noelps-git/payeecheck.git
cd payeecheck

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run.py
```

Open **http://localhost:8000** — the live B2B dashboard.
Swagger UI: **http://localhost:8000/docs**

> **Note:** First run of L4/L5 name matching downloads `all-MiniLM-L6-v2` (~90MB) from HuggingFace. Needs internet once, then cached locally. L1–L3 and the GAT run fully offline.

### Sandbox (no API key required)

```bash
curl -X POST http://localhost:8000/sandbox/v1/payee-checks \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "clipboard_scam"}'
```

**Available scenario IDs:** `clean` · `mule` · `mule_consortium` · `lookalike` · `clipboard_scam` · `biz_individual_mismatch` · `name_mismatch_abbr` · `fresh_vpa` · `ring_member` · `ring_satellite` · `sanction` · `seasonal_burst` · `session_anomaly`

### Score a transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "entered_name": "Paytm Support",
    "actual_name": "Vikram Nair",
    "payee_vpa": "paytm.support@axl",
    "amount": 45000,
    "mule_flagged": false,
    "mule_flag_count": 0,
    "vpa_age_days": 4,
    "prior_tx_count": 0,
    "unique_senders_7d": 0,
    "input_method": "paste",
    "paste_trust_score": 0.08,
    "first_time_payee": true,
    "amount_vs_avg_ratio": 3.2
  }'
```

### Name matching only

```bash
# L3 TF-IDF (no model download)
curl -X POST http://localhost:8000/match/3 \
  -H "Content-Type: application/json" \
  -d '{"entered": "SBI", "actual": "State Bank of India"}'

# L4 embeddings (downloads model first run)
curl -X POST http://localhost:8000/match/4 \
  -H "Content-Type: application/json" \
  -d '{"entered": "Mohammed Riaz", "actual": "Mohammad Riaz K"}'
```

### Run the unit tests

```bash
pip install -r requirements-dev.txt
pytest                                  # tests/unit — offline, no model downloads
pytest --cov=. --cov-report=term-missing
```

### Run benchmarks

```bash
# Generate the synthetic dataset
python data/generate_dataset.py --n 1000 --seed 42

# Full pipeline precision/recall/AUPRC
python tests/benchmark_pipeline.py --data data/synthetic_transactions.csv

# Name matching level comparison
python tests/benchmark.py --levels 1,2,3,4,5

# Retrain the GAT ring detector
python gnn/gat_mule_detector.py --epochs 100
```

---

## Configuration

All configuration is via environment variables — no config files needed.

| Variable | Default | Description |
|---|---|---|
| `PAYEECHECK_API_KEY` | *(unset — open)* | When set, all scoring/matching endpoints require `X-API-Key: <value>` header. Comparison is timing-safe (`hmac.compare_digest`). Sandbox and health endpoints stay public regardless. |
| `PAYEECHECK_ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Comma-separated CORS whitelist. Set to your deployed domain(s) in production. `*` is honoured only if you explicitly pass it. |
| `PAYEECHECK_HOST` | `127.0.0.1` | Server bind address. Set to `0.0.0.0` to expose on the network. |
| `PAYEECHECK_PORT` | `8000` | Server port. |
| `PAYEECHECK_RELOAD` | *(off)* | Set to `1` or `true` to enable uvicorn hot-reload (development only). |

Example — lock down for internal deployment:

```bash
PAYEECHECK_API_KEY=my-secret-key \
PAYEECHECK_ALLOWED_ORIGINS=https://dashboard.yourbank.com \
PAYEECHECK_HOST=0.0.0.0 \
python run.py
```

---

## Repo structure

```
payeecheck/
│
├── README.md
├── LICENSE
├── requirements.txt
├── run.py                          # Entry point → http://localhost:8000
├── api.py                          # FastAPI app — all routes
├── attribution.py                  # Three-layer attribution record builder
│
├── docs/
│   └── index.html                  # Landing page (payeecheck.vercel.app)
│
├── static/
│   └── index.html                  # B2B dashboard — served at GET /
│
├── matchers/                       # Name Matching (L1 → L6)
│   ├── corpus.py                   # Real NPCI/RBI seed data + MCA loader
│   ├── l1_fuzzy.py                 # Jaro-Winkler + Token Sort
│   ├── l2_phonetic.py              # + Double Metaphone
│   ├── l3_tfidf.py                 # + TF-IDF char n-grams
│   ├── l4_embeddings.py            # + sentence-transformers
│   ├── l5_siamese.py               # + fine-tuned Siamese network
│   ├── l5_lookalike.py             # Look-alike VPA classifier
│   ├── l6_graph.py                 # Graph entity resolution (NetworkX)
│   └── train_l5.py                 # L5 training script
│
├── risk_engine/                    # Phase 2 — Risk Scorer
│   ├── risk_scorer.py              # Main entry — combines all 7 signals
│   ├── sanctions_screening.py      # PEP/sanctions fuzzy match
│   ├── sanctions_loader.py         # List management
│   ├── sanctions_data.json         # Seed lists (illustrative)
│   ├── mule_consortium.py          # Cross-bank consortium simulation
│   ├── velocity_advanced.py        # Rolling window + seasonal z-score
│   ├── bilinear_fusion.py          # Bilinear score fusion (experimental)
│   └── calibrate_weights.py        # Signal weight calibration tooling
│
├── gnn/                            # Graph Attention Network ring detector
│   ├── gat_mule_detector.py        # Model definition + training loop
│   ├── gat_mule_detector.pt        # Trained weights (636-node synthetic graph)
│   ├── gat_scorer.py               # Inference wrapper for risk_scorer
│   ├── graph_smote.py              # Graph-aware oversampling (experimental)
│   └── transaction_graph.py        # Transaction-edge graph (in progress)
│
├── sandbox/
│   └── router.py                   # 13 scenario IDs → fixed responses, no DB writes
│
├── data/
│   ├── generate_dataset.py
│   ├── synthetic_transactions.csv
│   ├── synthetic_ring_accounts.csv
│   ├── benchmark_report_final.json
│   └── README.md
│
├── tests/
│   ├── unit/                       # pytest suite — offline, no model downloads
│   ├── test_cases.py               # Shared Indian FinCrime test set
│   ├── test_l1.py → test_l6.py     # standalone level runners (not pytest)
│   ├── benchmark.py
│   └── benchmark_pipeline.py
│
└── CHANGELOG.pdf                   # Build log — decisions, real bugs found and fixed
```

---

## Attribution record

Every verdict is traceable. No black box.

```json
{
  "verdict": "block",
  "risk_score": 91,
  "attribution": {
    "L1_transaction": [
      {
        "signal": "clipboard_paste",
        "value": 0.08,
        "threshold": 0.30,
        "assertion": "VPA pasted from clipboard within 4.2s of returning from another app"
      }
    ],
    "L2_payee_network": [
      {
        "signal": "lookalike_vpa",
        "value": 1.0,
        "threshold": 0.70,
        "assertion": "Protected brand 'paytm' + suspicious suffix {'verify'} — not in registered VPA list"
      },
      {
        "signal": "vpa_age",
        "value": 4,
        "threshold": 30,
        "assertion": "VPA is 4 days old with 0 prior transactions"
      }
    ],
    "L3_temporal": [
      {
        "signal": "first_time_payee",
        "value": 1,
        "threshold": 1,
        "assertion": "First transaction to this payee"
      }
    ]
  }
}
```

Designed to meet [RBI Draft Model Risk Management Guidelines (2023)](https://www.rbi.org.in) interpretability requirements — every assertion carries a specific numerical value and threshold, not just a narrative label.

---

## Regulatory context

| Jurisdiction | Mechanism | Status |
|---|---|---|
| 🇬🇧 UK | Confirmation of Payee — mandated by PSR | **Live since 2020** — ~17% APP fraud reduction in year one |
| 🇪🇺 EU | Verification of Payee — all SEPA Credit Transfer participants | **Mandated October 2025** |
| 🇮🇳 India | RBI CoP-equivalent proposal for UPI | **Proposed** — implementation timeline pending |

---

## What's real vs illustrative

| Component | Status |
|---|---|
| NPCI promoter banks, UPI-live banks list | ✅ Real — verified public data |
| Bank abbreviation aliases (SBI, PNB, BOB…) | ✅ Real |
| Known merchant VPA patterns (Amazon Pay, Paytm…) | ✅ Real — public knowledge |
| FastAPI server, endpoints, input validation, auth | ✅ Real — field-level constraints (min/max length) on all inputs; optional API key auth via `PAYEECHECK_API_KEY` |
| CORS restriction | ✅ Real — locked to localhost by default; `*` is not the default |
| GAT architecture (`GATConv(edge_dim=…)`) | ✅ Real PyTorch Geometric — trained end-to-end |
| GAT graceful degradation | ✅ Real — a GAT inference failure logs the error and omits the signal rather than crashing `/score` |
| Three-layer attribution record | ✅ Real — every assertion has a numerical value + threshold |
| Sanctions fallback | ✅ Real — corrupt or missing `sanctions_data.json` falls back to bundled seed lists at import time; never a hard crash |
| Synthetic benchmark dataset | ⚠️ Rule-generated — measures pipeline correctness, not real-world accuracy |
| L5 Siamese training pairs | ⚠️ Illustrative seed (~15 pairs) — extend with real KYC data |
| Sanctions/PEP seed lists | ⚠️ Illustrative — production requires OFAC/UN/domestic sync |
| MCA company data | ❌ Not included — requires manual download (see `matchers/corpus.py`) |

---

## Honest framing

All benchmark numbers (Precision 1.00, Recall 82.6%, AUPRC ~0.996) are from rule-generated synthetic data on 1,000 transactions. They measure whether the pipeline correctly recovers patterns **deliberately encoded in the generator** — not real-world accuracy on live bank data. This mirrors how SCAFDS (Uddin, 2026) frames its own synthetic Track B evaluation.

Real bugs found and fixed during this build:

**Benchmark methodology (initial build):**
- **Dataset inconsistency:** `mule_flagged` set independently from `mule_flag_count` — silently capped recall at 47%. Fixed: `mule_flagged = mule_flag_count > 0`. Recall jumped to 82.6%.
- **Vacuous GAT ablation:** Non-overlapping fraud/clean feature ranges made the ablation meaningless. Fixed by overlapping ranges — produced a real 6.7pp edge-feature contribution.

**Reliability / correctness (post-merge hardening):**
- **GAT scorer silent failure:** A `_WEIGHTS` name typo caused the GAT model to fail to load on every start without any error. Inference always fell back to the zero-signal default. Fixed: corrected the variable name; failures now log explicitly and the signal degrades gracefully.
- **Indexing bug in `transaction_graph.py`:** `probs[i][0]` was used in a context where `i` was an edge index, not a row index — potential out-of-bounds on larger graphs. Fixed: corrected indexing; GAT inference failure now logged rather than swallowed.
- **Silent data loss in `calibrate_weights.py`:** Rows that failed coercion were silently skipped via a bare `except: pass`. Fixed: logged and counted; bare `except` narrowed to `(ValueError, TypeError)`.
- **Sanctions screening import crash:** A corrupt or missing `sanctions_data.json` raised an unhandled exception at module import time, taking down the whole API. Fixed: falls back to bundled seed lists with a warning log.

All are documented in full in `CHANGELOG.pdf`.

---

## Research references

- **SCAFDS** (Uddin, 2026) — Primary architectural reference for Stage 3 edge-feature GAT and Stage 6 attribution-conditioned output
- **Cheng et al. (2024)** — "Graph Neural Networks for Financial Fraud Detection: A Review." Frontiers of Computer Science
- **Deprez et al. (2025)** — "Advances in Continual Graph Learning for Anti-Money Laundering Systems"
- **Poon et al. (2025)** — "LineMVGNN: Anti-Money Laundering with Line-Graph-Assisted Multi-View GNNs." MDPI AI
- **Vocalink / Mastercard (2019)** — "The Rise of the Mule." Real-world precedent for cross-bank mule detection at 100M-account scale

---

## API reference

Full Swagger UI at `/docs` when running locally.

Auth column shows requirement when `PAYEECHECK_API_KEY` is set (unset = all open).

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | None | Liveness check |
| `POST` | `/score` | API key | Full 7-signal risk check — returns verdict, risk_score, attribution |
| `POST` | `/match/{level}` | API key | Name matching at a specific level (1–6) |
| `POST` | `/compare` | API key | Run all levels and return a side-by-side comparison |
| `POST` | `/resolve` | API key | Graph entity resolution across the KYC name corpus |
| `GET` | `/rings` | API key | Return detected fraud rings (`min_size` query param, default 2) |
| `POST` | `/attribution` | API key | Build a three-layer attribution record for a given input |
| `GET` | `/sandbox/v1/scenarios` | None | List all 13 scenario IDs and their descriptions |
| `POST` | `/sandbox/v1/payee-checks` | None | Run a scenario by `scenario_id` — deterministic, no DB writes |
| `GET` | `/sandbox/v1/payee-checks/{scenario_id}` | None | Fetch a scenario response by ID |

---

## Built by

**Noel Rajakumar** — Co-founder & COO, [GoSense AI](https://gosense.ai) · Lead Product & Business, [IppoPay](https://ippopay.com)

[linkedin.com/in/noel-rajakumar](https://linkedin.com/in/noel-rajakumar) · [github.com/noelps-git](https://github.com/noelps-git) · noelrajakumarps@gmail.com

*Independent research project. Not affiliated with NPCI, RBI, or any bank.*
