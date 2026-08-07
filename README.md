# PayeeCheck — Name Matching Intelligence (L1-L6)

Local, runnable implementation of the PayeeCheck Engineering Playbook.
Six levels of name matching, a three-layer attribution record, and a
local FastAPI server you can hit from a browser or curl.

## Setup

```bash
cd payeecheck
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run of Level 4/5/6 will download the `all-MiniLM-L6-v2` model
(~90MB) from HuggingFace — needs an internet connection once, then it's
cached locally.

## Run the local server

```bash
python run.py
```

Open **http://localhost:8000/docs** — interactive Swagger UI, no Postman needed.

Quick test from terminal:
```bash
curl -X POST http://localhost:8000/match/4 \
  -H "Content-Type: application/json" \
  -d '{"entered": "SBI", "actual": "State Bank of India"}'
```

## Run standalone (no server)

```bash
python -m tests.test_l1        # Level 1 only
python -m tests.test_l4        # Level 4 only (downloads model first time)
python tests/benchmark.py --levels 1,2,3,4,5    # compare all levels side by side
```

## Train Level 5 (optional)

```bash
python matchers/train_l5.py
```

This uses a small illustrative training set (~15 pairs) baked into
`train_l5.py` — enough to make the pipeline run, **not enough to beat
Level 4 in production**. To get real gains, replace the training pairs
with your own KYC data or a real MCA company export (instructions in
`matchers/corpus.py`).

## What's real vs illustrative

| Component | Status |
|---|---|
| NPCI promoter banks, UPI-live banks list | Real, verified public data |
| Bank abbreviation aliases (SBI, PNB, BOB...) | Real, commonly used in practice |
| Known merchant VPA patterns (Amazon Pay, Paytm...) | Real, public knowledge |
| L5 training pairs | Illustrative seed only — extend with real data |
| L6 fraud ring demo entities | Constructed for demonstration, not real fraud cases |
| MCA company data | Not included — requires manual download, see `corpus.py` |
| IEEE-CIS fraud dataset | Not included — requires free Kaggle account |

## Project structure

```
payeecheck/
├── matchers/
│   ├── corpus.py          # Real seed data + MCA CSV loader
│   ├── l1_fuzzy.py         # Level 1: Jaro-Winkler + Token Sort
│   ├── l2_phonetic.py      # Level 2: + Double Metaphone
│   ├── l3_tfidf.py         # Level 3: + TF-IDF char n-grams
│   ├── l4_embeddings.py    # Level 4: + sentence embeddings
│   ├── l5_siamese.py       # Level 5: fine-tuned model matcher
│   ├── train_l5.py         # Level 5: training script
│   └── l6_graph.py         # Level 6: graph entity resolution
├── tests/
│   ├── test_cases.py       # Shared Indian FinCrime test set
│   ├── test_l1.py ... test_l6.py
│   └── benchmark.py        # Side-by-side comparison across levels
├── attribution.py          # Three-layer attribution record
├── api.py                  # FastAPI app
├── run.py                  # Server entry point
└── requirements.txt
```

## Extending with real MCA data

```bash
# 1. Download a state CSV (free, no login):
#    https://www.data.gov.in/catalog/company-master-data
# 2. Save it to payeecheck/data/mca_companies.csv
# 3. In a Python shell:
from matchers.corpus import load_from_mca_csv
real_names = load_from_mca_csv("data/mca_companies.csv")
# Add real_names into the CORPUS list in corpus.py, re-fit L3's TfidfVectorizer
```
