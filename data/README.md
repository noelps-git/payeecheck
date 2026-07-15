# Data

All datasets in this directory are **synthetically generated** — no real bank customer data is used or included anywhere in this repository.

## Files

| File | Description |
|---|---|
| `generate_dataset.py` | Generator script — 1,000 transactions across 13 fraud categories |
| `synthetic_transactions.csv` | v1 benchmark dataset (1,000 rows) |
| `synthetic_transactions_v2.csv` | v2 dataset with corrected mule flag consistency |
| `synthetic_ring_accounts.csv` | 636-node ring graph used to train the GAT detector |
| `benchmark_report_final.json` | Full precision/recall/AUPRC results from `tests/benchmark_pipeline.py` |
| `dataset_stats_v2.json` | Dataset composition statistics |

## Regenerating

```bash
python data/generate_dataset.py --n 1000 --seed 42
```

## Honest framing

The synthetic benchmark measures whether the pipeline correctly recovers fraud patterns **deliberately encoded in the generator** — not real-world accuracy. Precision 1.00 / Recall 82.6% / AUPRC ~0.996 are correct on synthetic data; real-world performance requires calibration against a labelled production dataset from a bank deployment.

The mule flag consistency bug (v1 → v2 fix) is documented in `CHANGELOG.pdf`.
