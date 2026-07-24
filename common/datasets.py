"""
common/datasets.py — Dataset locations and CSV loading.

Every trainer/loader in the project resolved the synthetic dataset path
by hand and fell back from the v2 file to v1; that logic lives here now.
"""
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RING_ACCOUNTS_CSV = os.path.join(DATA_DIR, "synthetic_ring_accounts.csv")


def transactions_csv(path: str = None) -> str:
    """
    Path to the synthetic transactions dataset: `path` when given,
    otherwise the v2 file, falling back to v1 when v2 is absent.
    """
    if path:
        return path
    v2 = os.path.join(DATA_DIR, "synthetic_transactions_v2.csv")
    if os.path.exists(v2):
        return v2
    return os.path.join(DATA_DIR, "synthetic_transactions.csv")


def read_rows(csv_path: str) -> list:
    """Read a CSV into a list of dict rows."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))
