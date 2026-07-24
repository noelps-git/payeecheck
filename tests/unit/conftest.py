"""Shared fixtures for the unit test suite."""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CSV_HEADER = (
    "tx_id,entered_name,actual_name,payee_vpa,amount,input_method,"
    "vpa_age_days,prior_tx_count,unique_senders_7d,mule_flagged,"
    "mule_flag_count,is_lookalike,name_score,fraud_ring_id,bank_flags,label\n"
)


def _row(**kw) -> str:
    defaults = {
        "tx_id": "1", "entered_name": "A", "actual_name": "A",
        "payee_vpa": "a@ybl", "amount": "1000", "input_method": "type",
        "vpa_age_days": "400", "prior_tx_count": "20", "unique_senders_7d": "2",
        "mule_flagged": "False", "mule_flag_count": "0", "is_lookalike": "False",
        "name_score": "0.99", "fraud_ring_id": "", "bank_flags": "0",
        "label": "clean",
    }
    defaults.update({k: str(v) for k, v in kw.items()})
    order = CSV_HEADER.strip().split(",")
    return ",".join(defaults[c] for c in order) + "\n"


@pytest.fixture
def make_csv(tmp_path):
    """Write a synthetic-transactions-shaped CSV from row overrides."""
    def _make(rows, name="tx.csv"):
        path = tmp_path / name
        path.write_text(CSV_HEADER + "".join(_row(**r) for r in rows))
        return str(path)
    return _make
