"""
runner.py — Shared test runner for the level test scripts.

Every level test walks the same canonical TEST_CASES and prints the same
report; only the matcher (and whether it returns a `reason`) differs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tests.test_cases import TEST_CASES


def run_match_suite(match) -> tuple:
    """Run `match(entered, actual)` over TEST_CASES; return (passed, failed)."""
    passed = failed = 0
    for entered, actual, expected, category, notes in TEST_CASES:
        r = match(entered, actual)
        ok = r["match"] == expected
        passed += ok
        failed += not ok
        icon = "PASS" if ok else "FAIL"
        print(f"[{icon}] ({category}) {notes}")
        print(f"       entered={entered!r}  actual={actual!r}")
        detail = f"       result={r['match']} score={r['score']:.2f}"
        if "reason" in r:
            detail += f"  reason={r['reason']}"
        print(detail)
        if not ok:
            print(f"       expected={expected}")
        print()
    print(f"{passed} passed, {failed} failed out of {len(TEST_CASES)}")
    return passed, failed
