"""test_l4.py — Level 4 standalone test runner."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from matchers.l4_embeddings import match
from tests.test_cases import TEST_CASES

def run():
    passed = failed = 0
    for entered, actual, expected, category, notes in TEST_CASES:
        r = match(entered, actual)
        ok = r["match"] == expected
        passed += ok; failed += not ok
        icon = "PASS" if ok else "FAIL"
        print(f"[{icon}] ({category}) {notes}")
        print(f"       entered={entered!r}  actual={actual!r}")
        print(f"       result={r['match']} score={r['score']:.2f}")
        if not ok:
            print(f"       expected={expected}")
        print()
    print(f"{passed} passed, {failed} failed out of {len(TEST_CASES)}")

if __name__ == "__main__":
    run()
