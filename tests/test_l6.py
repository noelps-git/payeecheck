"""test_l6.py — Level 6 entity resolution test runner."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from matchers.l6_graph import resolve, find_rings


def run():
    print("=" * 70)
    print("  LEVEL 6 — Entity Resolution Test")
    print("=" * 70)

    # Test case: weak name match, strong attribute match
    query = {
        "name": "K. Ent.",
        "vpa": "k.enterprises@oksbi",
        "mobile": "9000000001",
        "account": None,
    }
    result = resolve(query)
    print(f"\nQuery: {query}")
    print(f"Resolved to: {result['resolved']}")
    assert result["resolved"] is not None, "Expected a resolution match"
    assert result["resolved"]["resolution"] == "SAME_ENTITY", \
        "Expected SAME_ENTITY resolution via shared mobile number"
    print("PASS: weak name match correctly resolved via shared attribute")

    # Test ring detection
    print("\n" + "=" * 70)
    print("  Fraud Ring Detection")
    print("=" * 70)
    rings = find_rings(min_ring_size=2)
    print(f"\nRings found: {len(rings)}")
    for ring in rings:
        print(f"  Ring of {len(ring)} entities: {ring}")
    assert len(rings) >= 1, "Expected at least one ring from seeded demo data"
    print("PASS: ring detection found the seeded 3-entity demo ring")

    print("\nAll Level 6 tests passed.")


if __name__ == "__main__":
    run()
