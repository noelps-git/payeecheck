"""
benchmark.py — Run all 6 matcher levels against the same Indian FinCrime
test cases and print a side-by-side comparison.

This is the single most useful file in the repo: it lets you literally
watch accuracy climb as you move from L1 to L6.

Run:
    python tests/benchmark.py                  # all levels
    python tests/benchmark.py --levels 1,2,3   # just L1-L3 (skip slow ML levels)
    python tests/benchmark.py --category abbreviation   # filter by category
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.test_cases import TEST_CASES


def load_matcher(level: int):
    if level == 1:
        from matchers import l1_fuzzy
        return l1_fuzzy.match
    elif level == 2:
        from matchers import l2_phonetic
        return l2_phonetic.match
    elif level == 3:
        from matchers import l3_tfidf
        return l3_tfidf.match
    elif level == 4:
        from matchers import l4_embeddings
        return l4_embeddings.match
    elif level == 5:
        from matchers import l5_siamese
        return l5_siamese.match
    else:
        raise ValueError(f"Level {level} has no standalone match() — "
                          f"L6 is entity resolution, run l6_graph.py directly")


def run_benchmark(levels, category_filter=None):
    cases = TEST_CASES
    if category_filter:
        cases = [c for c in cases if c[3] == category_filter]
        print(f"Filtered to category '{category_filter}': {len(cases)} cases\n")

    results = {}  # level -> (passed, failed, total_time, per_case_results)

    for level in levels:
        print(f"\n{'='*70}")
        print(f"  LEVEL {level}")
        print(f"{'='*70}")
        try:
            matcher = load_matcher(level)
        except Exception as e:
            print(f"  Could not load Level {level}: {e}")
            continue

        passed, failed = 0, 0
        per_case = []
        t0 = time.time()
        for entered, actual, expected, category, notes in cases:
            r = matcher(entered, actual)
            ok = r["match"] == expected
            passed += ok
            failed += not ok
            per_case.append((entered, actual, expected, r["match"], ok, category, notes))
            icon = "PASS" if ok else "FAIL"
            print(f"  [{icon}] ({category}) {entered!r} vs {actual!r}  "
                  f"-> {r['match']} (score={r['score']:.2f})  expected={expected}")
        elapsed = time.time() - t0

        print(f"\n  Level {level}: {passed}/{len(cases)} passed "
              f"({100*passed/len(cases):.0f}%)  —  {elapsed:.2f}s total")
        results[level] = (passed, failed, elapsed, per_case)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SUMMARY — Accuracy by Level")
    print(f"{'='*70}")
    print(f"  {'Level':<8}{'Passed':<10}{'Total':<10}{'Accuracy':<12}{'Time (s)'}")
    for level, (passed, failed, elapsed, _) in results.items():
        total = passed + failed
        acc = 100 * passed / total if total else 0
        print(f"  L{level:<7}{passed:<10}{total:<10}{acc:<10.1f}{'%':<2}{elapsed:.2f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, default="1,2,3,4,5",
                        help="Comma-separated levels to run, e.g. 1,2,3")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter test cases by category")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    run_benchmark(levels, category_filter=args.category)
