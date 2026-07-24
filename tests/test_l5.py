"""test_l5.py — Level 5 standalone test runner."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from matchers.l5_siamese import match
from tests.runner import run_match_suite

def run():
    run_match_suite(match)

if __name__ == "__main__":
    run()
