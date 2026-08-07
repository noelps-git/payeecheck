"""
test_cases.py — Canonical Indian FinCrime name-pair test set.

Used by every level (L1-L6) and by benchmark.py so accuracy can be
compared apples-to-apples as the algorithm improves level over level.

Each case: (entered_name, actual_name, expected_match, category, notes)
expected_match is one of: "exact", "close", "no_match"
"""

TEST_CASES = [
    # ── Clean / exact matches ──────────────────────────────────────
    ("Amazon Pay", "Amazon Pay", "exact", "clean", "Identical strings"),
    ("PhonePe", "PhonePe", "exact", "clean", "Identical strings"),

    # ── Entity suffix noise ─────────────────────────────────────────
    ("Suresh Kumar Pvt Ltd", "Suresh Kumar", "close", "entity_suffix",
     "Business suffix added — entity type mismatch"),
    ("Krishna Enterprises", "Krishna Enterprises Pvt Ltd", "exact", "entity_suffix",
     "Same entity, suffix variant"),

    # ── Transliteration variants ────────────────────────────────────
    ("Mohammed Riaz", "Mohammad Riaz", "exact", "transliteration",
     "Common Islamic name spelling variant"),
    ("Muhammed Khan", "Mohammed Khan", "exact", "transliteration",
     "Triple spelling variant root"),

    # ── Abbreviations (the hardest category for L1-L3) ──────────────
    ("SBI", "State Bank of India", "exact", "abbreviation",
     "L1/L2 FAIL — requires semantic understanding (L4+)"),
    ("HDFC", "HDFC Bank", "exact", "abbreviation",
     "Common shorthand used in casual transfers"),
    ("PNB", "Punjab National Bank", "exact", "abbreviation",
     "Standard banking abbreviation"),

    # ── Word order (South Indian naming pattern) ─────────────────────
    ("Kumar Suresh", "Suresh Kumar", "exact", "word_order",
     "Reversed given/family name order"),
    ("Muthu Kumar", "Kumar Muthu", "exact", "word_order",
     "Reversed compound name"),

    # ── Compound / hyphenation variants ──────────────────────────────
    ("Muthukumar", "Muthu Kumar", "close", "compound",
     "One word vs two words — same name"),
    ("Venkata Raman", "Venkataraman", "close", "compound",
     "Long South Indian compound name"),

    # ── Honorifics and initials ───────────────────────────────────────
    ("Shri Ramesh Deepak", "Ramesh Deepak", "exact", "honorific",
     "Honorific prefix should not affect match"),
    ("R. Deepak", "Ramesh Deepak", "close", "honorific",
     "Initial vs full first name"),

    # ── Same group, different legal entity (should NOT auto-match) ────
    ("Reliance Jio", "Reliance Industries", "no_match", "entity_disambiguation",
     "Same corporate group — but legally distinct entities. Must not collapse."),
    ("HDFC Bank", "HDFC Life", "no_match", "entity_disambiguation",
     "Same brand family — different regulated entity (bank vs insurer)"),

    # ── Mule account / fraud patterns ──────────────────────────────────
    ("Krishna Enterprises", "Ramesh D", "no_match", "mule_pattern",
     "Business name entered, individual account on file — classic mule pattern"),
    ("HDFC Support", "Mohammed Riaz K", "no_match", "mule_pattern",
     "Fake bank-support name vs real account holder — post-payment catch case"),

    # ── Look-alike / impersonation VPA name patterns ─────────────────
    ("Paytm Support", "Paytm", "close", "lookalike",
     "Scam VPA display name impersonating legitimate merchant"),
    ("Amazon Pay India", "Amazon Pay", "exact", "lookalike",
     "Legitimate regional variant — should NOT be flagged as impersonation"),

    # ── Completely unrelated (sanity check) ───────────────────────────
    ("John Smith", "Jane Doe", "no_match", "unrelated", "No relationship at all"),
    ("Axis Bank", "Wipro Limited", "no_match", "unrelated",
     "Bank vs unrelated IT company"),
]


def cases_by_category():
    """Group test cases by category for category-level accuracy reporting."""
    grouped = {}
    for entered, actual, expected, category, notes in TEST_CASES:
        grouped.setdefault(category, []).append(
            (entered, actual, expected, notes)
        )
    return grouped


if __name__ == "__main__":
    grouped = cases_by_category()
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Categories: {len(grouped)}\n")
    for cat, cases in grouped.items():
        print(f"  {cat}: {len(cases)} cases")
