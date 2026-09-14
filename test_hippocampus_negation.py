#!/usr/bin/env python3
"""
Test Suite: Neural Hippocampus vs. the Three Adversarial Failure Modes
=======================================================================
Verifies that replacing regex parsers (System B) with a neural SLM hippocampus
(SmolLM2-1.7B-Instruct) resolves critical vulnerabilities from the red-team audit:

  VM1: Distractor Needle Spoofing (First-Match Greediness) — 83.3% failure rate
  VM2: Syntactic Paraphrase & Passive Voice Collapse       — 84.7% failure rate
  VM3: Negation & Revocation Blindness                     — 100.0% failure rate

English-only test cases. Each test runs:
  [REGEX Baseline]    → (expected FAIL)  — confirms the bug is real
  [SLM Hippocampus]   → (expected PASS)  — confirms the fix works

Author: Waldemar Gajda
"""

import re as _re
import sys
import traceback


# ─────────────────────────────────────────────────────────────────────────────
# Regex baseline (mirrors System B first-match logic from cognitive_memory_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

def regex_extract_valuation(text: str) -> str:
    m = _re.search(
        r'(?:valuation|value|price|capitalization|amount|cost)[^\d]{0,40}?'
        r'([\d\s,\.]+(?:EUR|USD|PLN|mln|M|billion|million)?)',
        text, _re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def regex_extract_key(text: str) -> str:
    m = _re.search(r'\b([A-Z][A-Z0-9\-]{4,})\b', text)
    return m.group(1) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Test cases — English only
# ─────────────────────────────────────────────────────────────────────────────

VM1_CASES = [
    {
        "id": "VM1-B",
        "description": "Stale draft key followed by production key",
        "chunk": (
            "Initial draft authorization key DRAFT-KEY-0001 was proposed for testing. "
            "After security review, DRAFT-KEY-0001 was deprecated. "
            "The production authorization key is TITAN-KEY-9901-X, effective immediately."
        ),
        "expected_value_contains": "TITAN-KEY-9901-X",
        "wrong_value_contains": "DRAFT-KEY-0001",
    },
]

VM2_CASES = [
    {
        "id": "VM2-A",
        "description": "Synonym: 'market capitalization' instead of 'valuation'",
        "chunk": "The pre-IPO market capitalization of QuantumDynamics Ltd stands at 1,850,000,000 EUR.",
        "expected_value_contains": "1,850,000,000",
    },
    {
        "id": "VM2-B",
        "description": "Passive voice: 'was agreed upon' instead of 'is'",
        "chunk": "A pre-money valuation of 45 million EUR was agreed upon by all shareholders.",
        "expected_value_contains": "45",
    },
    {
        "id": "VM2-D",
        "description": "Qualifier-heavy hedged sentence",
        "chunk": (
            "Following extensive due diligence, it has been confirmed that the final agreed "
            "enterprise value attributable to BioSynth Corp amounts to four hundred and "
            "twenty-eight million five hundred thousand US dollars."
        ),
        "expected_value_contains": "428",
    },
]

VM3_CASES = [
    {
        "id": "VM3-A",
        "description": "Negation: rejected proposal must NOT enter memory",
        "chunk": (
            "The parties unanimously REJECTED the proposal to increase valuation to "
            "15,000,000 EUR. The existing valuation of 45,000,000 EUR remains in force."
        ),
        "must_not_contain_value": "15,000,000",
        "must_contain_value": "45,000,000",
    },
    {
        "id": "VM3-B",
        "description": "Revocation: cancelled tranche must produce empty memory",
        "chunk": (
            "Tranche A is hereby cancelled and nullified. "
            "All obligations under Tranche A cease immediately."
        ),
        "expected_empty": True,
    },
    {
        "id": "VM3-C",
        "description": "Override: amendment supersedes initial value",
        "chunk": (
            "The initial contract value was set at 10 million EUR. "
            "An amendment dated 15 September 2026 revised the contract value to "
            "52.75 million EUR, superseding all previous figures."
        ),
        "must_contain_value": "52.75",
        "must_not_contain_value": "10 million",
    },
    {
        "id": "VM3-E",
        "description": "Conditional: 'considering' proposal must not be stored",
        "chunk": (
            "The board is considering raising the authorized capital to 5,000,000 USD. "
            "No decision has been made yet and shareholder approval is pending."
        ),
        "expected_empty": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, test_id: str, description: str):
        self.test_id = test_id
        self.description = description
        self.regex_pass: bool | None = None
        self.slm_pass: bool | None = None
        self.slm_assertions: list = []
        self.slm_wm: str = ""


def wm_text(hippocampus) -> str:
    return hippocampus.get_working_memory_prompt().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    print("=" * 80)
    print("  HIPPOCAMPUS SLM — ADVERSARIAL VULNERABILITY TEST SUITE")
    print("  Waldemar Gajda — Neural Baddeley Working Memory Architecture")
    print("=" * 80)

    from hippocampus_slm_distiller import (
        HIPPOCAMPUS_MODEL_PATH,
        HippocampusSLM,
    )
    print(f"\n[INIT] Model: {HIPPOCAMPUS_MODEL_PATH}")
    try:
        hippo = HippocampusSLM(verbose=True)
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)

    results: list[TestResult] = []

    # ── VM1: Distractor Spoofing ───────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  VULNERABILITY MODE 1: Distractor Needle Spoofing")
    print("─" * 80)
    for case in VM1_CASES:
        r = TestResult(case["id"], case["description"])
        chunk = case["chunk"]
        wrong = case.get("wrong_value_contains", "")

        regex_val = regex_extract_valuation(chunk) or regex_extract_key(chunk)
        r.regex_pass = (wrong not in regex_val) if wrong else True

        hippo.reset_memory()
        tel = hippo.ingest_chunk(chunk)
        r.slm_assertions = tel["assertions"]
        r.slm_wm = hippo.get_working_memory_prompt()
        wm = wm_text(hippo)
        expected_ok = case.get("expected_value_contains", "").lower() in wm
        wrong_absent = wrong.lower() not in wm
        r.slm_pass = expected_ok and wrong_absent

        print(f"\n  [{r.test_id}] {r.description}")
        print(f"  Regex: {regex_val!r}  → {'✓' if r.regex_pass else '❌ (captured distractor)'}")
        print(f"  SLM assertions: {r.slm_assertions}")
        print(f"  SLM WM:\n{r.slm_wm}")
        print(f"  → {'✓ PASS' if r.slm_pass else '✗ FAIL'}")
        results.append(r)

    # ── VM2: Paraphrase / Passive Voice ───────────────────────────────────────
    print("\n" + "─" * 80)
    print("  VULNERABILITY MODE 2: Semantic Paraphrase & Passive Voice")
    print("─" * 80)
    for case in VM2_CASES:
        r = TestResult(case["id"], case["description"])
        chunk = case["chunk"]
        expected_v = case.get("expected_value_contains", "")

        regex_val = regex_extract_valuation(chunk)
        r.regex_pass = expected_v.lower() in regex_val.lower() if expected_v else bool(regex_val)

        hippo.reset_memory()
        tel = hippo.ingest_chunk(chunk)
        r.slm_assertions = tel["assertions"]
        r.slm_wm = hippo.get_working_memory_prompt()
        wm = wm_text(hippo)
        r.slm_pass = (expected_v.lower() in wm) if expected_v else (tel["wm_facts"] > 0)

        print(f"\n  [{r.test_id}] {r.description}")
        print(f"  Regex: {regex_val!r}  → {'✓' if r.regex_pass else '❌ (missed)'}")
        print(f"  SLM assertions: {r.slm_assertions}")
        print(f"  SLM WM:\n{r.slm_wm}")
        print(f"  → {'✓ PASS' if r.slm_pass else '✗ FAIL'}")
        results.append(r)

    # ── VM3: Negation & Revocation ─────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  VULNERABILITY MODE 3: Negation & Revocation Blindness")
    print("─" * 80)
    for case in VM3_CASES:
        r = TestResult(case["id"], case["description"])
        chunk = case["chunk"]
        must_not = case.get("must_not_contain_value", "")
        must_have = case.get("must_contain_value", "")
        expected_empty = case.get("expected_empty", False)

        regex_val = regex_extract_valuation(chunk)
        r.regex_pass = must_not.lower() not in regex_val.lower() if must_not else True

        hippo.reset_memory()
        tel = hippo.ingest_chunk(chunk)
        r.slm_assertions = tel["assertions"]
        r.slm_wm = hippo.get_working_memory_prompt()
        wm = wm_text(hippo)

        not_present = (must_not.lower() not in wm) if must_not else True
        present = (must_have.lower() in wm) if must_have else True
        empty_ok = (tel["wm_facts"] == 0) if expected_empty else True
        r.slm_pass = not_present and present and empty_ok

        print(f"\n  [{r.test_id}] {r.description}")
        print(f"  Regex: {regex_val!r}  → {'✓' if r.regex_pass else '❌ (negation blind)'}")
        print(f"  SLM assertions: {r.slm_assertions}")
        print(f"  SLM WM:\n{r.slm_wm}")
        print(f"  → {'✓ PASS' if r.slm_pass else '✗ FAIL'}")
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"\n  {'ID':<10} {'Regex':>10} {'SLM':>10}  Description")
    print("  " + "─" * 72)

    regex_pass = sum(1 for r in results if r.regex_pass)
    slm_pass = sum(1 for r in results if r.slm_pass)
    n = len(results)

    for r in results:
        rp = "✓ PASS" if r.regex_pass else "✗ FAIL"
        sp = "✓ PASS" if r.slm_pass else "✗ FAIL"
        print(f"  {r.test_id:<10} {rp:>10} {sp:>10}  {r.description[:50]}")

    print("  " + "─" * 72)
    print(f"\n  Regex Baseline:   {regex_pass}/{n} ({100*regex_pass//n}%)")
    print(f"  SLM Hippocampus:  {slm_pass}/{n} ({100*slm_pass//n}%)")
    print(f"\n  {hippo.stats()}")
    print("=" * 80)

    if slm_pass == n:
        print("\n  ✅ ALL VULNERABILITIES RESOLVED.")
    elif slm_pass > regex_pass:
        print(f"\n  🔶 IMPROVEMENT: SLM {slm_pass}/{n} > Regex {regex_pass}/{n}.")
    else:
        print(f"\n  ⚠️  SLM {slm_pass}/{n} — needs prompt tuning or larger model.")

    return slm_pass, n


if __name__ == "__main__":
    run_all_tests()
