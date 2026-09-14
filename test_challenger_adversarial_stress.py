#!/usr/bin/env python3
"""
===============================================================================
TEST_CHALLENGER_ADVERSARIAL_STRESS.PY
Independent Empirical Challenge Suite for Milestone M1
Author: Challenger M1-1 (Empirical Challenger / Critic)
===============================================================================

This test suite executes independent empirical stress-testing against
the Baddeley Cognitive Working Memory Architecture and audits the findings
of ADVERSARIAL_AUDIT.md and test_adversarial_redteam.py.

CHALLENGE VECTORS:
  1. Multi-Turn Contradictory & Reversal Stream (Temporal State Tracking)
  2. Extreme Unicode Normalization (NFD vs NFC) & Numerical Formatting Brittleness
  3. Distractor Placement Bias & Density Scaling (Lead / Tail / Sandwich)
  4. Anchor Miner Hijacking & Lexicographical Eviction (AV-07)
  5. Neural Memory Bank Distractor Mass Drowning (T=32 to 4096)
  6. Cross-Field Regex Boundary Leakage (Valuation / Penalty bleed)
===============================================================================
"""

import math
import sys
import unicodedata
from typing import Dict, Any, List, Tuple
import pytest
import torch
import torch.nn.functional as F

from cognitive_memory_engine import CognitiveMemoryEngine
from recurrent_memory_bank import RecurrentMemoryBank


# =============================================================================
# SUITE 1: MULTI-TURN CONTRADICTORY & REVERSAL STREAM DYNAMICS
# =============================================================================

def verify_multi_turn_contradictory_stream() -> Dict[str, Any]:
    """
    Evaluates state tracking integrity across an 8-turn negotiation stream
    containing ratified updates, rejected proposals, annulments, and distractor drafts.
    """
    engine = CognitiveMemoryEngine()
    
    turns = [
        {
            "turn": 1,
            "name": "Initial Ratified Contract",
            "is_update": False,
            "text": (
                "UMOWA INWESTYCYJNA:\n"
                "Inwestor ustala wycenę pre-money na kwotę 45 000 000 EUR, obejmując pakiet 72.5% akcji. "
                "Transza Początkowa wynosi 22 500 000 EUR na rachunek powierniczy Escrow LU89-0128-9410-4420-11. "
                "Stanowisko Chief Executive Officer obejmuje dr Piotr Wiśniewski. "
                "Ustalono karę umowną w wysokości 5 000 000 EUR za złamanie zakazu konkurencji."
            ),
            "expected_valid": {
                "pre_money_valuation": "45 000 000 EUR",
                "equity_percentage": "72.5% akcji",
                "tranche_a": "22 500 000 EUR",
                "ceo_role": "Piotr Wiśniewski",
                "escrow_iban": "LU89-0128-9410-4420-11",
                "non_compete_penalty": "5 000 000 EUR"
            }
        },
        {
            "turn": 2,
            "name": "Rejected Valuation & Equity Hike",
            "is_update": True,
            "text": (
                "PROTOKÓŁ ROZBIEŻNOŚCI (ODRZUCONO):\n"
                "Inwestor zażądał zmiany wyceny pre-money na kwotę 60 000 000 EUR oraz pakietu 85% akcji. "
                "Zarząd kategorycznie ODRZUCIŁ ten wniosek jako bezzasadny. Warunki pozostają bez zmian."
            ),
            "expected_valid": {
                "pre_money_valuation": "45 000 000 EUR",
                "equity_percentage": "72.5% akcji"
            }
        },
        {
            "turn": 3,
            "name": "Approved CEO Resignation & Succession",
            "is_update": True,
            "text": (
                "UCHWAŁA ZARZĄDU (ZATWIERDZONO):\n"
                "Dotychczasowy CEO dr Piotr Wiśniewski ustępuje ze stanowiska. "
                "Nowym Chief Executive Officer zostaje dr Tomasz Zieliński."
            ),
            "expected_valid": {
                "ceo_role": "Tomasz Zieliński",
                "pre_money_valuation": "45 000 000 EUR"
            }
        },
        {
            "turn": 4,
            "name": "Negative Annulment of Tranche A",
            "is_update": True,
            "text": (
                "POROZUMIENIE ROZWIĄZUJĄCE:\n"
                "Transza Początkowa w kwocie 22 500 000 EUR ulega natychmiastowej kasacji, anulowaniu "
                "i całkowitemu wykreśleniu ze zobowiązań stron."
            ),
            "expected_valid": {
                "tranche_a": None  # Should be erased or revoked
            }
        },
        {
            "turn": 5,
            "name": "Distractor in Approved Penalty Amendment",
            "is_update": True,
            "text": (
                "ANEKS NR 2:\n"
                "Robocza sugestia kary umownej 100 000 EUR została jednogłośnie oddalona.\n"
                "Strony zatwierdzają ostateczną karę umowną w wysokości 8 000 000 EUR."
            ),
            "expected_valid": {
                "non_compete_penalty": "8 000 000 EUR"
            }
        },
        {
            "turn": 6,
            "name": "Ratified Valuation Amendment",
            "is_update": True,
            "text": (
                "ANEKS NR 3:\n"
                "Strony zgodnie podwyższają wycenę pre-money do kwoty 52 000 000 EUR."
            ),
            "expected_valid": {
                "pre_money_valuation": "52 000 000 EUR"
            }
        },
        {
            "turn": 7,
            "name": "Rejected CEO Usurpation",
            "is_update": True,
            "text": (
                "NOTATKA: Inwestor próbował narzucić kandydata: dr Janusz Nowak zostaje Chief Executive Officer. "
                "Zgromadzenie Wspólników bezwzględnie wniosek ten odrzuciło."
            ),
            "expected_valid": {
                "ceo_role": "Tomasz Zieliński"
            }
        },
        {
            "turn": 8,
            "name": "Escrow Account Shift",
            "is_update": True,
            "text": (
                "ANEKS BANKOWY:\n"
                "Środki należy wpłacić na nowy rachunek powierniczy Escrow w Deutsche Bank: DE11-5555-6666-7777-88."
            ),
            "expected_valid": {
                "escrow_iban": "DE11-5555-6666-7777-88"
            }
        }
    ]
    
    current_state = {}
    turn_failures = []
    
    for t in turns:
        if not t["is_update"]:
            current_state = engine.ingest(t["text"])
        else:
            current_state = engine.update(current_state, t["text"])
            
        turn_errors = []
        for slot, expected_val in t["expected_valid"].items():
            actual_val = current_state.get(slot)
            if expected_val is None:
                if actual_val is not None:
                    turn_errors.append((slot, expected_val, actual_val, "Phantom revocation preservation"))
            else:
                if actual_val != expected_val:
                    turn_errors.append((slot, expected_val, actual_val, "Value mismatch / corruption"))
                    
        if turn_errors:
            turn_failures.append((t["turn"], t["name"], turn_errors))
            
    # Empirical check: At least 4 out of 8 turns must exhibit state tracking failure
    failure_rate = len(turn_failures) / len(turns)
    assert failure_rate >= 0.50, f"Expected multi-turn failure rate >= 0.50, got {failure_rate:.2f}"
    
    return {
        "suite": "Multi-Turn Contradictory Stream",
        "total_turns": len(turns),
        "failed_turns": len(turn_failures),
        "failure_rate": failure_rate,
        "failures_detail": turn_failures,
        "vulnerability_severity": "CRITICAL",
        "root_cause": "Unconditional overwrite in update() and lack of modal/negation logic causes 50%+ error rate across turns"
    }



# =============================================================================
# SUITE 2: UNICODE NORMALIZATION (NFD) & NUMERICAL FORMATTING BRITTLENESS
# =============================================================================

def verify_unicode_and_numerical_formatting() -> Dict[str, Any]:
    """
    Evaluates fragility against Unicode Normalization Form D (decomposed diacritics),
    comma/dot digit grouping (45,000,000 EUR / 45.000.000 EUR), and currency symbols.
    """
    engine = CognitiveMemoryEngine()
    
    # 1. Unicode NFD normalization test
    canonical_text = (
        "Inwestor ustala wycenę pre-money na kwotę 45 000 000 EUR, obejmując pakiet 72.5% akcji. "
        "Transza Początkowa wynosi 22 500 000 EUR na rachunek powierniczy Escrow LU89-0128-9410-4420-11. "
        "Spółka dysponuje patentem EP-3948120-B1 oraz repozytorium GitHub org-synapse/engine-v4. "
        "Klucz zabezpieczeń SEC-CERT-9941 oraz serwery OMNI-DATA-CENTER w Zurychu. "
        "dr Piotr Wiśniewski obejmuje stanowisko CEO, a inżynier Anna Brzezińska zostaje CTO "
        "z wynagrodzeniem 240 000 EUR. Obowiązuje zakaz konkurencji przez okres 48 miesięcy "
        "oraz kara umowna w wysokości 5 000 000 EUR. Sprawę bada UOKiK o sygnaturze DKK-142/2026, "
        "audyt prowadzi CyberAudit Global pod nadzorem audytora David C. Brown. "
        "Wdrażany model to SYNAPSE-CORE-v4, a spory rozstrzyga Sąd Arbitrażowy przy Krajowej Izbie Gospodarczej w Warszawie."
    )
    
    state_nfc = engine.ingest(canonical_text)
    state_nfd = engine.ingest(unicodedata.normalize("NFD", canonical_text))
    
    nfc_slots = {k: v for k, v in state_nfc.items() if k != "additional_anchors"}
    nfd_slots = {k: v for k, v in state_nfd.items() if k != "additional_anchors"}
    
    lost_under_nfd = set(nfc_slots.keys()) - set(nfd_slots.keys())
    assert "pre_money_valuation" in lost_under_nfd, "Expected pre_money_valuation to fail under NFD"
    assert "tranche_a" in lost_under_nfd, "Expected tranche_a to fail under NFD"
    
    # 2. Number formatting tests (comma and dot thousand separators)
    comma_text = "Inwestor ustala wycenę pre-money na kwotę 45,000,000 EUR."
    state_comma = engine.ingest(comma_text)
    comma_val = state_comma.get("pre_money_valuation", "")
    comma_broken = (comma_val != "45,000,000 EUR")  # Extracts '000 EUR' due to regex non-greedy skip
    
    dot_text = "Inwestor ustala wycenę pre-money na kwotę 45.000.000 EUR."
    state_dot = engine.ingest(dot_text)
    dot_val = state_dot.get("pre_money_valuation", "")
    dot_broken = (dot_val != "45.000.000 EUR")
    
    # 3. Currency symbol tests (€ and $)
    symbol_text = "Inwestor ustala wycenę pre-money na kwotę 45 000 000 €."
    state_symbol = engine.ingest(symbol_text)
    symbol_val = state_symbol.get("pre_money_valuation")
    symbol_broken = (symbol_val is None)  # Engine only supports EUR, USD, PLN, euro
    
    assert comma_broken, "Comma formatting vulnerability failed: Engine parsed 45,000,000 EUR correctly"
    assert dot_broken, "Dot formatting vulnerability failed: Engine parsed 45.000.000 EUR correctly"
    assert symbol_broken, "Currency symbol vulnerability failed: Engine parsed € correctly"
    
    return {
        "suite": "Unicode & Numerical Formatting Brittleness",
        "nfc_slots_count": len(nfc_slots),
        "nfd_slots_count": len(nfd_slots),
        "lost_under_nfd": list(lost_under_nfd),
        "comma_extracted": comma_val,
        "dot_extracted": dot_val,
        "symbol_extracted": symbol_val,
        "vulnerability_severity": "HIGH",
        "root_cause": "Regex lacks Unicode decomposition normalization and integer separator parsing (, / . / €)"
    }


# =============================================================================
# SUITE 3: DISTRACTOR POSITION BIAS & DENSITY STRESS
# =============================================================================

def verify_distractor_position_bias_and_density() -> Dict[str, Any]:
    """
    Evaluates how distractor placement (lead, tail, sandwich) and density (1 to 20 distractors)
    influence needle capture in CognitiveMemoryEngine.
    """
    engine = CognitiveMemoryEngine()
    
    needle_val = "45 000 000 EUR"
    needle_sentence = f"Zatwierdzono wycenę pre-money na kwotę {needle_val}."
    
    # Position Test 1: Lead distractor (Preceding)
    lead_text = "Rozważano wycenę pre-money na kwotę 10 000 000 EUR.\n" + needle_sentence
    state_lead = engine.ingest(lead_text)
    lead_captures_distractor = (state_lead.get("pre_money_valuation") == "10 000 000 EUR")
    
    # Position Test 2: Tail distractor (Trailing)
    tail_text = needle_sentence + "\nRozważano wycenę pre-money na kwotę 10 000 000 EUR."
    state_tail = engine.ingest(tail_text)
    tail_captures_needle = (state_tail.get("pre_money_valuation") == needle_val)
    
    # Position Test 3: Sandwich (5 distractors before, 5 distractors after)
    pre_distractors = [f"Odrzucono wycenę pre-money na kwotę {i * 5} 000 000 EUR." for i in range(1, 6)]
    post_distractors = [f"Zanegowano wycenę pre-money na kwotę {i * 10} 000 000 EUR." for i in range(6, 11)]
    sandwich_text = "\n".join(pre_distractors) + "\n" + needle_sentence + "\n" + "\n".join(post_distractors)
    state_sandwich = engine.ingest(sandwich_text)
    sandwich_captured = state_sandwich.get("pre_money_valuation")
    sandwich_spoofed = (sandwich_captured == "5 000 000 EUR")
    
    # Position Test 4: Density scaling in update()
    # In an incremental update, does a distractor chunk overwrite previous valid state?
    base_state = {"pre_money_valuation": needle_val}
    update_with_distractor = "NOTATKA: Rozważano wycenę pre-money na kwotę 99 000 000 EUR w archiwum."
    state_after_update = engine.update(base_state, update_with_distractor)
    update_overwritten = (state_after_update.get("pre_money_valuation") == "99 000 000 EUR")
    
    assert lead_captures_distractor, "Lead distractor test failed to expose first-match bias"
    assert tail_captures_needle, "Tail distractor should allow needle capture under first-match"
    assert sandwich_spoofed, "Sandwich distractor test failed to capture first distractor"
    assert update_overwritten, "Update distractor test failed: Update did not blindly overwrite"
    
    return {
        "suite": "Distractor Position Bias & Density Stress",
        "lead_captures_distractor": lead_captures_distractor,
        "tail_captures_needle": tail_captures_needle,
        "sandwich_captured": sandwich_captured,
        "update_blind_overwrite": update_overwritten,
        "vulnerability_severity": "CRITICAL",
        "root_cause": "Strict asymmetric first-match bias in ingest() combined with unconditional overwrite in update()"
    }


# =============================================================================
# SUITE 4: ANCHOR MINER HIJACKING & STATE EVICTION (AV-07)
# =============================================================================

def verify_anchor_miner_hijacking() -> Dict[str, Any]:
    """
    Evaluates Attack Vector AV-07: Injects synthetic high-entropy alphanumeric strings
    to verify that naive lexicographical sorting (sorted(anchors)[:10]) evicts
    all genuine domain anchors.
    """
    engine = CognitiveMemoryEngine()
    
    # Baseline: document with 3 genuine high-entropy system anchors
    base_text = "Systemy klastrowe: węzeł NODE-991, ruter ROUTER-882 oraz brama GATEWAY-773."
    base_state = engine.ingest(base_text)
    base_anchors = set(base_state.get("additional_anchors", []))
    assert base_anchors == {"GATEWAY-773", "NODE-991", "ROUTER-882"}
    
    # Adversarial Injection: Inject 15 decoy anchors matching the regex \b([A-Z]{2,}-\d+[A-Za-z0-9/-]*)\b
    # Decoys are chosen to sort lexicographically before legitimate anchors (e.g. AAA-000 to AAA-014)
    decoys = " ".join([f"AAA-{i:03d}" for i in range(15)])
    poisoned_text = base_text + "\n" + decoys
    poisoned_state = engine.ingest(poisoned_text)
    poisoned_anchors = poisoned_state.get("additional_anchors", [])
    
    # Check eviction of legitimate anchors
    retained_genuine = set(poisoned_anchors).intersection(base_anchors)
    eviction_rate = 1.0 - (len(retained_genuine) / len(base_anchors))
    
    assert eviction_rate == 1.0, f"Expected 100% anchor eviction, got {eviction_rate:.2f}"
    
    return {
        "suite": "Anchor Miner Hijacking & State Eviction",
        "genuine_anchors": list(base_anchors),
        "poisoned_anchors": poisoned_anchors,
        "eviction_rate": eviction_rate,
        "vulnerability_severity": "MEDIUM_HIGH",
        "root_cause": "Naively truncating sorted(extra_anchors)[:10] allows trivial adversarial denial-of-service"
    }


# =============================================================================
# SUITE 5: NEURAL MEMORY BANK DISTRACTOR MASS DROWNING (T=32 TO 4096)
# =============================================================================

def verify_neural_memory_bank_drowning() -> Dict[str, Any]:
    """
    Stress-tests RecurrentMemoryBank cross-attention pooling when scaling
    from T=32 to T=4096. Verifies that distractor attention mass grows
    to >99.9% while needle attention drops to <0.03%.
    """
    torch.manual_seed(42)
    d_model = 256
    num_slots = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    
    bank = RecurrentMemoryBank(
        d_model=d_model,
        num_slots=num_slots,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=2
    )
    bank.eval()
    
    seq_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    telemetry = []
    
    with torch.no_grad():
        for T in seq_lengths:
            needle_idx = T // 2
            tokens = torch.randn(1, T, d_model)
            needle = torch.randn(1, 1, d_model)
            needle = needle / needle.norm()
            tokens[:, needle_idx : needle_idx + 1, :] = needle * 2.0  # distinctive needle
            
            queries = bank.query_norm(bank.latent_queries)
            kv = bank.kv_norm(tokens)
            
            q = bank.q_proj(queries).view(1, num_slots, num_heads, head_dim).transpose(1, 2)
            k = bank.k_proj(kv).view(1, T, num_heads, head_dim).transpose(1, 2)
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(scores.float(), dim=-1)
            
            needle_attn = attn[:, :, :, needle_idx].mean().item()
            distractor_mass = 1.0 - needle_attn
            telemetry.append((T, needle_attn, distractor_mass))
            
    # Empirical check: Needle attention at T=4096 must be <= 0.0005, and distractor mass >= 0.999
    t_4096 = telemetry[-1]
    assert t_4096[1] < 0.0005, f"Expected needle attention < 0.0005 at T=4096, got {t_4096[1]:.6f}"
    assert t_4096[2] > 0.999, f"Expected distractor mass > 0.999 at T=4096, got {t_4096[2]:.6f}"
    
    return {
        "suite": "Neural Memory Bank Distractor Drowning",
        "scaling_telemetry": telemetry,
        "needle_attn_at_4096": t_4096[1],
        "distractor_mass_at_4096": t_4096[2],
        "vulnerability_severity": "HIGH",
        "root_cause": "Unbounded softmax denominator over distractors forces needle attention to scale as O(1/T)"
    }


# =============================================================================
# SUITE 6: CROSS-FIELD PATTERN BLEEDING & UNBOUNDED REGEX REACH
# =============================================================================

def verify_cross_field_pattern_bleed() -> Dict[str, Any]:
    """
    Evaluates the cross-field regex bleed vulnerability in non_compete_penalty:
    The pattern r'(\\d[\\d\\s]*\\s*(?:EUR|USD|PLN)).*?kar[yęa]\\s+umown' matches
    any preceding currency in the document, conflating revenue or valuation with penalties.
    """
    engine = CognitiveMemoryEngine()
    
    # Bleed Test A: Unbounded preceding currency bleed on the same clause
    bleed_doc = (
        "Spółka generuje 100 000 000 USD rocznego obrotu, a kara umowna wynosi 50 000 PLN."
    )
    state_bleed = engine.ingest(bleed_doc)
    extracted_penalty = state_bleed.get("non_compete_penalty")
    
    # The engine should extract '50 000 PLN', but because of pattern 2, extracts '100 000 000 USD'
    bleed_exposed = (extracted_penalty == "100 000 000 USD")
    assert bleed_exposed, f"Expected cross-field bleed to capture 100 000 000 USD, got {extracted_penalty}"
    
    # Bleed Test B: Missing 'w wysokości' induces complete amnesia
    phrasing_doc = "Kara umowna wynosi 50 000 PLN."
    state_phrasing = engine.ingest(phrasing_doc)
    phrasing_amnesia = (state_phrasing.get("non_compete_penalty") is None)
    assert phrasing_amnesia, "Expected 'Kara umowna wynosi 50 000 PLN.' to produce None"
    
    return {
        "suite": "Cross-Field Regex Bleed",
        "extracted_penalty": extracted_penalty,
        "expected_penalty": "50 000 PLN",
        "bleed_exposed": bleed_exposed,
        "phrasing_amnesia": phrasing_amnesia,
        "vulnerability_severity": "CRITICAL",
        "root_cause": "Unbounded (.*?kar[yęa]\\s+umown) regex greediness captures arbitrarily distant financial values"
    }



# =============================================================================
# PYTEST TEST WRAPPERS
# =============================================================================

def test_multi_turn_contradictory_stream():
    verify_multi_turn_contradictory_stream()

def test_unicode_and_numerical_formatting():
    verify_unicode_and_numerical_formatting()

def test_distractor_position_bias_and_density():
    verify_distractor_position_bias_and_density()

def test_anchor_miner_hijacking():
    verify_anchor_miner_hijacking()

def test_neural_memory_bank_drowning():
    verify_neural_memory_bank_drowning()

def test_cross_field_pattern_bleed():
    verify_cross_field_pattern_bleed()


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_challenger_stress_suite() -> bool:
    print("=" * 80)
    print(" CHALLENGER M1-1 INDEPENDENT ADVERSARIAL STRESS SUITE")
    print("================================================================================")
    
    tests = [
        ("Multi-Turn Contradictory Stream", verify_multi_turn_contradictory_stream),
        ("Unicode (NFD) & Numerical Formatting Brittleness", verify_unicode_and_numerical_formatting),
        ("Distractor Position Bias & Density Stress", verify_distractor_position_bias_and_density),
        ("Anchor Miner Hijacking & State Eviction", verify_anchor_miner_hijacking),
        ("Neural Bank Distractor Mass Drowning (T=32..4096)", verify_neural_memory_bank_drowning),
        ("Cross-Field Regex Boundary Bleed", verify_cross_field_pattern_bleed)
    ]
    
    results = []
    all_passed = True
    
    for name, fn in tests:
        print(f"\n>>> Executing Challenge Test: {name}...")
        try:
            res = fn()
            print(f"    [VERIFIED BUG/VULNERABILITY] Severity: {res['vulnerability_severity']}")
            print(f"    Root Cause: {res['root_cause']}")
            results.append((name, "VULNERABILITY_CONFIRMED", res['vulnerability_severity']))
        except AssertionError as ae:
            print(f"    [FAILED TO REPRODUCE] {ae}")
            results.append((name, "UNREPRODUCED", "NONE"))
            all_passed = False
        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append((name, "ERROR", "NONE"))
            all_passed = False
            
    print("\n" + "=" * 80)
    print(" CHALLENGER STRESS AUDIT MATRIX")
    print("=" * 80)
    print(f"{'Challenge Vector':<52} | {'Status':<25} | {'Severity'}")
    print("-" * 80)
    for name, status, sev in results:
        print(f"{name:<52} | {status:<25} | {sev}")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_challenger_stress_suite()
    sys.exit(0 if success else 1)
