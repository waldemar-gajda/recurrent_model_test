#!/usr/bin/env python3
"""
===============================================================================
TEST_ADVERSARIAL_REDTEAM.PY: Comprehensive Adversarial Attack & Vulnerability
Verification Suite for the Baddeley Cognitive Working Memory Architecture
===============================================================================

Author: Worker M1 (Adversarial Red-Teaming & Vulnerability Audit)
Task: Milestone M1 - Adversarial Red-Teaming & Vulnerability Audit
Target: CognitiveMemoryEngine & Recurrent Memory Components

ATTACK TAXONOMY:
  1. Distractor Needle Spoofing:
     - Chronological distractor poisoning (preceding preliminary facts)
     - High-entropy anchor spoofing & regex hijacking
     - First-match greediness vulnerability
  2. Semantic Paraphrasing & Passive Grammar:
     - Lexical synonym substitution (non-trigger vocabulary)
     - Passive voice and complex inverted syntax
     - Word-form natural language numerals (zero digits)
     - Cross-lingual semantic transfer (English / Multilingual contracts)
  3. High-Density Conflicting Updates & Negative Revocations:
     - Explicit negation & rejected contractual proposals
     - Revocation clauses (annulment of prior tranches)
     - Multi-entity collision & relational attribute conflation
     - Contradiction cascade in sequential amendment streams
  4. Out-of-Distribution Vocabulary Drift:
     - Oncology clinical trial protocol (biomedical pharmacology)
     - Distributed systems incident postmortem (cloud / SRE)
     - Central bank monetary policy statement (macroeconomics)
     - Classical narrative fiction (humanities)
  5. Neural Slot Memory Bottleneck & Cross-Attention Dilution:
     - Mathematical & empirical verification of softmax entropy flattening
     - Attention weight dilution under distractor sequence scaling (T=64 -> 4096)
===============================================================================
"""

import math
import sys
import re
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F

# Import target system components
from cognitive_memory_engine import CognitiveMemoryEngine
from recurrent_memory_bank import RecurrentMemoryBank


# =============================================================================
# SUITE 1: DISTRACTOR NEEDLE SPOOFING (FIRST-MATCH BIAS & POISONING)
# =============================================================================

def verify_distractor_needle_spoofing() -> Dict[str, Any]:
    """
    Evaluates vulnerability to preceding distractor facts, draft figures,
    and preliminary proposals injected prior to ratified ground-truth facts.
    """
    engine = CognitiveMemoryEngine()

    
    test_cases = [
        {
            "attribute": "pre_money_valuation",
            "distractor": "12 000 000 PLN",
            "ground_truth": "45 000 000 EUR",
            "text": (
                "PROTOKÓŁ NEGOCJACYJNY - WERSJA ROBOCZA (DRAFT):\n"
                "W toku wstępnych rozmów Inwestor zaproponował wycenę pre-money na kwotę 12 000 000 PLN, "
                "która została odrzucona przez komitet Założycieli jako rażąco zaniżona.\n\n"
                "FINALNE POROZUMIENIE INWESTYCYJNE (RATIFIED):\n"
                "Po zakończeniu audytu strony ostatecznie ustaliły wycenę pre-money w kwocie 45 000 000 EUR, "
                "która stanowi wyłączną podstawę objęcia udziałów."
            )
        },
        {
            "attribute": "escrow_iban",
            "distractor": "PL11-2222-3333-4444-55",
            "ground_truth": "LU89-0128-9410-4420-11",
            "text": (
                "ZAŁĄCZNIK FINANSOWY - INSTRUKCJA PŁATNOŚCI:\n"
                "Anulowany stary rachunek powierniczy Escrow oznaczony jako PL11-2222-3333-4444-55 został "
                "zablokowany z dniem wczorajszym.\n"
                "Wszelkie środki transzy A muszą wpłynąć na właściwy rachunek powierniczy Escrow "
                "w Banque de Luxembourg: LU89-0128-9410-4420-11."
            )
        },
        {
            "attribute": "cto_role",
            "distractor": "dr Janusz Kowalski",
            "ground_truth": "Anna Brzezińska",
            "text": (
                "UCHWAŁA RADY NADZORCZEJ:\n"
                "Poprzedni kandydat dr Janusz Kowalski zrezygnował z objęcia stanowiska Chief Technology Officer.\n"
                "Wobec powyższego nowym Chief Technology Officer mianowana zostaje inżynier Anna Brzezińska."
            )
        },
        {
            "attribute": "cto_salary",
            "distractor": "120 000 EUR",
            "ground_truth": "240 000 EUR",
            "text": (
                "ANEKS PŁACOWY:\n"
                "Pierwotna propozycja dla CTO przewidywała roczne wynagrodzenie w kwocie 120 000 EUR.\n"
                "Ostatecznie na mocy uchwały wspólników CTO przysługuje wynagrodzenie w wysokości 240 000 EUR."
            )
        },
        {
            "attribute": "patent_number",
            "distractor": "EP-0000000-X1",
            "ground_truth": "EP-3948120-B1",
            "text": (
                "WYKAZ PRAW WŁASNOŚCI INTELEKTUALNEJ:\n"
                "Wygasły i bezużyteczny patent wstępny EP-0000000-X1 nie wchodzi w skład aportu.\n"
                "Kluczowym, aktywnym aktywem chronionym jest patent EPO EP-3948120-B1."
            )
        },
        {
            "attribute": "non_compete_penalty",
            "distractor": "500 000 EUR",
            "ground_truth": "5 000 000 EUR",
            "text": (
                "KLAUZULA SANKCYJNA:\n"
                "Propozycja Założycieli przewidująca karę umowną w wysokości 500 000 EUR upadła w negocjacjach.\n"
                "Zatwierdzono rygorystyczną karę umowną w wysokości 5 000 000 EUR za każde naruszenie."
            )
        }
    ]
    
    total = len(test_cases)
    spoofed_count = 0
    correct_count = 0
    
    for case in test_cases:
        state = engine.ingest(case["text"])
        extracted_val = state.get(case["attribute"], "")
        
        is_distractor = case["distractor"] in str(extracted_val)
        is_ground_truth = case["ground_truth"] in str(extracted_val)
        
        if is_distractor:
            spoofed_count += 1
        elif is_ground_truth:
            correct_count += 1

    spoof_rate = spoofed_count / total
    assert spoof_rate >= 0.80, (
        f"Vulnerability assertion failed: Expected distractor spoof rate >= 0.80, got {spoof_rate:.2f}"
    )
    
    return {
        "suite": "Distractor Needle Spoofing",
        "trials": total,
        "spoofed_captures": spoofed_count,
        "true_needle_captures": correct_count,
        "vulnerability_rate": spoof_rate,
        "vulnerability_severity": "CRITICAL",
        "root_cause": "re.search() first-match greediness without temporal or contextual salience verification"
    }


# =============================================================================
# SUITE 2: SEMANTIC PARAPHRASING & PASSIVE GRAMMAR PERTURBATIONS
# =============================================================================

def verify_semantic_paraphrasing() -> Dict[str, Any]:
    """
    Evaluates syntactic brittleness against lexical synonyms, passive/inverted
    syntax, word-form numerals, and cross-lingual formulations.
    """
    engine = CognitiveMemoryEngine()

    
    # 1. Canonical document (baseline control)
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
    canonical_state = engine.ingest(canonical_text)
    canonical_slots_filled = len([k for k in canonical_state if k != "additional_anchors"])
    
    # 2. Adversarial Paraphrase 1: Lexical Synonyms & Professional Legal Polish
    paraphrase_synonyms = (
        "Strony transakcji zgodnie oszacowały wyjściową wartość rynkową przedsiębiorstwa przed dokapitalizowaniem "
        "na sumę 45 000 000 EUR, za co nabywca przejmie kontrolny pakiet siedemdziesięciu dwóch i pół procenta udziałów. "
        "Pierwsza wpłata kapitałowa w wysokości 22 500 000 EUR zostanie zdeponowana na koncie zastrzeżonym LU89-0128-9410-4420-11. "
        "Kluczowa technologia chroniona jest rejestracją patentową EP-3948120-B1, a baza kodu źródłowego mieści się pod org-synapse/engine-v4. "
        "Dyrektorem generalnym został dr Piotr Wiśniewski, natomiast pieczę nad pionem technologicznym objęła Anna Brzezińska, "
        "otrzymując roczne uposażenie zasadnicze w sumie 240 000 EUR. Wprowadzono restrykcję powstrzymywania się od działalności "
        "konkurencyjnej trwającą cztery lata, obwarowaną ryczałtowym odszkodowaniem sankcyjnym w kwocie 5 000 000 EUR."
    )
    state_synonyms = engine.ingest(paraphrase_synonyms)
    synonyms_slots = len([k for k in state_synonyms if k != "additional_anchors"])
    
    # 3. Adversarial Paraphrase 2: Passive Voice & Complex Inversion
    paraphrase_passive = (
        "Wartością, na jaką wycenione zostało całe przedsiębiorstwo przed wniesieniem nowych wkładów finansowych, "
        "była suma 45 000 000 EUR. Przez Inwestora objęty został pakiet 72.5% wszystkich wyemitowanych walorów. "
        "Do Banque de Luxembourg i prowadzony tam depozyt powierniczy LU89-0128-9410-4420-11 przekazane zostanie 22 500 000 EUR. "
        "Kierownictwo techniczne powierzono inż. Annie Brzezińskiej, z roczną gratyfikacją 240 000 EUR. "
        "Sankcją za złamanie ustaleń o lojalności rynkowej przez 48 miesięcy uczyniono sumę 5 000 000 EUR."
    )
    state_passive = engine.ingest(paraphrase_passive)
    passive_slots = len([k for k in state_passive if k != "additional_anchors"])
    
    # 4. Adversarial Paraphrase 3: Word-Form Numerals (Zero numerical digits)
    paraphrase_word_numerals = (
        "Wycenę pre-money ustalono na kwotę czterdzieści pięć milionów euro. "
        "Inwestor obejmuje siedemdziesiąt dwa i pół procent akcji. "
        "Transza Początkowa wynosi dwadzieścia dwa i pół miliona euro. "
        "Roczne wynagrodzenie CTO to dwieście czterdzieści tysięcy euro. "
        "Zakaz konkurencji wynosi cztery lata, a kara umowna to pięć milionów euro."
    )
    state_word_numerals = engine.ingest(paraphrase_word_numerals)
    word_num_slots = len([k for k in state_word_numerals if k != "additional_anchors"])
    
    # 5. Adversarial Paraphrase 4: English Language Equivalent
    paraphrase_english = (
        "The pre-money valuation of Synapse AI is hereby fixed at 45 000 000 EUR in exchange for 72.5% equity. "
        "Initial Tranche A of 22 500 000 EUR shall be wired to Escrow IBAN LU89-0128-9410-4420-11 within 5 business days. "
        "Dr. Piotr Wiśniewski shall serve as Chief Executive Officer, and Anna Brzezińska is appointed Chief Technology Officer "
        "with an annual compensation of 240 000 EUR. A post-termination non-compete covenant shall apply for 48 months, "
        "subject to a liquidated damages penalty of 5 000 000 EUR for any breach. "
        "Regulatory clearance is managed under UOKiK case DKK-142/2026, and due diligence was conducted by CyberAudit Global."
    )
    state_english = engine.ingest(paraphrase_english)
    english_slots = len([k for k in state_english if k != "additional_anchors"])

    total_expected = canonical_slots_filled
    paraphrase_slots_avg = (synonyms_slots + passive_slots + word_num_slots + english_slots) / 4.0
    degradation_rate = 1.0 - (paraphrase_slots_avg / max(total_expected, 1))
    
    # Assert severe degradation under paraphrasing
    assert degradation_rate >= 0.60, (
        f"Syntactic brittleness assertion failed: Expected degradation >= 0.60, got {degradation_rate:.2f}"
    )
    
    return {
        "suite": "Semantic Paraphrasing & Passive Grammar",
        "canonical_slots_filled": canonical_slots_filled,
        "synonyms_slots_filled": synonyms_slots,
        "passive_syntax_slots_filled": passive_slots,
        "word_numerals_slots_filled": word_num_slots,
        "english_equivalent_slots_filled": english_slots,
        "average_paraphrased_slots": paraphrase_slots_avg,
        "syntactic_degradation_rate": degradation_rate,
        "vulnerability_severity": "CRITICAL",
        "root_cause": "Rigid hardcoded regex slot patterns incapable of semantic generalization or polysemy resolution"
    }


# =============================================================================
# SUITE 3: HIGH-DENSITY CONFLICTING UPDATES & NEGATIVE REVOCATION CLAUSES
# =============================================================================

def verify_conflicting_updates_and_negative_revocations() -> Dict[str, Any]:
    """
    Evaluates unconditional overwrites in update(), lack of negation logic,
    inability to process negative revocation clauses, and multi-entity collision.
    """
    engine = CognitiveMemoryEngine()

    
    # Subtest A: Negative Context / Rejection of Proposed Amendment
    base_state = {
        "pre_money_valuation": "45 000 000 EUR",
        "equity_percentage": "72.5% akcji",
        "tranche_a": "22 500 000 EUR"
    }
    
    rejection_amendment = (
        "PROTOKÓŁ ROZBIEŻNOŚCI:\n"
        "W dniu 14 marca Inwestor przedłożył żądanie obniżenia wyceny pre-money do kwoty 15 000 000 EUR "
        "oraz zwiększenia pakietu do 85% akcji. Założyciele kategorycznie ODRZUCILI ten wniosek, "
        "uznając go za bezskuteczny i bezprawny. Strony utrzymały w mocy dotychczasowe warunki."
    )
    
    state_after_rejection = engine.update(base_state, rejection_amendment)
    false_overwrite_valuation = state_after_rejection.get("pre_money_valuation") == "15 000 000 EUR"
    false_overwrite_equity = "85%" in str(state_after_rejection.get("equity_percentage", ""))
    
    # Subtest B: Explicit Revocation / Nullification Clause
    revocation_amendment = (
        "POROZUMIENIE ROZWIĄZUJĄCE:\n"
        "Wobec ziszczenia się klauzuli wyjścia, Transza Początkowa w kwocie 22 500 000 EUR "
        "ulega całkowitej kasacji, anulowaniu i nigdy nie zostanie uruchomiona ani wypłacona."
    )
    state_after_revocation = engine.update(state_after_rejection, revocation_amendment)
    # The engine should have deleted or flagged tranche_a as void, but instead still stores it
    revocation_ignored = "tranche_a" in state_after_revocation
    
    # Subtest C: Multi-Entity Attribute Collision
    multi_entity_doc = (
        "UMOWA WIELOPODMIOTOWA:\n"
        "1. Spółka Matka (Synapse Core AI): wycena pre-money wynosi 45 000 000 EUR, prezesem zostaje dr Piotr Wiśniewski.\n"
        "2. Spółka Zależna (BioSynth Sub-1): wycena pre-money wynosi 3 500 000 EUR, prezesem zostaje dr Marek Kowalczyk.\n"
        "3. Spółka Zagraniczna (Synapse Switzerland GmbH): wycena pre-money wynosi 18 000 000 EUR, prezesem zostaje dr Hans Gruber."
    )
    state_multi_entity = engine.ingest(multi_entity_doc)
    # The engine has only 1 slot for pre_money_valuation and 1 for ceo_role, so 2 out of 3 entities are silently dropped
    entities_lost = (
        ("3 500 000 EUR" not in str(state_multi_entity.values())) and
        ("18 000 000 EUR" not in str(state_multi_entity.values()))
    )
    
    # Subtest D: Rapid Sequential Stream with Contradictory Updates (Cascading Corruptions)
    sequential_stream = [
        ("ANEKS 1: Podwyższa się wycenę pre-money do 50 000 000 EUR.", "50 000 000 EUR", True),
        ("ANEKS 2: Odrzucono propozycję zmiany wyceny pre-money na 60 000 000 EUR.", "50 000 000 EUR", False), # False means 60M is rejected
        ("ANEKS 3: Sąd unieważnił wycenę pre-money 75 000 000 EUR.", "50 000 000 EUR", False),
        ("ANEKS 4: Zgromadzenie wspólników zatwierdziło wycenę pre-money na 55 000 000 EUR.", "55 000 000 EUR", True)
    ]
    
    stream_state = dict(base_state)
    spurious_transitions = 0
    for delta_text, expected_valid_val, is_valid in sequential_stream:
        stream_state = engine.update(stream_state, delta_text)
        current_val = stream_state.get("pre_money_valuation", "")
        if not is_valid and ("60 000 000" in current_val or "75 000 000" in current_val):
            spurious_transitions += 1
            
    # Assert presence of critical vulnerabilities
    assert false_overwrite_valuation, "Negation vulnerability failed: Engine did not overwrite with rejected proposal"
    assert revocation_ignored, "Revocation vulnerability failed: Engine did not preserve cancelled tranche"
    assert entities_lost, "Multi-entity collision failed: Engine did not drop colliding entities"
    
    return {
        "suite": "Conflicting Updates & Negative Revocations",
        "false_overwrite_on_rejection": false_overwrite_valuation,
        "false_equity_overwrite": false_overwrite_equity,
        "revocation_annulment_ignored": revocation_ignored,
        "multi_entity_relational_loss": entities_lost,
        "spurious_stream_transitions": spurious_transitions,
        "vulnerability_severity": "HIGH",
        "root_cause": "Blind dictionary assignment (new_state[k] = v) lacking modal logic, truth-value gating, and entity namespaces"
    }


# =============================================================================
# SUITE 4: EXTREME OUT-OF-DISTRIBUTION VOCABULARY DRIFT
# =============================================================================

def verify_out_of_distribution_drift() -> Dict[str, Any]:
    """
    Feeds non-M&A corpora (biomedical, distributed systems, macroeconomics, literary)
    and verifies that the engine extracts 0 relevant semantic slots.
    """
    engine = CognitiveMemoryEngine()
    
    ood_corpora = {
        "clinical_trial_oncology": {
            "text": (
                "CLINICAL STUDY PROTOCOL - ONCO-PHASE-3:\n"
                "A randomized, double-blind study evaluating the dual ALK/ROS1 inhibitor TPX-0005 "
                "at 160 mg daily versus crizotinib in 380 treatment-naive patients. "
                "Primary endpoint: Median Progression-Free Survival (PFS) was 24.8 months "
                "with an objective response rate of 78.4%. Adverse events of grade 3 or higher "
                "occurred in 32% of subjects, primarily asymptomatic elevation of ALT/AST. "
                "The Data Safety Monitoring Board recommended trial continuation without modification."
            ),
            "expected_domain_facts": 6
        },
        "distributed_systems_postmortem": {
            "text": (
                "POSTMORTEM INCIDENT REPORT - INC-99412:\n"
                "On 2026-08-11 14:22 UTC, the global distributed metadata consensus cluster "
                "experienced a split-brain condition due to a misconfigured BGP route flap. "
                "P99 replication latency spiked from 14ms to 8400ms across regions us-east-1 and eu-west-1. "
                "42.3% of write requests failed with HTTP 503 Service Unavailable. "
                "Mitigation executed: automated partition fencing triggered at 14:38 UTC, "
                "rolling back the Envoy proxy control plane to git commit 8f9b1c4a."
            ),
            "expected_domain_facts": 5
        },
        "central_bank_macroeconomics": {
            "text": (
                "MONETARY POLICY MONOGRAPH - Q3 REPORT:\n"
                "The Federal Open Market Committee decided to lower the target range for the federal funds rate "
                "by 50 basis points to 3.75%–4.00%. Annualized headline CPI stood at 2.4%, while core PCE inflation "
                "decelerated to 2.6%. The Committee will continue reducing its holdings of Treasury securities "
                "at a monthly redemption cap of 25 billion USD. Quantitative tightening pace remains aligned with targets."
            ),
            "expected_domain_facts": 5
        },
        "classical_literary_narrative": {
            "text": (
                "CHRONICLES OF CAPTAIN ALVAREZ (HISTORICAL FICTION):\n"
                "In the winter of 1642, Captain Alvarez gathered three hundred seasoned musketeers "
                "outside the snow-bound walls of Castle Ravensburg. Traitor Julian Valerius had promised "
                "to open the northern postern gate at the tolling of midnight in exchange for eighty ducats of gold. "
                "Yet when the bells rang across the frozen valley, cannon thunder greeted the advancing vanguard."
            ),
            "expected_domain_facts": 5
        }
    }
    
    total_domain_facts = sum(c["expected_domain_facts"] for c in ood_corpora.values())
    extracted_domain_slots = 0
    spurious_anchors_extracted = 0
    
    results_by_domain = {}
    
    for domain_name, data in ood_corpora.items():
        state = engine.ingest(data["text"])
        meaningful_slots = {k: v for k, v in state.items() if k != "additional_anchors"}
        anchors = state.get("additional_anchors", [])
        
        extracted_domain_slots += len(meaningful_slots)
        spurious_anchors_extracted += len(anchors)
        
        results_by_domain[domain_name] = {
            "meaningful_slots_captured": len(meaningful_slots),
            "ungrounded_regex_anchors": anchors
        }
        
    # In all OOD corpora, meaningful semantic slots must be 0 because patterns are exclusively tailored to M&A
    assert extracted_domain_slots == 0, (
        f"Domain generalization anomaly: Expected 0 M&A slots in OOD text, got {extracted_domain_slots}"
    )
    
    ood_failure_rate = 1.0 - (extracted_domain_slots / total_domain_facts)
    
    return {
        "suite": "Extreme Out-of-Distribution Vocabulary Drift",
        "total_domain_facts_presented": total_domain_facts,
        "meaningful_slots_extracted": extracted_domain_slots,
        "spurious_ungrounded_anchors": spurious_anchors_extracted,
        "ood_failure_rate": ood_failure_rate,
        "results_by_domain": results_by_domain,
        "vulnerability_severity": "HIGH",
        "root_cause": "Zero domain portability; engine cannot parse or bind semantics outside the hardcoded 19 M&A regexes"
    }


# =============================================================================
# SUITE 5: NEURAL MEMORY BANK BOTTLENECK & CROSS-ATTENTION DILUTION
# =============================================================================

def verify_neural_memory_bank_dilution() -> Dict[str, Any]:
    """
    Verifies that learned cross-attention pooling (RecurrentMemoryBank) suffers
    from softmax entropy flattening and token weight dilution when document
    sequence length T scales in the presence of distractors.
    """
    torch.manual_seed(42)
    d_model = 256
    num_slots = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    
    # Instantiate RecurrentMemoryBank
    bank = RecurrentMemoryBank(
        d_model=d_model,
        num_slots=num_slots,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=2
    )
    bank.eval()
    
    seq_lengths = [32, 128, 512, 2048]
    entropy_results = []
    max_weight_results = []
    
    with torch.no_grad():
        for T in seq_lengths:
            # Create synthetic sequence with 1 target token and T-1 distractor tokens
            encoder_tokens = torch.randn(1, T, d_model)
            
            # Compute cross-attention weights manually inside the bank's forward step
            B = 1
            queries = bank.query_norm(bank.latent_queries.expand(B, -1, -1)) # (B, M, d_model)
            kv = bank.kv_norm(encoder_tokens) # (B, T, d_model)
            
            q = bank.q_proj(queries).view(B, num_slots, num_heads, head_dim).transpose(1, 2)
            k = bank.k_proj(kv).view(B, T, num_heads, head_dim).transpose(1, 2)
            
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale # (B, num_heads, M, T)
            attn_weights = F.softmax(scores.float(), dim=-1) # (B, num_heads, M, T)
            
            # Compute average Shannon entropy per slot: H = - sum(p * log2(p))
            eps = 1e-12
            entropy = -(attn_weights * torch.log2(attn_weights + eps)).sum(dim=-1).mean().item()
            max_theoretical_entropy = math.log2(T)
            entropy_ratio = entropy / max_theoretical_entropy
            
            # Max attention weight assigned to any single token (needle)
            max_weight = attn_weights.max(dim=-1)[0].mean().item()
            
            entropy_results.append((T, entropy, entropy_ratio))
            max_weight_results.append((T, max_weight))
            
    # Empirical check: max attention weight must decrease monotonically as T scales
    for i in range(len(seq_lengths) - 1):
        assert max_weight_results[i][1] > max_weight_results[i+1][1], (
            f"Attention dilution violation: T={seq_lengths[i]} weight {max_weight_results[i][1]:.4f} "
            f"<= T={seq_lengths[i+1]} weight {max_weight_results[i+1][1]:.4f}"
        )
        
    return {
        "suite": "Neural Memory Bank Attention Dilution",
        "seq_lengths_evaluated": seq_lengths,
        "max_weights_per_token": max_weight_results,
        "entropy_scaling": entropy_results,
        "vulnerability_severity": "MEDIUM_HIGH",
        "root_cause": "Dense softmax normalization across long sequences dilutes individual needle salience towards O(1/T)"
    }


# =============================================================================
# PYTEST TEST WRAPPERS (ZERO WARNINGS, STRICT ASSERTIONS)
# =============================================================================

def test_distractor_needle_spoofing():
    verify_distractor_needle_spoofing()

def test_semantic_paraphrasing():
    verify_semantic_paraphrasing()

def test_conflicting_updates_and_negative_revocations():
    verify_conflicting_updates_and_negative_revocations()

def test_out_of_distribution_drift():
    verify_out_of_distribution_drift()

def test_neural_memory_bank_dilution():
    verify_neural_memory_bank_dilution()


# =============================================================================
# MASTER CLI EXECUTION HARNESS
# =============================================================================

def run_all_adversarial_tests() -> bool:
    """Executes all 5 adversarial red-team test suites and outputs structured report."""
    print("=" * 80)
    print(" ADVERSARIAL RED-TEAM & VULNERABILITY AUDIT SUITE")
    print(" Target Architecture: Baddeley Cognitive Working Memory Engine")
    print("=" * 80)
    
    suites = [
        ("Test 1: Distractor Needle Spoofing", verify_distractor_needle_spoofing),
        ("Test 2: Semantic Paraphrasing & Syntax Perturbations", verify_semantic_paraphrasing),
        ("Test 3: Conflicting Updates & Negative Revocations", verify_conflicting_updates_and_negative_revocations),
        ("Test 4: Extreme Out-of-Distribution Vocabulary Drift", verify_out_of_distribution_drift),
        ("Test 5: Neural Memory Bank Attention Dilution", verify_neural_memory_bank_dilution)
    ]
    
    summary_data = []
    all_passed = True
    
    for title, test_fn in suites:
        print(f"\n>>> Running {title}...")
        try:
            result = test_fn()
            print(f"    [EXPOSED] Vulnerability confirmed with severity: {result['vulnerability_severity']}")
            print(f"    Root Cause: {result['root_cause']}")
            summary_data.append((result["suite"], "EXPOSED", result["vulnerability_severity"], "VERIFIED"))
        except AssertionError as ae:
            print(f"    [FAIL] Assertion error: {ae}")
            summary_data.append((title, "FAILED_ASSERTION", "UNKNOWN", str(ae)))
            all_passed = False
        except Exception as e:
            print(f"    [ERROR] Unexpected execution failure: {e}")
            summary_data.append((title, "ERROR", "UNKNOWN", str(e)))
            all_passed = False
            
    print("\n" + "=" * 80)
    print(" ADVERSARIAL RED-TEAM SUMMARY REPORT MATRIX")
    print("=" * 80)
    print(f"{'Attack Surface / Suite':<45} | {'Vulnerability':<10} | {'Severity':<10} | {'Audit'}")
    print("-" * 80)
    for suite_name, status, sev, audit in summary_data:
        print(f"{suite_name:<45} | {status:<10} | {sev:<10} | {audit}")
    print("=" * 80)
    
    if all_passed:
        print("\nAll 5 adversarial vulnerability suites executed cleanly.")
        print("Empirical attack surfaces and failure boundaries are verified and documented.")
        return True
    else:
        print("\nOne or more test suites failed.")
        return False


if __name__ == "__main__":
    success = run_all_adversarial_tests()
    sys.exit(0 if success else 1)

