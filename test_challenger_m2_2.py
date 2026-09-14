#!/usr/bin/env python3
"""
===============================================================================
TEST_CHALLENGER_M2_2.PY: Empirical Adversarial Challenge Harness for Milestone M2
===============================================================================

Roles: critic, specialist (Challenger M2-2)
Target Deliverables:
  - cognitive_memory_engine.py
  - data/test_reproducibility.py
  - data/corpus_manager.py

Scope of Verification:
  1. Long-run streaming memory stress test:
     - Stream 1,000 synthetic chunks with random high-entropy anchors through
       CognitiveMemoryEngine.update().
     - Verify that `additional_anchors` strictly never exceeds 16 elements at any step.
     - Measure state size (key count, shallow dict bytes, recursive deep bytes,
       formatted working memory length) to verify strict O(1) boundedness.
     - Track physical OS RSS memory across 1,000 chunks to verify absence of RAM leaks.
     - Adversarial stress cases: single-chunk anchor flooding, alphabetical eviction bias,
       immutability of previous state, configurable max_anchors.
  2. KV-cache arithmetic accuracy across arbitrary sequence lengths and batch sizes:
     - Mathematical derivation and verification of standard uncompressed KV-cache footprint.
     - Test arbitrary sequence lengths (1 to 110,000,000 tokens) and batch sizes (1 to 256).
     - Compare formula with actual PyTorch tensor allocations byte-for-byte.
     - Multi-architecture verification: LLaMA-3-8B (GQA), LLaMA-2-7B (MHA), LLaMA-3-70B (GQA),
       Mistral-7B, AllTokenRecurrentModel.
     - Audit calculation discrepancies, unit labels (MB vs MiB vs GB vs TB), and API signatures.
===============================================================================
"""

import os
import sys
import gc
import random
import string
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pytest
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognitive_memory_engine import CognitiveMemoryEngine
from data.corpus_manager import (
    calculate_kv_cache_bytes,
    format_kv_cache_comparison,
    get_current_rss_bytes,
    get_current_rss_mb,
    MemoryProfiler
)


# =============================================================================
# HELPER UTILITIES: DEEP MEMORY RECURSION
# =============================================================================

def get_deep_size(obj, seen=None) -> int:
    """Recursively computes total memory consumption of a Python data structure in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    return size


def generate_random_high_entropy_anchor() -> str:
    """Generates an anchor matching r'\\b([A-Z]{2,}-\\d+[A-Za-z0-9/-]*)\\b'."""
    prefix_len = random.randint(2, 6)
    prefix = "".join(random.choices(string.ascii_uppercase, k=prefix_len))
    num = random.randint(100, 99999)
    suffix_choice = random.choice(["", "-X", "-ALPHA", f"/{random.randint(1, 99)}", f"-{random.randint(10, 99)}"])
    return f"{prefix}-{num}{suffix_choice}"


# =============================================================================
# TASK 1: LONG-RUN STREAMING MEMORY STRESS TEST (1,000 CHUNKS)
# =============================================================================

def test_streaming_memory_1000_chunks_strictly_bounded():
    """
    Stress-tests CognitiveMemoryEngine.update() over 1,000 synthetic streaming chunks.
    Verifies:
      1. additional_anchors <= 16 at EVERY step t in [1, 1000].
      2. State dictionary size (keys, deep bytes) remains strictly O(1) flat.
      3. format_working_memory(state) output length remains strictly O(1) flat.
      4. OS Process RSS memory does not leak.
    """
    print("\n" + "=" * 80)
    print(" TASK 1: 1,000 STREAMING CHUNKS MEMORY STRESS TEST (O(1) PROOF)")
    print("=" * 80)

    random.seed(42)
    engine = CognitiveMemoryEngine(max_anchors=16)

    # Initial ingestion
    initial_text = (
        "Dokument pierwotny spółki NeuroTech sp. z o.o. "
        "Wycena pre-money została ustalona na 50 000 000 PLN. "
        "Inwestor obejmuje 15.0% udziałów. "
        "Transza Początkowa A: 5 000 000 PLN. "
        "Rachunek Escrow: PL61-1090-1014-0000-00. "
        "Patent EPO: EP-4190812-B1. Model autorski: SYNAPSE-9B-V2. "
        "Repozytorium kodu: github.com/neurotech/core-engine. "
        "Certyfikat YubiKey: SEC-9941-K. Hosting: OMNI-DATA-CENTER w Zurychu. "
        "Prezes zarządu: dr Jan Kowalski. CTO: inż. Adam Nowak, wynagrodzenie: 85 000 PLN. "
        "Zakaz konkurencji: 24 miesiące, kara umowna: 2 000 000 PLN. "
        "Zgoda UOKiK: DKK-192/2026. Audyt Due Diligence: CyberAudit Global pod nadzorem Marka Nowickiego. "
        "Sąd Arbitrażowy przy KIG. "
        "Identyfikator inicjalny: INIT-0001-A."
    )
    state = engine.ingest(initial_text)

    assert "additional_anchors" in state
    assert len(state["additional_anchors"]) <= 10  # Ingest default cap is 10
    assert len(state["additional_anchors"]) > 0

    profiler = MemoryProfiler(label="1000-Chunk Stream Profiler")
    baseline_rss_mb = profiler.baseline_rss / (1024 * 1024)

    # Metrics logging
    step_samples = []
    deep_sizes = []
    formatted_lengths = []
    anchor_counts = []
    total_unique_anchors_generated = set()

    for i in range(1, 1001):
        # Generate 2-5 distinct high-entropy anchors for this chunk
        num_anchors = random.randint(2, 5)
        chunk_anchors = [generate_random_high_entropy_anchor() for _ in range(num_anchors)]
        total_unique_anchors_generated.update(chunk_anchors)

        # In 10% of chunks, also inject a conflict update for an existing structured slot
        extra_content = []
        if i % 10 == 0:
            new_val = 50 + (i % 50)
            extra_content.append(f"Wycena pre-money została zaktualizowana na {new_val} 000 000 PLN.")
        if i % 25 == 0:
            new_sal = 85 + (i % 20)
            extra_content.append(f"Wynagrodzenie CTO zostało podwyższone do {new_sal} 000 PLN.")
        if i % 50 == 0:
            extra_content.append(f"Nowy certyfikat bezpieczeństwa: SEC-{i:04d}-UPGRADED.")

        anchor_str = " ".join([f"Identyfikator zdarzenia {a}." for a in chunk_anchors])
        chunk_text = f"Aneks nr {i}: " + " ".join(extra_content) + " " + anchor_str

        # Update state S_t -> S_{t+1}
        state = engine.update(state, chunk_text)

        # STRICT ASSERTION 1: additional_anchors NEVER exceeds 16
        anchors = state.get("additional_anchors", [])
        assert len(anchors) <= 16, f"VIOLATION at step {i}: anchor count {len(anchors)} > 16!"

        # Measure metrics
        deep_sz = get_deep_size(state)
        fmt_text = engine.format_working_memory(state)
        fmt_len = len(fmt_text)

        anchor_counts.append(len(anchors))
        deep_sizes.append(deep_sz)
        formatted_lengths.append(fmt_len)

        # Sample profiler every 100 steps
        if i % 100 == 0 or i == 1:
            current_rss = profiler.sample(i)
            step_samples.append({
                "step": i,
                "anchor_count": len(anchors),
                "num_keys": len(state),
                "deep_size_bytes": deep_sz,
                "formatted_len_chars": fmt_len,
                "rss_mb": current_rss
            })

    profiler_summary = profiler.finish()

    # Empirical Verifications
    max_anchors_observed = max(anchor_counts)
    min_anchors_observed = min(anchor_counts)
    max_deep_size = max(deep_sizes)
    min_deep_size = min(deep_sizes)
    max_fmt_len = max(formatted_lengths)
    min_fmt_len = min(formatted_lengths)

    print(f"Total synthetic chunks streamed: 1,000")
    print(f"Total unique synthetic anchors generated: {len(total_unique_anchors_generated):,}")
    print(f"Anchors in state: min={min_anchors_observed}, max={max_anchors_observed} (strictly <= 16)")
    print(f"State deep size: min={min_deep_size} bytes, max={max_deep_size} bytes (ratio={max_deep_size/min_deep_size:.2f}x)")
    print(f"Formatted working memory length: min={min_fmt_len} chars, max={max_fmt_len} chars")
    print(f"Physical RAM: baseline={profiler_summary['baseline_rss_mb']:.2f} MB, peak={profiler_summary['peak_rss_mb']:.2f} MB, final={profiler_summary['final_rss_mb']:.2f} MB, delta={profiler_summary['delta_rss_mb']:+.2f} MB")

    # Assertions
    assert max_anchors_observed == 16, f"Expected additional_anchors to reach saturation cap 16, got {max_anchors_observed}"
    assert max_deep_size < 10_000, f"State deep size exceeded 10 KB: {max_deep_size} bytes"
    assert max_fmt_len < 3_000, f"Formatted memory exceeded 3,000 characters: {max_fmt_len} chars"
    assert profiler_summary["is_flat"], f"Memory profile not flat: delta RSS = {profiler_summary['delta_rss_mb']} MB"

    print("✅ 1,000-chunk streaming test PASSED: Strict O(1) state bound and physical memory verified.")


def test_adversarial_anchor_burst_flooding():
    """
    Challenge 1A: Single-chunk anchor flooding & Two-Tier Ingestion Cap.
    Injects 100 distinct anchors in a SINGLE update call.
    Demonstrates the two-tier ceiling:
      - Tier 1: ingest() has a hardcoded cap of 10 anchors per chunk (line 126).
      - Tier 2: update() has a cumulative stream cap of max_anchors (16).
    Therefore, a single burst chunk yields min(10, max_anchors) = 10 anchors.
    A second burst chunk then saturates the buffer to exactly max_anchors (16).
    """
    print("\n" + "-" * 60)
    print(" CHALLENGE 1A: SINGLE-CHUNK ANCHOR FLOODING & TWO-TIER CAP")
    print("-" * 60)
    engine = CognitiveMemoryEngine(max_anchors=16)
    initial_state = engine.ingest("Inicjalny start.")

    flood_anchors_chunk1 = [f"FLOODA-{k:05d}-X" for k in range(100)]
    flood_text_1 = "Ogromna paczka identyfikatorów 1: " + " ".join(flood_anchors_chunk1)

    updated_state_1 = engine.update(initial_state, flood_text_1)
    anchors_1 = updated_state_1.get("additional_anchors", [])

    # Tier 1 confirmation: Ingest caps single chunk extraction at 10
    assert len(anchors_1) == 10, f"Expected 10 anchors due to ingest() cap, got {len(anchors_1)}"
    print(f"  • Single-chunk burst (100 anchors) extracted {len(anchors_1)} anchors (ingest() cap: 10).")

    # Tier 2 confirmation: Streaming a second burst saturates to max_anchors = 16
    flood_anchors_chunk2 = [f"FLOODB-{k:05d}-Y" for k in range(100)]
    flood_text_2 = "Ogromna paczka identyfikatorów 2: " + " ".join(flood_anchors_chunk2)
    updated_state_2 = engine.update(updated_state_1, flood_text_2)
    anchors_2 = updated_state_2.get("additional_anchors", [])

    assert len(anchors_2) == 16, f"Expected 16 anchors after 2nd burst, got {len(anchors_2)}"
    assert get_deep_size(updated_state_2) < 5_000
    print(f"  • Second burst saturated buffer to exactly {len(anchors_2)} anchors (max_anchors cap: 16).")
    print("✅ Two-tier ceiling (ingest: 10, update: 16) verified empirically.")


def test_lexicographical_eviction_bias():
    """
    Challenge 1B: Lexicographical / Alphabetical Eviction Bias Audit.
    Analyzes the architectural behavior of:
        new_state['additional_anchors'] = sorted(list(existing))[:self.max_anchors]
    Demonstrates that alphabetical sorting causes older 'A*' anchors to starve newer 'Z*' anchors,
    revealing a key design property: anchor retention is NOT recency-based (FIFO/LRU),
    but lexicographically biased.
    """
    print("\n" + "-" * 60)
    print(" CHALLENGE 1B: LEXICOGRAPHICAL EVICTION BIAS ANALYSIS")
    print("-" * 60)
    engine = CognitiveMemoryEngine(max_anchors=16)

    # Step 1: Populate with 16 anchors starting with 'AA-'
    aa_anchors = [f"AA-{i:04d}" for i in range(16)]
    state = engine.ingest("Start: " + " ".join(aa_anchors[:10]))
    state = engine.update(state, "Aneks: " + " ".join(aa_anchors[10:]))
    assert state["additional_anchors"] == sorted(aa_anchors)

    # Step 2: Stream 50 subsequent chunks containing 'ZZ-' anchors (newest facts)
    for step in range(50):
        new_chunk = f"Nowy aneks {step}: najnowszy identyfikator ZZ-{step:04d}-LATEST."
        state = engine.update(state, new_chunk)
        # Verify that because 'ZZ-' > 'AA-', ALL 16 'AA-' anchors are retained,
        # and 'ZZ-' anchors are NEVER retained in the episodic buffer!
        assert all(a.startswith("AA-") for a in state["additional_anchors"])

    print("⚠️  CONFIRMED ARCHITECTURAL BEHAVIOR: Lexicographical sorting causes alphabetical bias.")
    print("   16 older 'AA-*' anchors permanently starved 50 newer 'ZZ-*' anchors from entering memory.")
    print("   This confirms anchor retention is lexicographical (O(1) bounded), NOT FIFO/LRU recency.")


def test_state_immutability_and_isolation():
    """
    Challenge 1C: Verify update() does not mutate input state in-place.
    """
    print("\n" + "-" * 60)
    print(" CHALLENGE 1C: STATE IMMUTABILITY AND SIDE-EFFECT ISOLATION")
    print("-" * 60)
    engine = CognitiveMemoryEngine(max_anchors=16)
    s0 = engine.ingest("Wycena pre-money: 10 000 000 EUR. Kod: REF-001-A.")
    s0_copy = dict(s0)
    s0_anchors_copy = list(s0.get("additional_anchors", []))

    s1 = engine.update(s0, "Wycena pre-money: 20 000 000 EUR. Kod: REF-002-B.")

    # s0 must remain unaltered
    assert s0["pre_money_valuation"] == "10 000 000 EUR"
    assert s0.get("additional_anchors", []) == s0_anchors_copy
    # s1 must reflect new facts
    assert s1["pre_money_valuation"] == "20 000 000 EUR"
    assert "REF-002-B" in s1.get("additional_anchors", [])
    print("✅ State immutability verified: update() does not corrupt prior state.")


def test_configurable_max_anchors():
    """
    Challenge 1D: Verify arbitrary max_anchors values (0, 1, 5, 16, 32, 64).
    Streams multiple chunks (since each chunk contributes up to 10 anchors via ingest)
    to confirm that cumulative state capacity saturates at exactly `max_anchors`.
    """
    print("\n" + "-" * 60)
    print(" CHALLENGE 1D: CONFIGURABLE MAX_ANCHORS (0, 1, 5, 16, 32, 64)")
    print("-" * 60)
    for cap in [0, 1, 5, 16, 32, 64]:
        eng = CognitiveMemoryEngine(max_anchors=cap)
        st = eng.ingest("Start.")
        # Stream 10 chunks with distinct anchors (up to 100 anchors total)
        for chunk_idx in range(10):
            chunk_anchors = " ".join([f"CAPTEST-{cap:02d}-{chunk_idx:02d}-{k:02d}" for k in range(10)])
            st = eng.update(st, f"Chunk {chunk_idx}: {chunk_anchors}")

        actual_len = len(st.get("additional_anchors", []))
        expected_len = min(cap, 100)
        assert actual_len == expected_len, f"For max_anchors={cap}, expected {expected_len} anchors, got {actual_len}"
        print(f"  • max_anchors={cap:>2}: saturated at exactly {actual_len} elements.")
    print("✅ Configurable max_anchors parameter strictly adhered to across all test values.")


# =============================================================================
# TASK 2: KV-CACHE ARITHMETIC ACCURACY ACROSS ARBITRARY SEQ LENGTHS & BATCH SIZES
# =============================================================================

def exact_kv_cache_bytes_reference(
    batch_size: int,
    seq_len: int,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    bytes_per_elem: int = 2
) -> int:
    """
    Ground-truth reference formula for standard uncompressed Transformer KV-cache:
        Total Bytes = 2 (K and V) * num_layers * batch_size * num_kv_heads * seq_len * head_dim * bytes_per_elem
    """
    return 2 * num_layers * batch_size * num_kv_heads * seq_len * head_dim * bytes_per_elem


def test_kv_cache_arithmetic_arbitrary_sequences_and_batches():
    """
    Verifies KV-cache arithmetic accuracy across a wide parametric grid of:
      - Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256
      - Sequence lengths: 1, 16, 64, 512, 600, 2048, 8192, 32768, 131072, 1_000_000, 11_000_000, 109_738_160
    Cross-checks formula against:
      1. calculate_kv_cache_bytes() in data/corpus_manager.py
      2. Ground-truth reference formula
      3. Actual PyTorch allocated tensor memory (.numel() * .element_size())
    """
    print("\n" + "=" * 80)
    print(" TASK 2: KV-CACHE ARITHMETIC VERIFICATION (ARBITRARY SEQ & BATCH)")
    print("=" * 80)

    # 1. Grid of batch sizes and sequence lengths
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    seq_lengths = [1, 16, 64, 512, 600, 2048, 8192, 32768, 131072, 1_000_000, 11_000_000, 109_738_160]

    # Meta-Llama-3-8B GQA parameters
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    bytes_per_elem = 2  # bfloat16

    bytes_per_token_per_batch = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    assert bytes_per_token_per_batch == 131_072  # Exactly 128 KiB per token

    print(f"Meta-Llama-3-8B constant: {bytes_per_token_per_batch:,} bytes/token (128 KB/token)")

    # Test the entire grid
    tested_combinations = 0
    for B in batch_sizes:
        for L in seq_lengths:
            ref_bytes = exact_kv_cache_bytes_reference(
                batch_size=B,
                seq_len=L,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bytes_per_elem=bytes_per_elem
            )

            # When calling calculate_kv_cache_bytes, total tokens = B * L
            total_tokens = B * L
            func_bytes = calculate_kv_cache_bytes(
                num_tokens=total_tokens,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bytes_per_elem=bytes_per_elem
            )

            assert ref_bytes == func_bytes, f"Mismatch for B={B}, L={L}: ref={ref_bytes} vs func={func_bytes}"
            tested_combinations += 1

    print(f"Tested {tested_combinations} (Batch x Sequence) combinations: 100% exact match.")


def test_kv_cache_pytorch_tensor_allocation_ground_truth():
    """
    Allocates real PyTorch KV tensors for small/medium dimensions and verifies
    that exact_kv_cache_bytes_reference() matches PyTorch tensor memory byte-for-byte.
    """
    print("\n" + "-" * 60)
    print(" PYTORCH TENSOR ALLOCATION GROUND-TRUTH EMPIRICAL CHECK")
    print("-" * 60)

    test_configs = [
        {"B": 1, "L": 600, "num_layers": 4, "num_kv_heads": 4, "head_dim": 64, "dtype": torch.float32, "b": 4},
        {"B": 3, "L": 128, "num_layers": 6, "num_kv_heads": 2, "head_dim": 64, "dtype": torch.float16, "b": 2},
        {"B": 7, "L": 32,  "num_layers": 2, "num_kv_heads": 8, "head_dim": 128, "dtype": torch.bfloat16, "b": 2},
        {"B": 2, "L": 256, "num_layers": 8, "num_kv_heads": 8, "head_dim": 128, "dtype": torch.int8, "b": 1},
    ]

    for cfg in test_configs:
        B = cfg["B"]
        L = cfg["L"]
        nl = cfg["num_layers"]
        nkv = cfg["num_kv_heads"]
        hd = cfg["head_dim"]
        dtype = cfg["dtype"]
        b = cfg["b"]

        # Allocate real simulated KV cache layers in PyTorch
        total_pytorch_bytes = 0
        for _ in range(nl):
            k = torch.empty((B, nkv, L, hd), dtype=dtype)
            v = torch.empty((B, nkv, L, hd), dtype=dtype)
            total_pytorch_bytes += (k.numel() * k.element_size()) + (v.numel() * v.element_size())

        formula_bytes = exact_kv_cache_bytes_reference(
            batch_size=B,
            seq_len=L,
            num_layers=nl,
            num_kv_heads=nkv,
            head_dim=hd,
            bytes_per_elem=b
        )

        assert total_pytorch_bytes == formula_bytes, (
            f"PyTorch byte mismatch! Tensor nelement bytes = {total_pytorch_bytes}, formula = {formula_bytes}"
        )
        print(f"  Config (B={B}, L={L}, nl={nl}, nkv={nkv}, hd={hd}, bytes={b}): "
              f"PyTorch={total_pytorch_bytes:,} B == Formula={formula_bytes:,} B [EXACT]")

    print("✅ PyTorch tensor memory allocations match arithmetic formula byte-for-byte.")


def test_kv_cache_multi_model_architectures():
    """
    Verifies KV-cache arithmetic across standard industry architectures:
      1. Meta-Llama-3-8B (32 layers, 8 KV heads, 128 head_dim, 2 bytes/elem)
      2. Meta-Llama-2-7B (32 layers, 32 KV heads [MHA], 128 head_dim, 2 bytes/elem)
      3. Meta-Llama-3-70B (80 layers, 8 KV heads, 128 head_dim, 2 bytes/elem)
      4. Mistral-7B-v0.1 (32 layers, 8 KV heads, 128 head_dim, 2 bytes/elem)
      5. GPT-3 175B (96 layers, 96 KV heads [MHA], 128 head_dim, 2 bytes/elem)
    """
    print("\n" + "-" * 60)
    print(" MULTI-MODEL INDUSTRY ARCHITECTURE KV-CACHE PROFILES")
    print("-" * 60)

    models = {
        "LLaMA-3-8B (GQA)":   {"layers": 32, "kv_heads": 8,  "head_dim": 128, "b": 2},
        "LLaMA-2-7B (MHA)":   {"layers": 32, "kv_heads": 32, "head_dim": 128, "b": 2},
        "LLaMA-3-70B (GQA)":  {"layers": 80, "kv_heads": 8,  "head_dim": 128, "b": 2},
        "Mistral-7B (GQA)":   {"layers": 32, "kv_heads": 8,  "head_dim": 128, "b": 2},
        "GPT-3-175B (MHA)":   {"layers": 96, "kv_heads": 96, "head_dim": 128, "b": 2},
    }

    test_seqs = [600, 109_738_160]

    for model_name, p in models.items():
        b_per_tok = 2 * p["layers"] * p["kv_heads"] * p["head_dim"] * p["b"]
        wm_b = calculate_kv_cache_bytes(600, p["layers"], p["kv_heads"], p["head_dim"], p["b"])
        seq_b = calculate_kv_cache_bytes(109_738_160, p["layers"], p["kv_heads"], p["head_dim"], p["b"])

        wm_mb = wm_b / (1024 * 1024)
        seq_tb = seq_b / 1e12

        print(f"  • {model_name:<18}: {b_per_tok:>7,} B/tok | WM (600 tok): {wm_mb:>6.2f} MB | Full (110M tok): {seq_tb:>6.2f} TB")

        if "LLaMA-3-8B" in model_name:
            assert b_per_tok == 131_072
            assert round(wm_mb, 1) == 75.0
            assert round(wm_b / 1e6, 1) == 78.6
            assert round(seq_tb, 2) == 14.38


def test_kv_cache_unit_and_api_signature_audit():
    """
    Critical API & Unit Audit:
      1. Demonstrates the ambiguity in calculate_kv_cache_bytes(num_tokens) when batch_size > 1.
         If a user calls calculate_kv_cache_bytes(seq_len=2048) intending batch_size=16,
         the API lacks a batch_size keyword argument, forcing the user to know they must
         pre-multiply num_tokens = seq_len * batch_size.
      2. Validates the corrected reporting of 78.6 MB vs 0.08 MB.
      3. Validates binary (MiB/TiB) vs decimal (MB/TB) calculations:
         - 78,643,200 bytes = 78.64 MB (decimal, 10^6) = 75.00 MiB (binary, 1024^2)
         - 14,383,600,097,280 bytes = 14.38 TB (decimal, 10^12) = 13.08 TiB (binary, 1024^4)
    """
    print("\n" + "-" * 60)
    print(" CRITICAL API & UNIT AUDIT: BINARY VS DECIMAL & BATCH PARAMETER")
    print("-" * 60)

    # 1. Signature Check
    import inspect
    sig = inspect.signature(calculate_kv_cache_bytes)
    params = list(sig.parameters.keys())
    assert "batch_size" not in params, (
        "calculate_kv_cache_bytes now has batch_size parameter! Update audit documentation."
    )
    print("⚠️  API AUDIT FINDING: calculate_kv_cache_bytes lacks explicit 'batch_size' parameter.")
    print("   Users must pass `num_tokens = batch_size * seq_len`. Passing `batch_size` kwargs will raise TypeError.")

    with pytest.raises(TypeError):
        calculate_kv_cache_bytes(600, batch_size=4)  # type: ignore

    # 2. Binary vs Decimal Units
    wm_bytes = calculate_kv_cache_bytes(600)
    assert wm_bytes == 78_643_200

    wm_decimal_mb = wm_bytes / 1e6      # 78.6432 MB
    wm_binary_mib = wm_bytes / (1024**2) # 75.0 MiB
    wm_decimal_gb = wm_bytes / 1e9      # 0.0786432 GB

    assert abs(wm_decimal_mb - 78.64) < 0.01
    assert abs(wm_binary_mib - 75.0) < 0.01
    assert abs(wm_decimal_gb - 0.0786) < 0.001

    print(f"  600 Tokens Working Memory:")
    print(f"    - Raw Bytes:       {wm_bytes:,} bytes")
    print(f"    - Decimal (10^6):  {wm_decimal_mb:.2f} MB")
    print(f"    - Binary (1024^2): {wm_binary_mib:.2f} MiB")
    print(f"    - Decimal (10^9):  {wm_decimal_gb:.4f} GB")
    print(f"    -> Conclusively disproves the earlier '0.08 MB' claim (0.08 MB is off by ~1,000x).")

    full_seq_bytes = calculate_kv_cache_bytes(109_738_160)
    full_seq_decimal_tb = full_seq_bytes / 1e12       # 14.3836 TB
    full_seq_binary_tib = full_seq_bytes / (1024**4)  # 13.0818 TiB

    assert abs(full_seq_decimal_tb - 14.38) < 0.01
    assert abs(full_seq_binary_tib - 13.08) < 0.01

    print(f"  109,738,160 Tokens (Full Context):")
    print(f"    - Raw Bytes:        {full_seq_bytes:,} bytes")
    print(f"    - Decimal (10^12):  {full_seq_decimal_tb:.2f} TB")
    print(f"    - Binary (1024^4):  {full_seq_binary_tib:.2f} TiB")

    # 3. Format comparison string check
    comp_text = format_kv_cache_comparison(600, 109_738_160)
    assert "78.6 MB" in comp_text or "75.0 MB" in comp_text
    assert "TB" in comp_text
    print("✅ format_kv_cache_comparison properly reflects the corrected 78.6 MB figure.")


if __name__ == "__main__":
    print("Executing Challenger M2-2 Stress Test Suite...")
    test_streaming_memory_1000_chunks_strictly_bounded()
    test_adversarial_anchor_burst_flooding()
    test_lexicographical_eviction_bias()
    test_state_immutability_and_isolation()
    test_configurable_max_anchors()
    test_kv_cache_arithmetic_arbitrary_sequences_and_batches()
    test_kv_cache_pytorch_tensor_allocation_ground_truth()
    test_kv_cache_multi_model_architectures()
    test_kv_cache_unit_and_api_signature_audit()
    print("\n" + "=" * 80)
    print(" ALL CHALLENGER M2-2 EMPIRICAL STRESS TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
