#!/usr/bin/env python3
"""
Unit and integration tests for Milestone M2:
End-to-End Reproducibility, Sanitization, Physical OS RAM Profiling,
and Corrected Hardware Formulas.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cognitive_memory_engine import CognitiveMemoryEngine
from data.corpus_manager import (
    get_model_dir,
    get_corpus_dir,
    get_bible_path,
    get_canonical_books,
    generate_synthetic_canonical_corpus,
    generate_synthetic_bible,
    get_current_rss_bytes,
    get_current_rss_mb,
    MemoryProfiler,
    calculate_kv_cache_bytes,
    format_kv_cache_comparison
)


def test_path_portability_and_env_overrides():
    """Verify that paths default to relative project paths and respect env overrides."""
    default_m = get_model_dir()
    assert default_m == PROJECT_ROOT / "llama-3-8b-instruct"

    # Test override
    os.environ["LLAMA_MODEL_DIR"] = "/tmp/test_llama_path"
    try:
        assert get_model_dir() == Path("/tmp/test_llama_path").resolve()
    finally:
        del os.environ["LLAMA_MODEL_DIR"]


def test_corpus_discovery_and_bible_path():
    """Verify canonical Bible path discovery and book existence."""
    bible_path = get_bible_path()
    assert bible_path.is_file()
    assert bible_path.stat().st_size > 100_000

    with open(bible_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(2000)
    assert "### " in head


def test_canonical_books_no_zero_division():
    """Verify that get_canonical_books always returns >= 19 files, preventing ZeroDivisionError."""
    books = get_canonical_books()
    assert len(books) >= 19
    for b in books:
        assert b.is_file()
        assert b.stat().st_size > 300 * 1024


def test_synthetic_fallback_generation():
    """Verify isolated clean-room generation of synthetic corpus if disk is completely clean."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "subcorpus"
        generate_synthetic_canonical_corpus(test_dir, num_books=19, target_size_bytes=320 * 1024)
        files = list(test_dir.glob("*.txt"))
        assert len(files) == 19
        for f in files:
            assert f.stat().st_size >= 320 * 1024


def test_cognitive_memory_engine_bounded_anchors():
    """Verify that CognitiveMemoryEngine strictly caps additional_anchors to max_anchors (16) under infinite stream."""
    engine = CognitiveMemoryEngine(max_anchors=16)
    state = engine.ingest("Dokument startowy z kodem ID-INIT-001.")
    
    # Perform 150 consecutive streaming updates with distinct high-entropy anchors
    for i in range(150):
        delta = f"Aneks nr {i}: nowy identyfikator SEC-HASH-{i:04d} oraz transakcja REG-{i:04d}-X."
        state = engine.update(state, delta)
        assert len(state["additional_anchors"]) <= 16, f"Memory leak at step {i}: {len(state['additional_anchors'])} > 16"

    assert len(state["additional_anchors"]) == 16
    # Ingest cap must remain 10 for standard docs
    ingest_state = engine.ingest(" ".join([f"ANCHOR-{k:03d}" for k in range(30)]))
    assert len(ingest_state["additional_anchors"]) <= 10


def test_psutil_physical_ram_measurement():
    """Verify genuine OS RSS measurement and MemoryProfiler mechanics."""
    rss_bytes = get_current_rss_bytes()
    rss_mb = get_current_rss_mb()
    assert rss_bytes > 1_000_000  # At least 1 MB
    assert rss_mb > 1.0

    profiler = MemoryProfiler(label="Unit Test Profiler")
    profiler.sample(1)
    profiler.sample(2)
    summary = profiler.finish()

    assert summary["baseline_rss_mb"] > 0
    assert summary["peak_rss_mb"] >= summary["baseline_rss_mb"]
    assert "delta_rss_mb" in summary
    assert summary["is_flat"] is True


def test_hardware_kv_cache_calculation():
    """
    Verify exact LLaMA-3-8B GQA KV-cache calculations:
    1 token = 131,072 bytes (128 KB)
    600 tokens = 78,643,200 bytes = 78.6 MB (0.079 GB) [NOT 0.08 MB]
    109,738,160 tokens = 14,383.6 GB = 14.38 TB (or 14.04 TB decimal)
    """
    # 1 token
    tok1_bytes = calculate_kv_cache_bytes(1)
    assert tok1_bytes == 131072

    # 600 tokens (working memory buffer)
    wm_bytes = calculate_kv_cache_bytes(600)
    assert wm_bytes == 78_643_200
    wm_mb = wm_bytes / (1024 * 1024)
    wm_gb = wm_bytes / 1e9
    assert 74.9 < wm_mb < 75.1  # 75.0 MiB
    assert 0.078 < wm_gb < 0.079  # 0.0786 GB

    # Full 110M sequence
    seq110m_bytes = calculate_kv_cache_bytes(109_738_160)
    seq110m_tb = seq110m_bytes / 1e12
    assert 14.3 < seq110m_tb < 14.5  # 14.38 TB

    comp_text = format_kv_cache_comparison(600, 109_738_160)
    assert "78.6 MB" in comp_text or "75.0 MB" in comp_text
    assert "14." in comp_text
    assert "TB" in comp_text
