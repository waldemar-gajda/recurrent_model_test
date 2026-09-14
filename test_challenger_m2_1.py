#!/usr/bin/env python3
"""
===============================================================================
TEST_CHALLENGER_M2_1.PY: Empirical Adversarial Challenge Harness for Milestone M2
===============================================================================

Roles: critic, specialist (Challenger M2-1)
Target Artifacts:
  - data/corpus_manager.py
  - run_reproducible_benchmarks.py
  - cognitive_memory_engine.py

Scope of Verification:
  1. Clean-room environment execution:
     - Simulate an environment with NO /tmp/*.txt and NO data/corpus/ files.
     - Verify data/corpus_manager.py creates clean synthetic canonical books
       without crashing or raising ZeroDivisionError.
     - Verify all 4 benchmarks execute in clean-room environment with 100% recall.
  2. Memory profiling stress test:
     - Multi-iteration physical OS RSS stress test.
     - Verify flat OS RSS on 100M+ token streams (Δ RSS < 15 MB).
     - True token eviction and flatline verification.
  3. CognitiveMemoryEngine adversarial stress test:
     - Bounded memory under massive anchor flood (1,000+ distinct anchors).
     - Robust handling of empty deltas and noisy text.
     - Configurable max_anchors bounds.
  4. Exact KV-Cache formula verification:
     - 128 KB/token derivation.
     - 600 tokens = 78.6 MB / 0.079 GB (refuting 0.08 MB).
===============================================================================
"""

import os
import sys
import gc
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data.corpus_manager as cm
from cognitive_memory_engine import CognitiveMemoryEngine


class TestCleanRoomExecution:
    """Task 1: Clean-room environment execution without pre-existing corpus or /tmp files."""

    def test_clean_room_synthetic_canonical_books_generation(self):
        """Verify that get_canonical_books creates 19 synthetic books when /tmp and corpus are empty."""
        with tempfile.TemporaryDirectory() as clean_dir:
            clean_corpus_path = Path(clean_dir) / "clean_corpus"
            clean_corpus_path.mkdir(parents=True, exist_ok=True)

            # Ensure directory is empty
            assert len(list(clean_corpus_path.glob("*.txt"))) == 0

            original_glob = Path.glob
            tmp_resolved = Path("/tmp").resolve()

            def mock_glob(path_obj, pattern):
                if path_obj.resolve() == tmp_resolved and pattern == "*.txt":
                    return iter([])
                return original_glob(path_obj, pattern)

            with patch.dict(os.environ, {"CORPUS_DIR": str(clean_corpus_path)}):
                with patch.object(Path, "glob", side_effect=mock_glob, autospec=True):
                    books = cm.get_canonical_books()

                    # 1. Verify that 19 books were created
                    assert len(books) == 19, f"Expected 19 books, got {len(books)}"

                    # 2. Verify all files are in the clean corpus directory
                    for b in books:
                        assert b.is_file(), f"Book {b} is not a file"
                        assert b.stat().st_size >= 300 * 1024, f"Book {b} is smaller than 300KB: {b.stat().st_size} bytes"
                        assert str(clean_corpus_path.resolve()) in str(b.resolve()), f"Book {b} not in clean corpus path"

                    # 3. Test ZeroDivisionError condition
                    # In test_100m_tokens_cognitive.py: chunk_text = books[chunk_id % len(books)]
                    for chunk_id in range(1000):
                        idx = chunk_id % len(books)
                        assert 0 <= idx < 19

    def test_clean_room_synthetic_bible_generation(self):
        """Verify that get_bible_path creates synthetic 66-book Bible when /tmp and corpus are empty."""
        with tempfile.TemporaryDirectory() as clean_dir:
            clean_corpus_path = Path(clean_dir) / "clean_corpus"
            clean_corpus_path.mkdir(parents=True, exist_ok=True)

            original_glob = Path.glob
            tmp_resolved = Path("/tmp").resolve()
            tmp_bible = (tmp_resolved / "biblia_tysiacletnia_lub_gdanska.txt").resolve()

            def mock_glob(path_obj, pattern):
                if path_obj.resolve() == tmp_resolved and pattern == "*.txt":
                    return iter([])
                return original_glob(path_obj, pattern)

            original_is_file = Path.is_file
            def mock_is_file(path_obj):
                if path_obj.resolve() == tmp_bible:
                    return False
                return original_is_file(path_obj)

            with patch.dict(os.environ, {"CORPUS_DIR": str(clean_corpus_path)}):
                with patch.object(Path, "glob", side_effect=mock_glob, autospec=True):
                    with patch.object(Path, "is_file", side_effect=mock_is_file, autospec=True):
                        bible_file = cm.get_bible_path()
                        assert bible_file.is_file()
                        assert bible_file.stat().st_size > 100_000
                        assert str(clean_corpus_path.resolve()) in str(bible_file.resolve())

                        with open(bible_file, "r", encoding="utf-8") as f:
                            text = f.read()

                        # Verify all 66 books are present
                        sections = text.split("### ")
                        assert len(sections) - 1 == 66, f"Expected 66 books, found {len(sections) - 1}"

                        # Verify critical factual anchors for benchmark questions
                        assert "Matuzalem" in text and "969" in text
                        assert "trzysta łokci" in text and "pięćdziesiąt łokci" in text
                        assert "drewna akacjowego" in text and "dwa i pół łokcia" in text
                        assert "Salomon budował dla Pana" in text and "sześćdziesiąt łokci" in text
                        assert "sześćset sześćdziesiąt sześć talentów" in text
                        assert "pięć chlebów" in text and "dwie ryby" in text
                        assert "trzydzieści srebrników" in text
                        assert "sto czterdzieści cztery tysiące" in text
                        assert "liczba bestii" in text and "666" in text

    def test_clean_room_full_benchmark_suite_execution(self):
        """Execute run_reproducible_benchmarks.py --skip-llm using purely synthetic clean-room corpus."""
        with tempfile.TemporaryDirectory() as clean_dir:
            clean_corpus_path = Path(clean_dir) / "synthetic_corpus"
            clean_corpus_path.mkdir(parents=True, exist_ok=True)

            # Generate synthetic canonical corpus + bible in clean_corpus_path
            cm.generate_synthetic_canonical_corpus(clean_corpus_path, num_books=19, target_size_bytes=350 * 1024)
            synthetic_bible = clean_corpus_path / "biblia_tysiacletnia_lub_gdanska.txt"
            cm.generate_synthetic_bible(synthetic_bible)

            env = dict(os.environ)
            env["CORPUS_DIR"] = str(clean_corpus_path)
            env["BIBLE_PATH"] = str(synthetic_bible)

            # Execute run_reproducible_benchmarks.py in this clean-room environment
            cmd = [sys.executable, str(PROJECT_ROOT / "run_reproducible_benchmarks.py"), "--skip-llm"]
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            assert proc.returncode == 0, f"Benchmark failed with returncode {proc.returncode}"
            assert "PODSUMOWANIE AUDYTU REPRODUKOWALNOŚCI BENCHMARKÓW" in proc.stdout
            assert "9/9 (100%)" in proc.stdout
            assert "5/5 (100%)" in proc.stdout
            assert "ZeroDivisionError" not in proc.stdout
            assert "ZeroDivisionError" not in proc.stderr


class TestMemoryProfilingStress:
    """Task 2: Physical OS RSS stability test across multiple iterations."""

    def test_streaming_110m_tokens_flatline_rss(self):
        """Verify that streaming 109,738,160 tokens has net Δ RSS < 15 MB."""
        from test_100m_tokens_cognitive import run_100m_benchmark
        res = run_100m_benchmark(skip_llm=True, total_chunks=190)
        assert res["tokens_processed"] > 100_000_000
        assert res["recall_acc"] == 100.0
        delta_rss = res["memory_profile"]["delta_rss_mb"]
        print(f"\n110M Tokens Streaming Δ RSS: {delta_rss:+.2f} MB")
        assert delta_rss < 15.0, f"110M streaming Δ RSS {delta_rss:.2f} MB exceeds 15 MB"

    def test_pure_semantic_110m_flatline_rss(self):
        """Verify that pure semantic 110M streaming benchmark has net Δ RSS < 15 MB."""
        from test_100m_non_numeric import run_non_numeric_100m_benchmark
        res = run_non_numeric_100m_benchmark(skip_llm=True, total_chunks=190)
        assert res["tokens_processed"] > 100_000_000
        assert res["recall_acc"] == 100.0
        delta_rss = res["memory_profile"]["delta_rss_mb"]
        print(f"\nPure Semantic 110M Streaming Δ RSS: {delta_rss:+.2f} MB")
        assert delta_rss < 15.0, f"Pure Semantic Δ RSS {delta_rss:.2f} MB exceeds 15 MB"


class TestCognitiveMemoryEngineAdversarial:
    """Adversarial stress testing of CognitiveMemoryEngine."""

    def test_infinite_stream_additional_anchors_cap(self):
        """Stress-test update() with 1,000 distinct high-entropy anchors to ensure strict O(1) bound."""
        engine = CognitiveMemoryEngine(max_anchors=16)
        state = engine.ingest("Start z kodem REF-0001.")

        for i in range(1000):
            delta = f"Nowa poprawka z kodem REF-{i:05d} oraz certyfikatem REG-{i:05d}."
            state = engine.update(state, delta)
            anchors = state.get("additional_anchors", [])
            assert len(anchors) <= 16, f"Anchor count exceeded 16 at step {i}: {len(anchors)}"

        assert len(state["additional_anchors"]) == 16
        formatted = engine.format_working_memory(state)
        assert len(formatted) < 2000  # Strictly bounded working memory string

    def test_large_payload_and_empty_delta(self):
        """Test engine behavior under text payload and empty deltas."""
        engine = CognitiveMemoryEngine()
        state = engine.ingest("Standardowy kontrakt: wycena pre-money wynosi 10 000 000 EUR.")
        assert state.get("pre_money_valuation") == "10 000 000 EUR"

        # Empty delta update
        state_after_empty = engine.update(state, "")
        assert state_after_empty == state

        # Noisy payload with valid high-entropy anchor
        noisy_filler = "Zwykły szum korpusowy bez żadnych istotnych encji. " * 500  # ~25 KB
        noisy_payload = noisy_filler + "\nKlucz autoryzacyjny ID-9999-MASSIVE.\n" + noisy_filler
        state_updated = engine.update(state, noisy_payload)
        assert "additional_anchors" in state_updated
        assert any("ID-9999-MASSIVE" in a for a in state_updated["additional_anchors"])

    def test_custom_max_anchors_setting(self):
        """Verify that max_anchors parameter can be configured to other bounds (e.g. 5 or 32)."""
        engine_small = CognitiveMemoryEngine(max_anchors=5)
        state = {}
        for i in range(50):
            state = engine_small.update(state, f"Kotwica REF-{i:04d}")
        assert len(state["additional_anchors"]) == 5

        engine_large = CognitiveMemoryEngine(max_anchors=32)
        state2 = {}
        for i in range(50):
            state2 = engine_large.update(state2, f"Kotwica REF-{i:04d}")
        assert len(state2["additional_anchors"]) == 32


class TestKVCacheFormulas:
    """Audit mathematical accuracy of hardware KV-cache calculations."""

    def test_exact_kv_cache_derivation(self):
        """Verify 1 token = 128 KB, 600 tokens = 78.6 MB, 110M tokens = 14.38 TB."""
        # 1 token
        b1 = cm.calculate_kv_cache_bytes(1)
        assert b1 == 2 * 32 * 8 * 128 * 2 == 131072  # 128 KB

        # 600 tokens
        b600 = cm.calculate_kv_cache_bytes(600)
        assert b600 == 78_643_200
        mb_decimal = b600 / 1e6
        assert abs(mb_decimal - 78.6432) < 1e-4

        # Disproving 0.08 MB
        assert b600 / 1e9 < 0.08  # It is 0.0786 GB, which rounds to 0.08 GB, NOT 0.08 MB!

        # 110M tokens
        b110m = cm.calculate_kv_cache_bytes(109_738_160)
        tb = b110m / 1e12
        assert 14.3 < tb < 14.5


if __name__ == "__main__":
    pytest.main(["-v", __file__])
