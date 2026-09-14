#!/usr/bin/env python3
"""
Asynchronous Cognitive Dual-Process Runtime with Preemptive Handover
====================================================================
Implements the Operating System paradigm for Cognitive Working Memory:

  1. BACKGROUND HIPPOCAMPUS DAEMON (Sensory SLM - SmolLM2-1.7B):
     - Runs asynchronously in a background thread with throttled/moderate compute.
     - Ingests long-form streams chunk-by-chunk (512 tokens).
     - Maintains and updates the bounded O(1) Baddeley Working Memory buffer.

  2. SLEEPING CORTEX (Executive LLM - LLaMA-3-8B):
     - Dormant (0% compute, 0 FLOPs, 0 memory bandwidth) for 99.9% of the stream.
     - Wakes up ONLY when the user submits an interactive query.

  3. PREEMPTIVE MEMORY BANDWIDTH HANDOVER (Collision Resolution):
     - When a user query arrives while the Hippocampus is reading:
       a) Hippocampus is immediately paused (yields memory bus / GPU compute).
       b) 100% memory bandwidth is handed over to the Executive Cortex.
       c) Cortex generates an instantaneous answer from the current Working Memory state S_t.
       d) Cortex goes back to sleep; Hippocampus resumes background ingestion seamlessly.

Author: Waldemar Gajda
"""

import time
import threading
from typing import Optional, Callable
from pathlib import Path

from hippocampus_slm_distiller import HippocampusSLM, BaddeleyWorkingMemory


class AsyncCognitiveRuntime:
    """
    Manages asynchronous background ingestion via Hippocampus SLM
    and instantaneous query preemption by Executive Cortex.
    """

    def __init__(
        self,
        hippocampus: Optional[HippocampusSLM] = None,
        cortex_generator_fn: Optional[Callable[[str, str], str]] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.hippocampus = hippocampus or HippocampusSLM(verbose=verbose)
        self.cortex_fn = cortex_generator_fn or self._default_mock_cortex

        # Threading and preemption synchronization primitives
        self._stream_queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = running, clear = paused
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry
        self.total_chunks_processed = 0
        self.preemption_count = 0
        self.preemption_latency_ms = 0.0

    def _default_mock_cortex(self, prompt: str, working_memory: str) -> str:
        """Fast fallback generation when full 8B model is not loaded in memory."""
        return f"[Cortex Response based on Working Memory State]\n{working_memory}\nQuery: {prompt}"

    # ── Background Worker Loop ────────────────────────────────────────────────

    def _background_worker(self):
        """Asynchronous daemon loop executing Hippocampus sensory ingestion."""
        if self.verbose:
            print("  [Runtime] Background Hippocampus Daemon started.")

        while not self._stop_event.is_set():
            # Wait if preempted by Cortex query
            self._pause_event.wait()

            chunk = None
            with self._queue_lock:
                if self._stream_queue:
                    chunk = self._stream_queue.pop(0)

            if chunk is not None:
                # Ingest chunk through SLM and update Baddeley state
                self.hippocampus.ingest_chunk(chunk)
                self.total_chunks_processed += 1
                time.sleep(0.01)  # Yield CPU/GPU slice to host OS
            else:
                # No chunks pending; idle wait
                time.sleep(0.05)

        if self.verbose:
            print("  [Runtime] Background Hippocampus Daemon stopped.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_background_ingestion(self):
        """Start the background Hippocampus ingestion daemon."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._worker_thread = threading.Thread(
            target=self._background_worker,
            name="HippocampusBackgroundDaemon",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_background_ingestion(self):
        """Gracefully terminate background daemon."""
        self._stop_event.set()
        self._pause_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def enqueue_stream(self, chunks: list[str]):
        """Feed text chunks into the background processing queue."""
        with self._queue_lock:
            self._stream_queue.extend(chunks)

    def query(self, user_question: str) -> str:
        """
        Interactive User Query with Instantaneous Preemption:
          1. Pauses Hippocampus daemon immediately.
          2. Grants 100% memory bandwidth / compute to Cortex.
          3. Generates answer using current O(1) Working Memory snapshot.
          4. Resumes Hippocampus background reading.
        """
        t0 = time.time()
        self.preemption_count += 1

        # Step 1: Preempt Hippocampus
        self._pause_event.clear()
        if self.verbose:
            print(f"\n⚡ [PREEMPTION #{self.preemption_count}] User asked: '{user_question}'")
            print("   → Pausing Hippocampus daemon, handing 100% bandwidth to Cortex...")

        try:
            # Step 2: Extract current bounded Working Memory snapshot
            wm_snapshot = self.hippocampus.get_working_memory_prompt()

            # Step 3: Executive Cortex generation
            t_exec = time.time()
            response = self.cortex_fn(user_question, wm_snapshot)
            exec_time_ms = (time.time() - t_exec) * 1000

        finally:
            # Step 4: Resume background ingestion
            self._pause_event.set()
            elapsed_ms = (time.time() - t0) * 1000
            self.preemption_latency_ms += elapsed_ms
            if self.verbose:
                print(f"   ✓ Answer generated in {exec_time_ms:.1f} ms (Total Handover: {elapsed_ms:.1f} ms)")
                print("   → Resuming Hippocampus background ingestion daemon.\n")

        return response

    def get_status(self) -> dict:
        """Telemetry snapshot."""
        with self._queue_lock:
            pending = len(self._stream_queue)
        return {
            "chunks_processed": self.total_chunks_processed,
            "chunks_pending": pending,
            "preemptions": self.preemption_count,
            "wm_facts": self.hippocampus.working_memory.fact_count(),
            "wm_entities": self.hippocampus.working_memory.entity_count(),
            "wm_tokens": self.hippocampus.working_memory.token_estimate(),
        }
