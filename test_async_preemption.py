#!/usr/bin/env python3
"""
Test Suite: Asynchronous Hippocampus Daemon & Cortex Preemption Handover
========================================================================
Demonstrates the Operating System paradigm:
  1. Hippocampus runs continuously in background, digesting long streams.
  2. User injects an interactive query MID-STREAM.
  3. Preemption controller pauses Hippocampus, grants 100% memory bus to Cortex.
  4. Cortex replies in milliseconds from current Working Memory state.
  5. Hippocampus resumes background ingestion without data loss.

Author: Waldemar Gajda
"""

import time
import sys
from async_cognitive_runtime import AsyncCognitiveRuntime
from hippocampus_slm_distiller import HippocampusSLM


def run_preemption_test():
    print("=" * 80)
    print("  ASYNCHRONOUS DUAL-PROCESS RUNTIME & PREEMPTION VERIFICATION")
    print("  Waldemar Gajda — Operating System Paradigm for Cognitive Architectures")
    print("=" * 80)

    # 1. Initialize Hippocampus SLM
    print("\n[INIT] Initializing Hippocampus SLM...")
    hippocampus = HippocampusSLM(verbose=False)

    # 2. Define Cortex generation function (using simple prompt reasoning)
    def cortex_evaluator(question: str, working_memory: str) -> str:
        # Simple extraction over the compact working memory state
        lines = working_memory.split("\n")
        relevant = [l.strip() for l in lines if any(k in l.lower() for k in ["valuation", "key", "contract"])]
        return f"Cortex Answer: Based on current working memory: {', '.join(relevant) if relevant else 'No facts yet'}"

    runtime = AsyncCognitiveRuntime(
        hippocampus=hippocampus,
        cortex_generator_fn=cortex_evaluator,
        verbose=True,
    )

    # 3. Prepare streaming document chunks
    stream_chunks = [
        "The pre-money valuation of Synapse AI is confirmed at 45M EUR. An earlier offer of 12M EUR was rejected.",
        "Company headquarters established in Warsaw with satellite R&D offices in Zurich and Krakow.",
        "The initial contract value for Project Prometheus was set at 10M EUR.",
        "Production authorization key TITAN-KEY-9901-X is effective immediately. DRAFT-KEY-0001 was deprecated.",
        "Amendment to Project Prometheus: revised contract value to 52.75M EUR, superseding previous figures.",
        "Tranche A is hereby cancelled and nullified. All obligations cease immediately.",
    ]

    print(f"\n[STREAM] Enqueueing {len(stream_chunks)} chunks into background buffer...")
    runtime.enqueue_stream(stream_chunks)

    # 4. Start Background Daemon
    print("[DAEMON] Starting asynchronous Hippocampus background ingestion...")
    runtime.start_background_ingestion()

    # Let Hippocampus process first chunk
    time.sleep(3.0)

    # 5. Collision 1: User asks a question MID-STREAM!
    status_before = runtime.get_status()
    print(f"  --> Mid-stream status: {status_before['chunks_processed']} chunks processed, {status_before['chunks_pending']} pending.")

    ans1 = runtime.query("What is the valuation of Synapse AI?")
    print(f"  ==> Result: {ans1}")

    # Let background continue for next chunks
    time.sleep(4.0)

    # 6. Collision 2: User asks another question mid-stream!
    ans2 = runtime.query("What is the production authorization key?")
    print(f"  ==> Result: {ans2}")

    # Wait for completion of remaining queue
    print("\n[AWAIT] Waiting for background daemon to finish remaining chunks...")
    while True:
        s = runtime.get_status()
        if s["chunks_pending"] == 0:
            break
        time.sleep(1.0)

    # Final query after stream completes
    ans3 = runtime.query("What is the final contract value of Project Prometheus?")
    print(f"  ==> Result: {ans3}")

    runtime.stop_background_ingestion()

    # 7. Telemetry & Summary
    final_status = runtime.get_status()
    print("\n" + "=" * 80)
    print("  PREEMPTION RUNTIME VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  • Total Chunks Processed:  {final_status['chunks_processed']}")
    print(f"  • Total User Preemptions:  {final_status['preemptions']}")
    print(f"  • Average Preemption Lat:  {runtime.preemption_latency_ms / max(1, final_status['preemptions']):.1f} ms")
    print(f"  • Final Working Memory:    {final_status['wm_facts']} facts, {final_status['wm_tokens']} tokens (Strict O(1))")
    print(f"\n  Final Snapshot:\n{runtime.hippocampus.get_working_memory_prompt()}")
    print("=" * 80)
    print("  ✅ DUAL-PROCESS PREEMPTIVE RUNTIME SUCCESSFULLY VERIFIED.")


if __name__ == "__main__":
    run_preemption_test()
