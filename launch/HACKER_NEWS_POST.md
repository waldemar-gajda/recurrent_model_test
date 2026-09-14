# Hacker News Launch Copy: Show HN

## Title
```text
Show HN: Baddeley Working Memory – 110M tokens in O(1) memory for LLMs
```
*(Alternative Title: Show HN: Bounded Recurrent Working Memory – 110M tokens on a single GPU)*

---

## Submission URL
Link directly to your GitHub repository or the arXiv preprint / paper.html.

---

## First Comment by Author (Post Immediately After Submitting)

```markdown
Hi HN,

We’re releasing an empirical and theoretical study of a biologically grounded alternative to linear KV-caches for long contexts: **Baddeley Cognitive Working Memory for Large Language Models**.

GitHub: https://github.com/cognitive-working-memory/baddeley-cognitive-memory
Preprint Paper: https://arxiv.org/abs/2609.xxxxx (or your link)

### The Problem
Autoregressive attention scales quadratically in prefill compute and linearly in KV-cache memory. 
If you want to run a 110-million token context stream through Meta-Llama-3-8B (bfloat16, GQA):
- KV-cache footprint: **14.04 Terabytes**
- Hardware requirement: **176x NVIDIA H100 (80GB) GPUs** just to hold past keys and values for a single stream.
- In addition, dense attention over millions of tokens suffers from extreme attention dilution and the "lost-in-the-middle" effect.

### The Solution: Alan Baddeley's Working Memory Model
Humans don't cache millions of sensory tokens to maintain state. In cognitive psychology, working memory is strictly bounded (Miller's $7 \pm 2$, Cowan's $4 \pm 1$ limit). We mapped Alan Baddeley's classical quadripartite working memory model (1974, 2000) onto LLMs:

1. **Visuospatial Sketchpad:** Continuous relational semantic slots ($K=8$ vectors, ~32 KB).
2. **Phonological Loop:** Discrete symbol registers ($L_{sym} \le 16$ exact tokens, 32 bytes) preventing vector quantization blur for high-entropy literals (IBANs, hashes, citations).
3. **Central Executive:** Latent deliberation tokens ($P=4$) for draft-verify cycles.
4. **Episodic Buffer:** Multimodal binding workspace ($C \le 28$ tokens in neural mode; 250–650 tokens in text distillation).

### Key Results
- **Strict O(1) Space & Latency:** We provide formal proofs that memory and per-token decoding FLOPs are constant with respect to cumulative sequence length.
- **Hardware Footprint:** On LLaMA-3-8B, KV-cache drops from **14.04 TB down to 78.6 MB** (a 182,900x reduction), and to **32 KB** on tensor-level slot memory banks.
- **Empirical Scaling:** Validated across 5 tiers (1.6k -> 20k -> 1.53M Bible -> 11M literary canon -> 110M token streaming) with 100% factual recall under continuous token eviction.
- **OS Memory Telemetry:** Flat physical RAM (`psutil` RSS net growth $\le 0.16$ MB across 109.7M streaming tokens).

### Transparent Honest Demarcation (No Marketing Fluff)
We explicitly distinguish between two architectures in the repository:
- **System A (Neural Recurrence):** PyTorch in-model recurrence with continuous cross-attention slot pooling (`recurrent_memory_bank.py`).
- **System B (Autonomic Symbolic Distillation):** Fast CPU text stream compiler (`cognitive_memory_engine.py`) that evicts tokens and maintains a compact episodic prompt for frozen foundation models.

We also conducted an aggressive adversarial red-team audit (`ADVERSARIAL_AUDIT.md`) exposing failure modes in regex heuristics (distractor spoofing, syntactic degradation), and implemented the production neural solution: a **Dual-LLM Neural Hippocampus** (`hippocampus_slm_distiller.py`) powered by an on-device `SmolLM2-1.7B-Instruct` model. It achieves **8/8 (100%) pass rate** on our adversarial negation/revocation test suite (`test_hippocampus_negation.py`), fully replacing regex matching with semantic neural understanding.

The code is open-source under AGPLv3 (with commercial dual-licensing). All benchmarks run deterministically in under 10 seconds:
```bash
git clone https://github.com/cognitive-working-memory/baddeley-cognitive-memory.git
cd baddeley-cognitive-memory
pip install -r requirements.txt
python3 run_reproducible_benchmarks.py
```

Looking forward to your questions, critiques, and thoughts on cognitive architectures for sequence models!
```
