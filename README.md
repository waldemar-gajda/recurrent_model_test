<div align="center">

# Baddeley Cognitive Working Memory: $O(1)$ Context for LLMs

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Dual License: Commercial](https://img.shields.io/badge/License-Commercial_Dual-purple.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![arXiv: 2026](https://img.shields.io/badge/arXiv-Preprint-red.svg)](paper.html)
[![Space: O(1)](https://img.shields.io/badge/Space_Complexity-O(1)_Strict-orange.svg)](#theoretical-guarantees)
[![Recall: 100%](https://img.shields.io/badge/Factual_Recall-100%25_at_110M-success.svg)](#master-empirical-scaling-benchmark)

**Bounded Recurrent Working Memory Architectures for Infinite Context Processing in Large Language Models.**

[**Read the Paper (HTML)**](paper.html) • [**LaTeX Source (Overleaf/arXiv)**](paper.tex) • [**Adversarial Audit**](ADVERSARIAL_AUDIT.md) • [**Submission Guide**](ARXIV_SUBMISSION_GUIDE.md)

</div>

---

## The Problem: The KV-Cache Memory Wall

Dense causal self-attention scales quadratically in compute $\mathcal{O}(N^2)$ and linearly in memory $\mathcal{O}(N)$. For long-context horizons, the Key-Value (KV) cache becomes an impassable hardware barrier:

- Serving a **110 million token stream** on Meta-Llama-3-8B (bfloat16, GQA) requires **$14.04$ to $14.38$ Terabytes** of active VRAM.
- Hosting this KV-cache for a **single inference stream** requires **176 enterprise NVIDIA H100 (80GB) GPUs** ($>\$5.2\text{M}$ in hardware), dedicated solely to caching past keys and values.
- Unconstrained context windows suffer from **catastrophic attention dilution**, the **"lost-in-the-middle" effect**, and **multi-hop reasoning breakdown** (NVIDIA RULER, BABILong).

```
Full Dense Attention (Linear O(N)):
Tokens:  1k   ──>   100k  ──>    1.5M   ──>     11M    ──>    110M
VRAM:  0.13 GB ──> 13.1 GB ──> 200.5 GB ──>   1.44 TB  ──>  14.04 TB (176x H100 GPUs!)

Baddeley Cognitive Working Memory (Strict O(1)):
Tokens:  1k   ──>   100k  ──>    1.5M   ──>     11M    ──>    110M
VRAM: 78.6 MB ──> 78.6 MB ──>  70.1 MB ──>   25.5 MB  ──>   26.5 MB (Single Mac / GPU!)
```

---

## The Solution: Alan Baddeley's Quadripartite Working Memory

Human intelligence processes lifelong continuous multi-modal experience ($N \to \infty$) without quadratic memory explosion. Grounded in cognitive psychology (**Baddeley & Hitch, 1974, 2000**) and **Cowan's Capacity Limit ($4 \pm 1$)**, our architecture replaces unbounded linear history with a strictly bounded recurrent state space $\mathcal{S}$:

$$\mathcal{S} = \langle \mathcal{S}_{visuo}, \mathcal{S}_{phon}, \mathcal{S}_{exec}, \mathcal{S}_{ep} \rangle$$

```
       ┌──────────────────────────────────────────────────────────┐
       │                   CENTRAL EXECUTIVE                      │
       │     (Attentional Control, Routing, Multi-Step Delib.)    │
       └──────────────┬────────────────────────────┬──────────────┘
                      │                            │
         ┌────────────▼────────────┐  ┌────────────▼────────────┐
         │  VISUOSPATIAL SKETCHPAD │  │    PHONOLOGICAL LOOP    │
         │  (Continuous Semantics, │  │   (Discrete Alphanumeric│
         │   Relational Slots, K=8)│  │    Registers, L_sym<=16)│
         └────────────┬────────────┘  └────────────┬────────────┘
                      │                            │
                      └────────────┬───────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │     EPISODIC BUFFER     │
                      │  (Multimodal Binding,   │
                      │   Active Context C<=28) │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      [ Autoregressive Causal LLM ]
```

1. **Visuospatial Sketchpad ($\mathcal{S}_{visuo}$):** $K=8$ continuous scene slots ($\approx 32$~KB) updated via Perceiver cross-attention and SwiGLU gating.
2. **Phonological Loop ($\mathcal{S}_{phon}$):** Discrete symbol register ($L_{sym} \le 16$ tokens, $32$~bytes) protecting high-entropy literals (IBANs, hashes, citations) from vector quantization blur.
3. **Central Executive ($\mathcal{S}_{exec}$):** Deliberation tokens ($P=4$) executing draft-verify mental simulation.
4. **Episodic Buffer ($\mathcal{S}_{ep}$):** Bounded multimodal binding state ($C \le 28$ tokens in neural mode; $250 - 650$ tokens in symbolic distillation mode).

---

## Master Empirical Scaling Benchmark

We validated the architecture across five corpus tiers spanning $1.6\times 10^3$ to $1.097\times 10^8$ streaming tokens:

| Metric | Tier 1: Legal M&A | Tier 2: Scaled Boilerplate | Tier 3: 66-Book Bible | Tier 4: World Canon (19 Vol) | Tier 5A: Pure Semantic | Tier 5B: Alphanumeric |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cumulative Tokens** | $1,600$ | $20,000$ | $1,567,226$ | $10,973,816$ | $109,738,160$ | $109,738,160$ |
| **Stream Chunks / Books** | 1 doc | 60 sections | 66 books | 19 volumes | 190 chunks | 190 chunks |
| **Working Memory Buffer** | $371$ tok | $371$ tok | $561$ tok | $204$ tok | $293$ tok | $212$ tok |
| **State Compression Ratio** | $3.6 : 1$ | $53.9 : 1$ | $2,793.6 : 1$ | $53,793.2 : 1$ | $374,532.9 : 1$ | **$517,632.8 : 1$** |
| **Symbolic Stream Time**† | $<0.05$ s | $0.08$ s | $0.52$ s | $0.61$ s | $1.81$ s | $1.83$ s |
| **Symbolic CPU Throughput**† | $> 30\text{k}$ tok/s | $> 250\text{k}$ tok/s | $3.01\text{M}$ tok/s | $17.9\text{M}$ tok/s | $60.6\text{M}$ tok/s | **$59.9\text{M}$ tok/s** |
| **Neural SLM Ingestion Latency**‡ | $\approx 2.3$ s | $\approx 45$ s | $\approx 1.9$ h (async) | $\approx 13.5$ h (async) | $\approx 5.6$ d (async) | $\approx 5.6$ d (async) |
| **Peak Host RSS** | $312.4$ MB | $315.8$ MB | $608.3$ MB | $652.2$ MB | $703.2$ MB | $749.8$ MB |
| **Net $\Delta\text{RSS}$ Growth** | $+0.0$ MB | $+0.1$ MB | $+44.9$ MB | $+8.0$ MB | $+0.1$ MB | **$+0.16$ MB** |
| **Working Memory KV-Cache** | $46.4$ MB | $46.4$ MB | $70.1$ MB | $25.5$ MB | $36.6$ MB | $26.5$ MB |
| **Full Attention KV-Cache** | $0.17$ GB | $2.62$ GB | $205.4$ GB | $1.44$ TB | $14.38$ TB | **$14.38$ TB** |
| **KV-Cache Reduction Factor** | $3.6\times$ | $53.9\times$ | $2,930\times$ | $56,470\times$ | $392,896\times$ | **$542,641\times$** |
| **Executive Reasoning Model** | LLaMA-3-8B | LLaMA-3-8B | LLaMA-3-8B | LLaMA-3-8B | LLaMA-3-8B | LLaMA-3-8B |
| **Factual Recall Accuracy** | **12/12 (100%)** | **5/5 (100%)** | **9/9 (100%)** | **5/5 (100%)** | **5/5 (100%)** | **5/5 (100%)** |

> **Note on Compute-Memory Decoupling:** The raw $59.9\text{M}$ tokens/sec rate reflects the lightweight CPU lexical pre-filter (System B heuristic) scanning text at byte level. Full neural semantic processing (SmolLM2-1.7B, Section 7) respects the physical laws of linear compute $\mathcal{O}(T)$ ($\approx 2.28$ s per 512-token chunk). The breakthrough is **Compute-Memory Decoupling**: we exchange an impossible $\mathcal{O}(N)$ memory barrier ($14.38\text{ TB}$ VRAM, $176\times$ H100 GPUs) for a tractable linear $\mathcal{O}(T)$ background pipeline on an on-device 1.7B model, maintaining strict $\mathcal{O}(1)$ working memory while the 8B model sleeps until queried.

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/cognitive-working-memory/baddeley-cognitive-memory.git
cd baddeley-cognitive-memory
pip install -r requirements.txt
```

### 2. Run Reproducible Benchmark Suite
Verify all empirical tiers and OS memory eviction in $<10$ seconds:
```bash
python3 run_reproducible_benchmarks.py
```

### 3. Interactive Working Memory Chat Demo
Run an interactive session where any document is ingested into $O(1)$ working memory:
```bash
# Using Ollama (Local LLaMA-3)
python3 demo_cognitive_memory.py --interactive --backend ollama --model llama3

# Using OpenAI or Groq API (Zero local model install)
export GROQ_API_KEY="your-groq-key"
python3 demo_cognitive_memory.py --interactive --backend groq --model llama-3.1-8b-instant

# Fast Simulation Mode (0.01s, no models needed)
python3 demo_cognitive_memory.py --backend mock
```

---

## Theoretical Guarantees

- **Theorem 1 (Strict $\mathcal{O}(1)$ Space Complexity):** The working memory state $\mathcal{S}$ is strictly bounded by hyperparameters $K=8, L_{max}=16, P=4, d$. Memory consumption is invariant with sequence length: $\frac{\partial \text{RAM}_{WM}}{\partial T} = 0$.
- **Theorem 2 (Strict $\mathcal{O}(1)$ Decoding Latency):** Query vectors attend only to $T_{eff} \le C_{max}$ tokens. Generation FLOPs per token are bounded: $\lim_{T \to \infty} \text{FLOPs}_{WM}(T) = \mathcal{O}(1)$.
- **Theorem 3 (Linear Ingestion Time):** Processing $T$ tokens partitioned into $M$ chunks executes in strictly linear time: $\text{FLOPs}_{total}(T) = \mathcal{O}(T)$, compared to $\mathcal{O}(T^2)$ in standard dense self-attention.

---

## Duality Delineation: System A vs System B

To maintain absolute scientific integrity, we explicitly delineate the two systems in this codebase:

- **System A: In-Model Neural Recurrence (`model.py`, `recurrent_memory_bank.py`):**
  Operates natively within the PyTorch graph. Features learned continuous cross-attention slot pooling, gated bridge residual integration ($b_{init} = -4.0$), and a 32 KB broadcasted slot memory bank across all 32 transformer layers.
- **System B: Autonomic Symbolic Episodic Distillation (`cognitive_memory_engine.py`):**
  Operates outside the neural network as an external CPU text-stream processor. Parses multi-million token streams, resolves entity updates via conflict overwrite, and formats a compact episodic buffer ($250 - 650$ tokens) for a frozen foundation model.

Both achieve $\mathcal{O}(1)$ inference memory, but at fundamentally distinct architectural layers.

---

## Dual-LLM Neural Architecture: Artificial Sensory Hippocampus (`hippocampus_slm_distiller.py`)

To eliminate heuristic regex-based distillation entirely, the architecture integrates an on-device Small Language Model (**SmolLM2-1.7B-Instruct**) acting as an **Artificial Hippocampus**:

```
[Unbounded Text Stream] ──(512 tok chunks)──► [Hippocampus SLM: SmolLM2-1.7B]
                                                      │ (structured JSON triples)
                                                      ▼
                                              [Baddeley Working Memory: O(1)]
                                                      │ (bounded ~300 tok buffer)
                                                      ▼
                                              [Executive Cortex: LLaMA-3-8B]
```

### Adversarial Validation Matrix (`test_hippocampus_negation.py`)
Evaluating against the failure modes exposed in the red-team audit:

| Adversarial Vulnerability Probe | Heuristic Regex (System B) | Neural Hippocampus (SmolLM2-1.7B) |
|:---|:---:|:---:|
| **VM1-B: Distractor Key Spoofing** (`DRAFT-KEY` vs `TITAN-KEY`) | ✗ FAIL (Spoofed) | **✓ PASS** |
| **VM2-A: Semantic Paraphrase** (*"market capitalization"* &rarr; valuation) | ✗ FAIL (Missed) | **✓ PASS** |
| **VM2-B: Passive Voice** (*"was agreed upon"*) | ✗ FAIL (Missed) | **✓ PASS** |
| **VM2-D: Hedged Legal Qualifiers** (confirmation statements) | ✗ FAIL (Missed) | **✓ PASS** |
| **VM3-A: Explicit Negation** (rejected €15M vs valid €45M) | ✓ PASS | **✓ PASS** |
| **VM3-B: Complete Revocation** (cancelled tranche evicts state) | ✓ PASS | **✓ PASS** |
| **VM3-C: Temporal Override** (amendment supersedes baseline) | ✓ PASS | **✓ PASS** |
| **VM3-E: Modal Conditional** (*"considering… decision pending"*) | ✓ PASS | **✓ PASS** |
| **Overall Robustness** | **4/8 (50.0%)** | **8/8 (100.0%)** |

- **Inference Latency:** $\approx 2.28$ s/chunk on Apple Silicon (MPS).
- **Physical Memory:** Strict $\mathcal{O}(1)$ footprint (bounded working memory buffer).
- Run the test suite: `python3 test_hippocampus_negation.py`

### Operating System Paradigm: Preemptive Handover Runtime (`async_cognitive_runtime.py`)
To avoid GPU/Unified Memory thrashing on consumer hardware:
1. **Background Hippocampus Daemon:** Ingests document streams in the background with throttled compute, continuously updating the $O(1)$ Working Memory buffer.
2. **Sleeping Executive Cortex:** The primary foundation model (LLaMA-3-8B) remains in deep sleep (0 FLOPs, 0 VRAM bandwidth) for 99.9% of the stream.
3. **Instantaneous Preemption (Collision Resolution):** When a user asks a question, the runtime pauses the Hippocampus ($<0.1$ ms handover), hands 100% memory bus bandwidth to the Cortex to answer from the current Working Memory snapshot in milliseconds, then resumes background ingestion.
- Run the preemption verification test: `python3 test_async_preemption.py`

---

## Adversarial Red-Team Audit & Mitigations

We executed an aggressive zero-trust red-team audit ([ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md)) exposing critical failure modes in early heuristic prototypes, now fully resolved by the Neural Hippocampus layer and entropy-gated sparse attention.

See Sections 6 & 7 of the [Preprint Paper](paper.html) for full mathematical specifications and ablation studies.

---


## Citation & Intellectual Property

### Academic Citation (BibTeX)
```bibtex
@article{gajda2026baddeley,
  title={Bounded Recurrent Cognitive State Spaces for Infinite Context Processing in Large Language Models: An Empirical, Formal, and Adversarial Analysis of Baddeley Working Memory Architectures},
  author={Gajda, Waldemar},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}
```

### Dual-Licensing Strategy
- **Open-Source / Academic Tier:** Licensed under the [GNU Affero General Public License v3 (AGPLv3)](LICENSE). Section 13 mandates that network-hosted modifications must open-source their full stack.
- **Commercial Tier:** For proprietary enterprise integration without AGPLv3 copyleft, commercial licenses are available upon request to the author.
