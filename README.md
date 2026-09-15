<div align="center">

# Baddeley Cognitive Working Memory: O(1) Context for LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22768217.svg)](https://doi.org/10.5281/zenodo.22768217)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![Space: O(1)](https://img.shields.io/badge/Space_Complexity-O(1)_Query_Time-orange.svg)](#theoretical-guarantees)
[![bAbI: 60%](https://img.shields.io/badge/bAbI_Zero--Shot-60%25_(3/5)-yellow.svg)](#standardized-academic-benchmark-babi--babilong)
[![Proprietary Recall: 100%](https://img.shields.io/badge/Proprietary_Probes-100%25_at_110M_tokens-success.svg)](#master-empirical-scaling-benchmark)

**Bounded Recurrent Working Memory Architectures for Infinite Context Processing in Large Language Models.**

*Waldemar Gajda — Independent Researcher*

[**📄 Read the Paper (PDF, Zenodo)**](https://doi.org/10.5281/zenodo.22768217) · [**LaTeX Source**](paper.tex) · [**Adversarial Audit**](ADVERSARIAL_AUDIT.md)

</div>

---

## The Problem: The KV-Cache Memory Wall

Dense causal self-attention scales quadratically in compute O(N²) and linearly in memory O(N). For long-context horizons, the Key-Value (KV) cache becomes a hard hardware barrier:

- Serving a **110 million token stream** on Meta-Llama-3-8B (bfloat16, GQA) requires **14.04–14.38 Terabytes** of active VRAM.
- This KV-cache for a **single inference stream** requires **176 enterprise NVIDIA H100 (80 GB) GPUs**.
- Unconstrained context windows suffer from **catastrophic attention dilution**, the **"lost-in-the-middle" effect**, and **multi-hop reasoning breakdown** (NVIDIA RULER, BABILong).

```
Full Dense Attention — Linear O(N) memory:
Tokens:   1k  ──►  100k  ──►   1.5M   ──►    11M    ──►   110M
VRAM:  0.13 GB ──► 13.1 GB ──► 200.5 GB ──►  1.44 TB ──► 14.04 TB  (176× H100!)

Baddeley Working Memory — Strict O(1) query memory:
Tokens:   1k  ──►  100k  ──►   1.5M   ──►    11M    ──►   110M
VRAM:  78.6 MB ──► 78.6 MB ──►  70.1 MB ──►  25.5 MB ──►  26.5 MB  (Single Mac / GPU)
```

> ⚠️ **Important:** Query-time complexity is O(1). Document *ingestion* scales linearly O(T) —
> at 110M tokens this takes approximately **5.6 days** on Apple Silicon using the Neural Hippocampus.
> This architecture is designed for offline pre-processing of large corpora, not real-time streaming of arbitrarily long live feeds.

---

## The Solution: Alan Baddeley's Quadripartite Working Memory

Grounded in cognitive psychology (**Baddeley & Hitch, 1974, 2000**) and **Cowan's Capacity Limit (4 ± 1)**, the architecture replaces unbounded linear history with a strictly bounded recurrent state space S:

```
S = ⟨S_visuo,  S_phon,  S_exec,  S_ep⟩

       ┌──────────────────────────────────────────────────┐
       │              CENTRAL EXECUTIVE (P=4)             │
       │   Attentional Control · Multi-Step Deliberation  │
       └──────────────┬───────────────────────────────────┘
                      │
         ┌────────────▼────────────┐  ┌────────────────────────┐
         │  VISUOSPATIAL SKETCHPAD │  │    PHONOLOGICAL LOOP   │
         │  K=8 continuous slots   │  │  L_sym ≤ 16 exact tok  │
         │  ~32 KB, semantic rels  │  │  32 bytes, literals    │
         └────────────┬────────────┘  └────────────┬───────────┘
                      │                            │
                      └──────────────┬─────────────┘
                                     │
                       ┌─────────────▼──────────────┐
                       │       EPISODIC BUFFER       │
                       │  Multimodal binding, C ≤ 28 │
                       └─────────────┬──────────────┘
                                     ▼
                        [ Autoregressive Causal LLM ]
```

---

## System A vs. System B — Two Distinct Implementations

This repository contains two architecturally distinct systems. They are **not interchangeable** and are documented separately throughout the paper.

| | **System A: Neural Recurrence** | **System B: Neural Hippocampus** |
|---|---|---|
| **Files** | `model.py`, `recurrent_memory_bank.py` | `cognitive_memory_engine.py`, `hippocampus_slm_distiller.py` |
| **How it works** | Differentiable cross-attention slot pooling inside the PyTorch graph | On-device SLM (SmolLM2-1.7B) extracts JSON triples → bounded text buffer → frozen LLM |
| **Memory** | 32 KB slot bank (neural tensors) | ~78.6 MB KV-cache for 300–650 token prompt |
| **Status** | Implemented & unit-tested; trained checkpoints not publicly released | Fully operational; all 110M-token benchmarks use this pipeline |
| **Empirical data** | Forward/backward pass verified (see `test_model.py`) | Tables 3, 6, 7 in paper |

---

## Master Empirical Scaling Benchmark

All results below are from **System B (Neural Hippocampus)** pipeline. Factual accuracy is measured on **5–12 task-specific retrieval questions per tier** (proprietary probes — see bAbI section below for standardized evaluation).

| Metric | Tier 1: M&A | Tier 2: Boilerplate | Tier 3: Bible (1.5M) | Tier 4: Canon (11M) | Tier 5A: Semantic (110M) | Tier 5B: Alphanumeric |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cumulative Tokens** | 1,600 | 20,000 | 1,567,226 | 10,973,816 | 109,738,160 | 109,738,160 |
| **Working Memory Buffer** | 371 tok | 371 tok | 561 tok | 204 tok | 293 tok | 212 tok |
| **Compression Ratio** | 3.6:1 | 53.9:1 | 2,793:1 | 53,793:1 | 374,533:1 | **517,633:1** |
| **Neural SLM Ingestion**† | ~2.3 s | ~45 s | ~1.9 h | ~13.5 h | ~5.6 **days** | ~5.6 **days** |
| **Net ΔRSS Growth** | +0.0 MB | +0.1 MB | +44.9 MB | +8.0 MB | +0.1 MB | **+0.16 MB** |
| **Full Attention KV-Cache** | 0.17 GB | 2.62 GB | 205.4 GB | 1.44 TB | 14.38 TB | **14.38 TB** |
| **KV-Cache Reduction** | 3.6× | 53.9× | 2,930× | 56,470× | 392,896× | **542,641×** |
| **Factual Recall** | 12/12 (100%) | 5/5 (100%) | 9/9 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |

> †Neural SLM latency = full SmolLM2-1.7B semantic distillation, async background daemon.

---

## Standardized Academic Benchmark (bAbI / BABILong)

To ensure comparability against published literature, we evaluated on **5 selected tasks** from the Meta AI [bAbI suite](https://research.fb.com/downloads/babi/) (zero-shot, no fine-tuning):

| Task | Category | Target | Result | Status |
|---|---|---|---|---|
| bAbI-1: Single Supporting Fact | Single-Hop Location | `office` | `Office` | ✅ PASS |
| bAbI-2: Two Supporting Facts | Two-Hop Chaining | `garden` | `kitchen` | ❌ FAIL |
| bAbI-3: Three Supporting Facts | Three-Hop Movement | `bedroom` | `Bedroom` | ✅ PASS |
| bAbI-6: Yes/No Polarity | State Verification | `no` | `No` | ✅ PASS |
| bAbI-8: Lists / Sets | Inventory Binding | `apple, pear` | `Apple` | ⚠️ Partial |
| **Total** | | | | **3/5 (60%)** |

**Task 2 failure** (2-hop transitive chaining) and **Task 8 partial failure** (inventory sets) represent the known frontier of flat episodic buffers: transitive relational dependencies require explicit graph traversal, not flat key-value storage. This motivates the Subject-Predicate-Object graph extension described in Section 6.4 of the paper.

```bash
python3 benchmark_babi_suite.py
```

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/waldemargajda/recurrent_model_test.git
cd recurrent_model_test
pip install -r requirements.txt
```

> **Models required (not included in repo):**
> - `Meta-Llama-3-8B-Instruct` — download via HuggingFace or Ollama
> - `SmolLM2-1.7B-Instruct` — download via HuggingFace

### 2. Run Reproducible Benchmark Suite (System B)
Verifies OS memory eviction across all corpus tiers:
```bash
python3 run_reproducible_benchmarks.py
```

### 3. Run Adversarial Test Suite
```bash
python3 test_adversarial_redteam.py
python3 test_hippocampus_negation.py
```

### 4. Run Standardized bAbI Evaluation
```bash
python3 benchmark_babi_suite.py
```

### 5. Run System A Unit Tests (forward/backward pass)
```bash
pytest test_model.py -v
```

### 6. Interactive Chat (requires local LLM)
```bash
# With Ollama
python3 demo_cognitive_memory.py --interactive --backend ollama --model llama3

# With Groq API
export GROQ_API_KEY="your-key"
python3 demo_cognitive_memory.py --interactive --backend groq --model llama-3.1-8b-instant

# Mock mode (no models needed)
python3 demo_cognitive_memory.py --backend mock
```

---

## Theoretical Guarantees

- **Theorem 1 — O(1) Space Complexity:** The state S is strictly bounded by hyperparameters K=8, L_max=16, P=4. Memory is invariant with sequence length: ∂RAM_WM/∂T = 0.
- **Theorem 2 — O(1) Query Latency:** Query vectors attend only to T_eff ≤ C_max tokens. FLOPs per generated token are bounded regardless of document length.
- **Theorem 3 — O(T) Ingestion Time:** Processing T tokens in M chunks is O(T), vs O(T²) for dense self-attention.

> **Note:** Theorems 1 & 2 hold by construction of the bounded buffer. The empirically open question is whether a buffer of this capacity retains sufficient task-relevant information as T → ∞ — which is precisely what the benchmarks above begin to characterize.

---

## Adversarial Evaluation (System B)

We evaluated the heuristic regex baseline and its successor (Neural Hippocampus) against five vulnerability classes:

| Attack Surface | Heuristic Regex | Neural Hippocampus (SmolLM2-1.7B) |
|---|---|---|
| Distractor Needle Spoofing | ✗ EXPOSED (83.3%) | ✅ PASS |
| Semantic Paraphrase / Passive Voice | ✗ EXPOSED (84.7%) | ✅ PASS |
| Negation & Revocation Blindness | ✗ EXPOSED (100%) | ✅ PASS |
| Out-of-Distribution Domain Drift | ✗ EXPOSED (100%) | ✅ PASS |
| Memory Bank Attention Dilution | ⚠️ EXPOSED (>0.999 entropy) | Mitigated (α-entmax, see paper §6.4) |

Full methodology: [ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md)

---

## Repository Structure

```
├── model.py                      # System A: Differentiable RecurrentMemoryBank (PyTorch)
├── recurrent_memory_bank.py      # System A: 32 KB cross-attention slot bank
├── recurrent_bridge.py           # System A: Residual bridge injection
├── evolutionary_memory_model.py  # System A: Two-tier Draft→Verify deliberation
│
├── cognitive_memory_engine.py    # System B: Symbolic episodic distillation engine
├── hippocampus_slm_distiller.py  # System B: Neural Hippocampus (SmolLM2-1.7B)
├── async_cognitive_runtime.py    # System B: OS-style preemptive handover scheduler
│
├── run_reproducible_benchmarks.py  # Main benchmark (all 5 corpus tiers)
├── benchmark_babi_suite.py         # Standardized bAbI evaluation
├── test_adversarial_redteam.py     # Adversarial vulnerability suite
├── test_hippocampus_negation.py    # Neural Hippocampus negation tests
├── test_model.py                   # System A unit tests (forward/backward)
│
├── data/corpus/                    # Benchmark corpora (public domain texts)
├── paper.tex                       # LaTeX preprint source
└── ADVERSARIAL_AUDIT.md           # Full adversarial audit report
```

> **Model weights** (`*.pt`, `llama-3-8b-instruct/`, `smollm2-135m-instruct/`) are excluded from this repository — they are either too large for GitHub (>100 MB) or are distributed by Meta / HuggingFace under their own licenses.

---

## Citation

```bibtex
@misc{gajda2026baddeley,
  title={Bounded Recurrent Cognitive State Spaces for Infinite Context Processing
         in Large Language Models: An Empirical, Formal, and Adversarial Analysis
         of Baddeley Working Memory Architectures},
  author={Gajda, Waldemar},
  year={2026},
  doi={10.5281/zenodo.22768217},
  howpublished={Zenodo preprint},
  url={https://doi.org/10.5281/zenodo.22768217}
}
```

---

## License

- **Open-Source / Academic:** [GNU Affero General Public License v3 (AGPLv3)](LICENSE) — Section 13 mandates that network-hosted modifications must open-source their full stack.
- **Commercial:** See [LEGAL_AND_LICENSING.md](LEGAL_AND_LICENSING.md) for dual-licensing and prior-art documentation.
