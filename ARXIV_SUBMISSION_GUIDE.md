# arXiv & Zenodo Submission Guide: Step-by-Step Package

This guide provides everything needed to submit the paper to **arXiv** and claim an instant, timestamped **Zenodo DOI** before public disclosure.

---

## 1. Quick Metadata (Copy & Paste Ready)

### Title
```text
Bounded Recurrent Cognitive State Spaces for Infinite Context Processing in Large Language Models: An Empirical, Formal, and Adversarial Analysis of Baddeley Working Memory Architectures
```

### Authors
```text
Cognitive Working Memory Research Consortium (or specify your name/co-authors)
```

### Primary Subject Classification
- **Primary Category:** `cs.CL` - Computation and Language

### Cross-List Classifications
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning

### License Recommendation
- **arXiv.org perpetual, non-exclusive license** (Standard arXiv default)
- *Alternative:* Creative Commons Attribution 4.0 International (CC-BY 4.0)

---

## 2. Abstract (Plain Text for arXiv Web Form)

```text
Autoregressive Large Language Models (LLMs) face fundamental physical limits when scaling to long contexts due to the linear O(N) memory growth of the causal Key-Value (KV) cache and the quadratic O(N^2) computational complexity of dense self-attention. For a sequence length of 110 million tokens, the standard bfloat16 KV-cache of an 8-billion parameter model with Grouped Query Attention (Meta-Llama-3-8B) demands approximately 14.04 to 14.38 Terabytes of high-bandwidth memory (VRAM)—an infrastructure requirement exceeding 176 enterprise NVIDIA H100 (80GB) GPUs for a single inference stream. Furthermore, unconstrained context expansion induces severe cognitive degradation, including catastrophic attention dilution and the "lost-in-the-middle" phenomenon.

In this work, we present a comprehensive formalization, empirical scaling evaluation, and adversarial security audit of an alternative paradigm: Baddeley Cognitive Working Memory Architectures. Grounded in the classical quadripartite working memory model of Baddeley and Hitch (1974, 2000), our architecture replaces unbounded linear history with a strictly bounded recurrent state space S = <S_visuo, S_phon, S_exec, S_ep>, comprising a continuous Visuospatial Sketchpad (K=8 scene slots, ~32 KB), a discrete Phonological Loop (L_sym <= 16 symbol tokens, 32 bytes), a Central Executive workspace (P=4 deliberation tokens), and a multimodal Episodic Buffer (C <= 28 tokens).

We prove that this architecture guarantees strict O(1) space complexity and O(1) per-token decoding latency. Across a five-tier empirical scaling benchmark spanning 1.6k -> 20k -> 1.53M -> 10.97M -> 109.7M tokens, the system achieves 100% factual recall while maintaining a flat hardware footprint (delta RSS <= 3.1 MB across 110 million tokens). We formally document and correct a historical 1,000x unit reporting discrepancy in early literature (0.08 GB mislabeled as 0.08 MB), proving an exact hardware KV-cache reduction factor exceeding 182,900x (from 14.04 TB down to 78.6 MB on LLaMA-3-8B, and 32 KB on tensor-level slot memory banks).

Finally, we report an exhaustive zero-trust adversarial red-team audit exposing critical failure modes in heuristic text-distillation prototypes (distractor needle spoofing, syntactic collapse, negation blindness, and out-of-distribution drift), delineate the boundary between symbolic prompt distillation and neural tensor recurrence, establish five concrete architectural mitigations, and propose an open-source intellectual property strategy governed by the GNU Affero General Public License v3 (AGPLv3) paired with commercial dual-licensing.
```

---

## 3. Two Publishing Routes: Zenodo (Instant) + arXiv (Peer Moderation)

### Route A: Zenodo Instant DOI (Recommended First Step — 3 Minutes)
Zenodo (CERN / OpenAIRE) gives an **immediate, immutable, timestamped DOI** without waiting for arXiv moderators.

1. Go to [https://zenodo.org](https://zenodo.org) and log in with GitHub or ORCID.
2. Click **New Upload**.
3. Resource type: **Publication -> Preprint** (or **Software** if publishing code).
4. Upload `paper.html` or compiled PDF of `paper.tex`.
5. Paste Title, Author, and Abstract from Section 1 & 2 above.
6. Click **Publish**.
7. **Result:** You receive an immediate permanent DOI: `10.5281/zenodo.XXXXXXX`. This legally anchors your prior art timestamp worldwide.

---

### Route B: arXiv Submission (Official Academic Archive)

#### Method 1: Via Overleaf (Zero Setup)
1. Open [https://www.overleaf.com](https://www.overleaf.com) and create a **New Project -> Blank Project**.
2. Copy the contents of `paper.tex` into the project's `main.tex`.
3. Click **Recompile** to verify compilation.
4. Click **Submit** in the Overleaf top bar -> select **arXiv**.
5. Overleaf packages and transfers the files directly to arXiv.

#### Method 2: Direct arXiv Web Upload
1. Package the LaTeX file into a `.tar.gz`:
   ```bash
   cd /Users/razor/teamwork_projects/recurrent_model_test
   tar -czvf arxiv_package.tar.gz paper.tex
   ```
2. Log in to [https://arxiv.org/submit](https://arxiv.org/submit).
3. Select **cs.CL** as Primary Classification.
4. Upload `arxiv_package.tar.gz`.
5. arXiv's AutoTeX will compile `paper.tex` and display a PDF preview.
6. Verify preview, review metadata, and click **Submit**.
7. Papers submitted before 14:00 EST announce the next day at 20:00 EST.

---

## 4. Endorsement Notice
If submitting to `cs.CL` or `cs.AI` for the first time on a new arXiv account, arXiv may ask for an endorsement from an existing author. If an endorsement is needed:
- Submit to **Zenodo** first to immediately secure the timestamp and DOI.
- Or request an endorsement link from any colleague who has published in `cs.CL` within the last 5 years.
