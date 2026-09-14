#!/usr/bin/env python3
"""
Hippocampus SLM Distiller — Neural Working Memory Extractor
============================================================
Implements the Dual-LLM Baddeley Cognitive Architecture:

    [Raw Text Stream]
         │
         ▼
  ┌─────────────────────┐
  │    HIPPOCAMPUS      │  ← SmolLM2-1.7B-Instruct (auto-selected)
  │   (Sensory SLM)     │    Reads chunks, understands negation /
  │                     │    paraphrase / revocation → emits assertions
  └──────────┬──────────┘
             │  Compact structured facts  (JSON)
             ▼
  ┌─────────────────────┐
  │   WORKING MEMORY    │  ← BaddeleyWorkingMemory (O(1), entity-scoped)
  │  (Episodic Buffer)  │    Bounded ~300-token episodic prompt
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │   CORTEX (LLM)      │  ← LLaMA-3-8B / GPT-4 / Claude
  │   (Executive)       │    Reasons ONLY over WM snapshot
  └─────────────────────┘

Properties vs. regex-based System B:
  ✓  Negation-aware    ("offer X was REJECTED" → do NOT store X)
  ✓  Paraphrase-aware  ("market cap" == "valuation")
  ✓  Revocation-aware  ("contract cancelled" → empty)
  ✓  Language-agnostic (Polish, English, German in same stream)
  ✓  Entity-scoped WM  (Parent Corp ≠ Subsidiary Corp attributes)

Model auto-selection:
  Priority 1: HuggingFaceTB/SmolLM2-1.7B-Instruct  (HF cache, ~3.5 GB)
  Priority 2: ./smollm2-135m-instruct               (local, 269 MB)

Author: Waldemar Gajda
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Model auto-selection
# ─────────────────────────────────────────────────────────────────────────────

_LOCAL_135M = Path(__file__).parent / "smollm2-135m-instruct"
_HF_1B7_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def _resolve_model() -> str:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    cached = hf_cache / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
    if cached.exists() and any(cached.rglob("*.safetensors")):
        return _HF_1B7_ID
    return str(_LOCAL_135M)


HIPPOCAMPUS_MODEL_PATH: str = _resolve_model()

# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt  (tuned empirically for SmolLM2-1.7B)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a semantic fact extractor. Extract confirmed facts from text as a JSON array.

Format: [{"entity": "<subject>", "attribute": "<property>", "value": "<fact>"}]

RULES (apply strictly):
1. INCLUDE only facts that are CONFIRMED / FINALIZED / ASSERTED.
   Examples: "agreed at X", "revised to X", "stands at X", "the value is X"
2. EXCLUDE facts that are REJECTED or REFUSED.
   Examples: "offer of X was REJECTED", "X was REFUSED", "proposal of X refused"
3. EXCLUDE facts that are CANCELLED / NULLIFIED.
   Examples: "Tranche A cancelled", "nullified", "all obligations cease"
4. EXCLUDE facts that are CONDITIONAL or PENDING (not yet decided).
   Examples: "parties consider raising to X", "decision pending", "proposed but not confirmed"
5. When a value is OVERRIDDEN, keep only the NEW value.
   Example: "initial value 10M, amended to 52M, superseding all previous" → keep 52M only
6. Output ONLY the JSON array on one line. No explanation. No markdown fences.
7. If no facts qualify: []

Examples:
User: "The pre-money valuation is agreed at 45M EUR. An earlier offer of 12M EUR was rejected."
Assistant: [{"entity": "Deal", "attribute": "pre_money_valuation", "value": "45M EUR"}]

User: "Tranche A is hereby cancelled and nullified. All obligations cease."
Assistant: []

User: "Parties are considering raising capital to 5M USD. Decision not yet made."
Assistant: []

User: "The production key is TITAN-KEY-9901-X. Draft key DRAFT-001 was deprecated."
Assistant: [{"entity": "Authorization", "attribute": "production_key", "value": "TITAN-KEY-9901-X"}]

User: "Initial contract value was 10M EUR. An amendment revised it to 52.75M EUR, superseding all previous figures."
Assistant: [{"entity": "Contract", "attribute": "value", "value": "52.75M EUR"}]\
"""

# Negation / cancellation safety filter (post-processing fallback).
# Uses stems WITHOUT trailing \b so inflected forms match:
#   cancel → cancelled/cancellation, reject → rejected/rejection,
#   consider → considering, nullif → nullified, ceas → ceased/ceases
_NEGATION_PATTERNS = re.compile(
    r"\b(reject|refus|cancel|nullif|revok|ceas|"
    r"pending|not yet|consider|hypothetical|draft|deprecat|proposed but)",
    re.IGNORECASE,
)

MAX_NEW_TOKENS = 256
CHUNK_TOKEN_LIMIT = 1024  # system prompt ~400 tok + user chunk up to 600 tok


# ─────────────────────────────────────────────────────────────────────────────
# Baddeley Working Memory  — O(1) Episodic Buffer
# ─────────────────────────────────────────────────────────────────────────────


class BaddeleyWorkingMemory:
    """Entity-scoped episodic state (O(1) bounded by entity/attribute count)."""

    def __init__(self, max_facts: int = 200):
        self.state: dict[str, dict[str, str]] = {}
        self.max_facts = max_facts
        self.update_count = 0

    def apply_assertions(self, assertions: list[dict]) -> int:
        applied = 0
        for a in assertions:
            entity = str(a.get("entity", "Global")).strip()
            attr = (
                str(a.get("attribute", "fact")).strip().lower().replace(" ", "_")
            )
            value = str(a.get("value", "")).strip()
            if not entity or not value:
                continue
            # Capacity guard
            if self.fact_count() >= self.max_facts and entity not in self.state:
                continue
            self.state.setdefault(entity, {})[attr] = value
            applied += 1
        self.update_count += 1
        return applied

    def revoke(self, entity: str, attribute: Optional[str] = None) -> None:
        if entity in self.state:
            if attribute:
                self.state[entity].pop(attribute, None)
            else:
                del self.state[entity]

    def to_prompt_buffer(self) -> str:
        if not self.state:
            return "[WORKING MEMORY: empty]"
        lines = ["[WORKING MEMORY — Baddeley Episodic Buffer]"]
        for entity, attrs in self.state.items():
            for attr, value in attrs.items():
                lines.append(f"  - {entity} | {attr}: {value}")
        return "\n".join(lines)

    def fact_count(self) -> int:
        return sum(len(v) for v in self.state.values())

    def entity_count(self) -> int:
        return len(self.state)

    def token_estimate(self) -> int:
        return len(self.to_prompt_buffer()) // 4

    def __repr__(self) -> str:
        return (
            f"<BaddeleyWM facts={self.fact_count()} "
            f"entities={self.entity_count()} "
            f"~{self.token_estimate()} tokens>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HippocampusSLM — Neural Distiller
# ─────────────────────────────────────────────────────────────────────────────


class HippocampusSLM:
    """
    Neural working memory distiller (SmolLM2-1.7B-Instruct by default).

    Replaces regex-based System B with language understanding.
    Naturally handles negation, paraphrase, revocation, multi-entity scoping.
    """

    def __init__(
        self,
        model_path: str | Path = HIPPOCAMPUS_MODEL_PATH,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_path = str(model_path)
        self.verbose = verbose

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.verbose:
            print(f"[Hippocampus] Model  : {self.model_path}")
            print(f"[Hippocampus] Device : {self.device}")

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # device_map works on CUDA; for MPS / CPU we .to(device) manually
        if self.device in ("mps", "cpu"):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.bfloat16
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.bfloat16, device_map=self.device
            )
        self.model.eval()

        if self.verbose:
            params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(
                f"[Hippocampus] Loaded {params_m:.0f}M params in "
                f"{time.time() - t0:.2f}s"
            )

        self.working_memory = BaddeleyWorkingMemory()
        self._total_chunks = 0
        self._total_facts = 0
        self._total_inference_ms = 0.0

    # ── Inference ─────────────────────────────────────────────────────────────

    def _extract_assertions(self, text_chunk: str) -> list[dict]:
        """Run one chunk through the SLM, return parsed assertion list."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_chunk},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=CHUNK_TOKEN_LIMIT,
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.05,
            )
        self._total_inference_ms += (time.time() - t0) * 1000

        input_len = inputs["input_ids"].shape[1]
        raw = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        return self._parse_and_filter(raw, text_chunk)

    def _parse_and_filter(self, raw: str, source_text: str) -> list[dict]:
        """
        Parse JSON from model output, then apply safety post-processing:
          1. Skip assertions whose *value* directly matches a negated phrase
             in the source text (safety net for prompt non-compliance).
        """
        assertions = self._parse_json(raw)
        if not assertions:
            return []

        filtered = []
        for a in assertions:
            value = str(a.get("value", ""))
            attr = str(a.get("attribute", ""))
            # If the value appears in the source alongside a negation marker,
            # drop it (defence-in-depth against prompt non-compliance)
            if value and _value_negated_in_source(value, source_text):
                continue
            filtered.append(a)
        return filtered

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        """Robust JSON extraction from raw model output."""
        # 1. Direct parse
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # 2. First [...] block (handles surrounding prose)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        # 3. Collect individual {...} objects
        objects = []
        for m in re.finditer(r"\{[^{}]+\}", raw):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "value" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                pass
        return objects

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_chunk(self, text_chunk: str) -> dict:
        """Process one text chunk; update working memory in-place."""
        assertions = self._extract_assertions(text_chunk)
        applied = self.working_memory.apply_assertions(assertions)
        self._total_chunks += 1
        self._total_facts += applied
        return {
            "assertions": assertions,
            "applied": applied,
            "wm_facts": self.working_memory.fact_count(),
            "wm_entities": self.working_memory.entity_count(),
        }

    def ingest_stream(
        self, chunks: list[str], show_progress: bool = True
    ) -> dict:
        """Process a list of chunks sequentially (O(1) footprint)."""
        t_start = time.time()
        n = len(chunks)
        for i, chunk in enumerate(chunks):
            result = self.ingest_chunk(chunk)
            if show_progress and (i % max(1, n // 10) == 0 or i == n - 1):
                avg_ms = self._total_inference_ms / (i + 1)
                print(
                    f"  [Hippocampus] Chunk {i+1:4d}/{n} "
                    f"| WM facts: {result['wm_facts']:3d} "
                    f"| avg {avg_ms:.0f} ms/chunk"
                )
            del chunk
            gc.collect()

        return {
            "total_chunks": self._total_chunks,
            "total_facts_extracted": self._total_facts,
            "wm_snapshot": self.working_memory.to_prompt_buffer(),
            "wm_facts": self.working_memory.fact_count(),
            "wm_entities": self.working_memory.entity_count(),
            "wm_token_estimate": self.working_memory.token_estimate(),
            "total_wall_time_s": time.time() - t_start,
            "avg_inference_ms": (
                self._total_inference_ms / max(1, self._total_chunks)
            ),
        }

    def reset_memory(self) -> None:
        self.working_memory = BaddeleyWorkingMemory()
        self._total_chunks = 0
        self._total_facts = 0
        self._total_inference_ms = 0.0

    def get_working_memory_prompt(self) -> str:
        return self.working_memory.to_prompt_buffer()

    def stats(self) -> str:
        avg = self._total_inference_ms / max(1, self._total_chunks)
        return (
            f"Hippocampus Stats | "
            f"Chunks: {self._total_chunks} | "
            f"Facts: {self._total_facts} | "
            f"WM: {self.working_memory} | "
            f"Avg: {avg:.0f} ms/chunk"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety post-processor: context-aware negation check
# ─────────────────────────────────────────────────────────────────────────────


def _value_negated_in_source(value: str, source: str) -> bool:
    """
    Return True only if `value` appears in a sentence that *also* contains
    a negation / cancellation marker — strictly sentence-scoped.

    Sentence-level scope prevents cross-sentence contamination, e.g.:
      "DRAFT-001 was deprecated. The key is TITAN-001."
      → TITAN-001 is in a sentence with no negation → NOT filtered.

    Normalization: commas and spaces stripped from both value and source
    so "5,000,000" matches "5000000".
    """
    # Split into sentences at .!? boundaries
    sentences = re.split(r'(?<=[.!?])\s+', source)
    val_norm = value.lower().replace(",", "").replace(" ", "")
    if not val_norm:
        return False

    for sent in sentences:
        sent_norm = sent.lower().replace(",", "").replace(" ", "")
        # Check if (normalized) value appears in this sentence
        if val_norm[:20] in sent_norm:
            # Now check if the SAME sentence has a negation marker
            if _NEGATION_PATTERNS.search(sent):
                return True
    return False
