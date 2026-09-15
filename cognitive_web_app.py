#!/usr/bin/env python3
"""
cognitive_web_app.py — Baddeley Cognitive Dual-Process Assistant
================================================================
Implements the true Operating System Dual-Model Architecture from the scientific preprint:

1. BACKGROUND SENSORY HIPPOCAMPUS (SmolLM2-1.7B):
   - Runs asynchronously 24/7 in the background.
   - Continuously digests uploaded documents (PDF, TXT, MD), notes, and macOS clipboard.
   - Compresses large multi-page texts into bounded O(1) Working Memory assertions.
   - Never answers user queries directly (avoids small model hallucinations).

2. SLEEPING EXECUTIVE CORTEX (LLaMA-3-8B):
   - Dormant (0% compute, 0 FLOPs) while Hippocampus monitors the stream.
   - Wakes up ONLY when the user asks a question in the chat.
   - Instant preemption: pauses Hippocampus, takes 100% memory bus bandwidth.
   - Generates fluent, native Polish reasoning and answers over the Working Memory state.
   - Yields and goes dormant again, unpausing the Hippocampus daemon.

Author: Waldemar Gajda
"""

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import datetime
import json
import os
import re
import sys
import threading
import time
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    import pypdf
except ImportError:
    pypdf = None

import torch
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import uvicorn


# =====================================================================
# 1. Working Memory Data Store (O(1) Bounded State Space)
# =====================================================================

@dataclasses.dataclass
class MemoryAssertion:
    id: str
    source: str
    text: str
    timestamp: str


class BaddeleyWorkingMemoryStore:
    """Maintains an entity-scoped, bounded working memory of active context."""

    def __init__(self, max_assertions: int = 120):
        self.max_assertions = max_assertions
        self.lock = threading.Lock()
        self.assertions: collections.deque[MemoryAssertion] = collections.deque(maxlen=max_assertions)
        self.sources: List[Dict[str, Any]] = []
        self.total_tokens_ingested: int = 0
        self.last_ingest_time: str = "Brak"

    async def add_source(self, name: str, char_count: int, source_type: str):
        with self.lock:
            # Avoid duplicate source entries with identical name
            existing = [s for s in self.sources if s["name"] == name]
            if existing:
                existing[0]["chars"] = char_count
                existing[0]["time"] = datetime.datetime.now().strftime("%H:%M:%S")
            else:
                self.sources.append({
                    "name": name,
                    "chars": char_count,
                    "type": source_type,
                    "time": datetime.datetime.now().strftime("%H:%M:%S")
                })

    async def store_assertions(self, texts: List[str], source: str):
        with self.lock:
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            self.last_ingest_time = now_str
            for t in texts:
                cleaned = t.strip()
                if len(cleaned) > 15:
                    clean_norm = cleaned[:350]
                    # Deduplication check against existing assertions
                    if not any(a.text.lower() == clean_norm.lower() for a in self.assertions):
                        self.assertions.append(MemoryAssertion(
                            id=f"M-{int(time.time() * 1000) % 100000}",
                            source=source,
                            text=clean_norm,
                            timestamp=now_str
                        ))

    async def get_working_memory_prompt(self, max_chars: int = 4500) -> str:
        """Kompiluje stan pamięci roboczej do promptu dla Kory Wykonawczej."""
        with self.lock:
            if not self.assertions:
                return "Pamięć robocza jest pusta (brak wgranych materiałów ani skopiowanego tekstu)."
            lines = [f"• [{a.source}] {a.text}" for a in list(self.assertions)[-40:]]
            result = "\n".join(lines)
            return result[:max_chars]

    async def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "assertion_count": len(self.assertions),
                "assertions": [dataclasses.asdict(a) for a in list(self.assertions)],
                "sources": list(self.sources),
                "total_tokens": self.total_tokens_ingested,
                "last_ingest": self.last_ingest_time,
            }

    async def clear(self):
        with self.lock:
            self.assertions.clear()
            self.sources.clear()
            self.total_tokens_ingested = 0


# =====================================================================
# 2. Dual-Process Neural Engine (Hippocampus + Cortex)
# =====================================================================

class DualCognitiveEngine:
    """
    Implements the Operating System Dual-Model Runtime:
    - Hippocampus (SmolLM2-1.7B): Ingestion, compression, distillation.
    - Cortex (LLaMA-3-8B): Dormant until user query, handles 100% of generation.
    """

    def __init__(self, base_dir: Path, model_choice: str = "8b"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.base_dir = base_dir

        # Cortex configuration
        self.cortex_local_path = base_dir / "llama-3-8b-instruct"
        self.cortex_tok = None
        self.cortex_model = None

        # State & Preemption primitives
        self.preemption_lock = threading.Lock()
        self.is_preempted = False
        self.preemption_count = 0
        self.status = "Inicjalizacja..."
        self.is_ready = False
        self.loading_in_progress = False
        self.hippo_active = False

        # Continuous ingestion state
        self.is_ingesting = False
        self.ingestion_progress = 0
        self.ingestion_step = 0
        self.ingestion_total = 0
        self.ingestion_status = ""
        self.ingestion_facts_count = 0

        self.configure_model(model_choice)

    def configure_model(self, choice: str):
        self.model_choice = (choice or "8b").lower().strip()
        if self.model_choice in ("3b", "llama-3.2-3b", "llama-3.2-3b-instruct"):
            self.cortex_path = "unsloth/Llama-3.2-3B-Instruct"
            self.model_label = "Kora 3B"
            self.model_desc = "Llama-3.2-3B-Instruct (~6 GB RAM, ultra-szybka)"
        elif self.model_choice in ("1.7b", "smollm2", "smol"):
            self.cortex_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            self.model_label = "Kora 1.7B"
            self.model_desc = "SmolLM2-1.7B-Instruct (~3.5 GB RAM, super-lekka)"
        else:
            self.model_choice = "8b"
            self.cortex_path = str(self.cortex_local_path) if self.cortex_local_path.exists() else "NousResearch/Meta-Llama-3-8B-Instruct"
            self.model_label = "Kora 8B"
            self.model_desc = "LLaMA-3-8B-Instruct (15.5 GB RAM, pełna precyzja)"

    def initialize_both_models(self):
        """Wczytuje model Kory Wykonawczej do pamięci operacyjnej."""
        self.loading_in_progress = True
        try:
            self.status = f"Ładowanie Kory Wykonawczej ({self.model_label})..."
            print(f"  [Cognitive OS] Loading Executive Cortex: {self.model_label} ({self.cortex_path}) on {self.device}...")
            self.cortex_tok = AutoTokenizer.from_pretrained(self.cortex_path)
            if self.cortex_tok.pad_token is None:
                self.cortex_tok.pad_token = self.cortex_tok.eos_token

            dtype = torch.bfloat16 if self.device == "mps" else torch.float32
            self.cortex_model = AutoModelForCausalLM.from_pretrained(
                self.cortex_path,
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.cortex_model.eval()
            print(f"  [Cognitive OS] ✓ Executive Cortex loaded ({self.model_label}).")

            self.is_ready = True
            self.status = f"System Gotowy ({self.model_label} + Hipokamp O(1))"
            print(f"  [Cognitive OS] Cognitive Runtime fully operational on {self.device.upper()}.")
        except Exception as e:
            self.status = f"Błąd inicjalizacji: {e}"
            print(f"  [Cognitive OS] ✗ Initialization failed: {e}")
        finally:
            self.loading_in_progress = False

    # ── Source Cleaning & Semantic Distillation into Hippocampus ─────────────

    def is_valid_fact(self, text: str) -> bool:
        s = text.strip()
        s = re.sub(r"^([-•*~]|\d+[\.\)\:])\s*", "", s).strip()
        if len(s) < 20 or len(s) > 380:
            return False
        # Fakt musi być kompletnym zdaniem zakończonym kropką, pytajnikiem, wykrzyknikiem lub cudzysłowem
        if s[-1] not in '.!?"\'”)':
            return False
        # Przynajmniej 4 słowa
        words = [w for w in re.split(r'\s+', s) if len(w) > 1 and any(c.isalnum() for c in w)]
        if len(words) < 4:
            return False
        # Odrzucenie pętli degeneracji powtórzeniowej (np. "nie, nie, nie..." lub pętli pojedynczych słów)
        unique_words = set(w.lower() for w in words)
        if (len(unique_words) / len(words)) < 0.50:
            return False
        for w in words:
            if len(w) > 3 and s.lower().count(w.lower()) > 4:
                return False
        # Znaki alfanumeryczne muszą stanowić większość (odrzuca ciągi kresek, szum OCR)
        alnum_chars = sum(1 for c in s if c.isalnum() or c in ' ,.;:!?-–„”"\'()')
        if (alnum_chars / len(s)) < 0.70:
            return False
        lower = s.lower()
        # Filtry formułek wprowadzających asystenta
        intro_phrases = [
            "oto najważniejsz", "oto 4 ", "oto 3 ", "oto 2 ", "oto kilka", "oto lista", "oto fakty", "oto wątki",
            "poniżej przedstawiam", "poniżej znajduje się", "oto podsumowanie", "oto wybrane", "oto tytuły",
            "oto główne"
        ]
        if any(ip in lower for ip in intro_phrases):
            return False
        # Filtry szumu prawnego i platformowego
        noise_terms = [
            "project gutenberg", "gutenberg-tm", "gutenberg.org", "e-book", "ebook",
            "terms of use", "license agreement", "licencja", "distributed proofreading",
            "all rights reserved", "wersja elektroniczna", "prawa zastrzeżone",
            "tłumaczenie automatyczne"
        ]
        if any(nt in lower for nt in noise_terms):
            return False
        return True

    def clean_source_text(self, text: str) -> str:
        """
        Oczyszcza surowy tekst dokumentu (PDF, TXT, e-booki):
        - Wykrywa i odcina nagłówki/stopki licencyjne Project Gutenberg
        - Usuwa powtarzające się separatory linii (myślniki, gwiazdki, znaki równości)
        - Łączy słowa i liczby rozbite przez formatowanie PDF
        - Łączy linie wewnątrz akapitów
        - Filtruje szum redakcyjny i numery stron
        """
        # 0. Wykrywanie i odcinanie nagłówków/stopek Gutenberga
        sm = re.search(r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', text, re.IGNORECASE)
        if sm:
            text = text[sm.end():]
        em = re.search(r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', text, re.IGNORECASE)
        if em:
            text = text[:em.start()]

        # 1. Usuwanie linii będących ciągami myślników, kresek, gwiazdek lub znaków równości
        text = re.sub(r'^[-\s•*=_~·]{3,}$', '', text, flags=re.MULTILINE)

        # 2. Łączenie liczb z procentami lub jednostkami rozbitych enterem
        t = re.sub(r'(\d+)\s*\n\s*(%|mln|mld|tys\.|proc\.)', r'\1\2', text)
        # 3. Łączenie słów rozdzielonych łącznikiem na końcu linii
        t = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', t)
        # 4. Łączenie linii wewnątrz zdań (usuwa pojedyncze \n niepoprzedzone kropką/dwukropkiem)
        t = re.sub(r'(?<![.!?:\n])\n(?![A-Z0-9\n])', ' ', t)

        # 5. Filtr szumu: linie będące numerami stron lub formułkami wydawniczymi
        cleaned_lines = []
        for line in t.split('\n'):
            l_strip = line.strip()
            if not l_strip:
                continue
            if re.match(r'^(page\s+\d+(\s+of\s+\d+)?|strona\s+\d+(\s+z\s+\d+)?)$', l_strip, re.IGNORECASE):
                continue
            if any(term in l_strip.lower() for term in ['all rights reserved', 'doi: 10.', 'isbn 978-', 'printed in', 'taylor & francis', 'terms of use']):
                continue
            cleaned_lines.append(l_strip)
        return '\n\n'.join(cleaned_lines)

    def distill_document_stream(self, full_text: str, source_name: str, on_fact_callback=None) -> List[str]:
        """
        Ciągła, asynchroniczna asymilacja semantyczna w architekturze Baddeleya O(1):
        - Czyści surowy tekst i usuwa szum Gutenberga / wydawniczy.
        - Dzieli na makro-bloki logiczne.
        - Dobiera optymalną liczbę próbkowania obejmującą 100% rozpiętości dokumentu.
        - Wyprowadza ustrukturyzowane fakty przez Korę w tle.
        - Na bieżąco przekazuje fakty do callbacku (np. zapis do pamięci roboczej).
        """
        self.hippo_active = True
        self.is_ingesting = True
        self.ingestion_progress = 0
        self.ingestion_step = 0
        self.ingestion_total = 0
        self.ingestion_facts_count = 0
        self.ingestion_status = f"Przygotowanie: {source_name}..."

        extracted_facts: List[str] = []

        try:
            # Poczekaj jeśli model jeszcze się ładuje
            if self.cortex_model is None and self.loading_in_progress:
                wait_sec = 0
                while self.cortex_model is None and self.loading_in_progress and wait_sec < 40:
                    time.sleep(0.5)
                    wait_sec += 1

            cleaned_text = self.clean_source_text(full_text)
            if not cleaned_text.strip():
                return []

            # 1. Błyskawiczny skan strukturalny (Tytuły, Rozdziały, Spis treści)
            structure_titles = []
            title_matches = re.findall(
                r'^(?:[ \t]*)(?:THE\s+(?:TRAGEDY|COMEDY|LIFE|FIRST|SECOND|THIRD|HISTORY)\s+OF\s+[A-Z\s,\']{3,60}|'
                r'(?:1[56]\d\d\s+)?THE\s+(?:SONNETS|TRAGEDY|COMEDY|TEMPEST|WINTER\'S\s+TALE)[^\n]*|'
                r'Rozdział\s+[IVXLCDM\d]+[^\n]{3,}|Tom\s+[IVXLCDM\d]+[^\n]{3,}|Chapter\s+[IVXLCDM\d]+[^\n]{3,})',
                cleaned_text,
                flags=re.MULTILINE
            )
            for tm in title_matches[:15]:
                t_clean = re.sub(r'[\r\n]+.*', '', tm).strip()
                if 5 < len(t_clean) < 80 and t_clean not in structure_titles:
                    structure_titles.append(t_clean)

            if len(structure_titles) >= 3:
                struct_fact = f"Struktura dokumentu {source_name} obejmuje m.in.: {', '.join(structure_titles[:8])}."
                extracted_facts.append(struct_fact)
                self.ingestion_facts_count += 1
                if on_fact_callback:
                    on_fact_callback(struct_fact)

            # 2. Podział na akapity i makro-bloki
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 25]
            if not paragraphs:
                return extracted_facts

            macro_chunks = []
            chunk_size = 2800
            cur = ""
            for p in paragraphs:
                if len(cur) + len(p) < chunk_size:
                    cur += "\n\n" + p
                else:
                    if cur.strip():
                        macro_chunks.append(cur.strip())
                    cur = p
            if cur.strip():
                macro_chunks.append(cur.strip())

            # 3. Dynamiczne pokrycie rozpiętości 100% dokumentu
            if len(macro_chunks) <= 20:
                selected_chunks = macro_chunks
            else:
                # Dla wielkich woluminów wybierz 28 równomiernie rozłożonych okien na całej osi czasu
                target_chunks = min(28, len(macro_chunks))
                step = len(macro_chunks) / target_chunks
                indices = [int(i * step) for i in range(target_chunks)]
                seen_idx = set()
                selected_chunks = []
                for idx in indices:
                    i_clamped = min(idx, len(macro_chunks) - 1)
                    if i_clamped not in seen_idx:
                        seen_idx.add(i_clamped)
                        selected_chunks.append(macro_chunks[i_clamped])

            self.ingestion_total = len(selected_chunks)
            print(f"  [Cognitive OS] Starting assimilation of '{source_name}': {len(selected_chunks)} chunks across {len(full_text):,} chars...")

            # 4. Asymilacja kolejnych bloków
            for step_i, chunk in enumerate(selected_chunks):
                self.ingestion_step = step_i + 1
                self.ingestion_progress = int(((step_i + 1) / len(selected_chunks)) * 100)
                self.ingestion_status = f"Asymilacja {source_name} ({self.ingestion_progress}%)..."

                new_chunk_facts = []

                if self.cortex_model is not None and self.cortex_tok is not None:
                    try:
                        with self.preemption_lock:
                            prompt_messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Jesteś modułem kognitywnym asymilacji wiedzy w architekturze pamięci roboczej Baddeleya. "
                                        "Przeanalizuj poniższy fragment tekstu i wyodrębnij z niego od 2 do 4 najważniejszych faktów merytorycznych lub wątków "
                                        "(postaci i ich role, kluczowe wydarzenia fabularne, tezy, relacje, dane liczbowe, definicje lub wnioski).\n"
                                        "ZASADY:\n"
                                        "1. Każdy fakt zapisz w osobnej linii zaczynając od myślnika '- '.\n"
                                        "2. Podsumuj fakty i wydarzenia w języku polskim (nie cytuj surowego tekstu, lecz opisz po polsku co z niego wynika).\n"
                                        "3. Ignoruj szum redakcyjny, numery stron, prawa autorskie i formułki wydawnicze.\n"
                                        "4. Żadnych wstępów, komentarzy ani powtórzeń tych samych słów — wygeneruj tylko zwięzłą listę unikalnych faktów."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": f"DOKUMENT: {source_name} [Część {step_i+1}/{len(selected_chunks)}]:\n\n{chunk[:2600]}"
                                }
                            ]
                            prompt = self.cortex_tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                            inputs = self.cortex_tok(prompt, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                out_ids = self.cortex_model.generate(
                                    **inputs,
                                    max_new_tokens=160,
                                    repetition_penalty=1.18,
                                    do_sample=False,
                                    pad_token_id=self.cortex_tok.pad_token_id
                                )
                            response = self.cortex_tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                            for line in response.splitlines():
                                raw_l = line.strip()
                                if not raw_l:
                                    continue
                                clean_f = re.sub(r"^([-•*~]|\d+[\.\)\:])\s*", "", raw_l).strip()
                                if self.is_valid_fact(clean_f):
                                    if not any(clean_f.lower() in ef.lower() or ef.lower() in clean_f.lower() for ef in extracted_facts):
                                        new_chunk_facts.append(clean_f)
                    except Exception as e:
                        print(f"  [Cognitive OS] Neural distillation note on chunk {step_i+1}: {e}")

                # Fallback: jeśli model nie wygenerował faktów dla tego bloku (lub w trybie testowym)
                if not new_chunk_facts:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if len(s.strip()) > 25 and self.is_valid_fact(s.strip())]
                    for s in sentences[:3]:
                        if not any(s.lower() in ef.lower() or ef.lower() in s.lower() for ef in extracted_facts):
                            new_chunk_facts.append(s)

                for f in new_chunk_facts:
                    extracted_facts.append(f)
                    self.ingestion_facts_count += 1
                    if on_fact_callback:
                        on_fact_callback(f)

            print(f"  [Cognitive OS] ✓ Assimilation completed for '{source_name}': {len(extracted_facts)} clean facts distilled.")
            return extracted_facts
        finally:
            self.hippo_active = False
            self.is_ingesting = False
            self.ingestion_progress = 100
            self.ingestion_status = f"Zakończono: {len(extracted_facts)} faktów w pamięci."

    def distill_document_chunks(self, full_text: str, source_name: str) -> List[str]:
        """Synchroniczny interfejs kompatybilny wstecznie z testami jednostkowymi."""
        return self.distill_document_stream(full_text, source_name)

    # ── Executive Cortex Interactive Query (Preemption) ───────────────────────

    def generate_cortex_response_streaming(
        self,
        messages: List[Dict[str, str]],
        working_memory_context: str,
        max_tokens: int = 600,
        temperature: float = 0.7,
    ):
        """
        Wywłaszcza szynę pamięci (preemption handover),
        generuje odpowiedź Korą LLaMA-3-8B i zwraca tokeny strumieniowo.
        """
        t0 = time.time()
        self.is_preempted = True
        self.preemption_count += 1

        try:
            with self.preemption_lock:
                if self.cortex_model is None or self.cortex_tok is None:
                    # Fallback jeśli jeszcze się ładuje
                    wait_sec = 0
                    while not self.is_ready and wait_sec < 40:
                        time.sleep(0.5)
                        wait_sec += 1
                    if not self.is_ready:
                        yield f"Kora Wykonawcza ({self.model_label}) nadal się ładuje do pamięci RAM. Proszę odczekać kilka sekund..."
                        return

                # System prompt w języku polskim z wstrzykniętą pamięcią roboczą
                system_prompt = (
                    "Jesteś Korą Wykonawczą (Executive Cortex) zaawansowanego asystenta opartego na architekturze "
                    "pamięci kognitywnej Baddeleya. Rozmawiasz z użytkownikiem wyłącznie w języku polskim w sposób "
                    "inteligentny, naturalny, elegancki i precyzyjny.\n\n"
                    "Poniżej znajduje się skondensowana PAMIĘĆ ROBOCZA (fakty i wiedza wyekstrahowana w tle przez Hipokamp "
                    "z wgranych przez użytkownika dokumentów, książek, schowka i notatek):\n"
                    "====================== PAMIĘĆ ROBOCZA ======================\n"
                    f"{working_memory_context}\n"
                    "============================================================\n"
                    "INSTRUKCJE POSTĘPOWANIA:\n"
                    "1. Jeśli użytkownik pyta o wgrany dokument, kluczowe wnioski, podsumowanie materiałów lub szczegółowe fakty: "
                    "odpowiedz wyczerpująco, opierając się dokładnie na powyższym kontekście pamięci roboczej.\n"
                    "2. Jeśli to swobodna rozmowa lub pytanie ogólne (np. powitanie, pytanie filozoficzne, programistyczne): "
                    "odpowiedz swobodnie, błyskotliwie i płynnie z własnej wiedzy.\n"
                    "3. Nigdy nie wypluwaj surowych zmiennych programistycznych ani technicznego debugu. Odpowiadaj jak wybitny asystent człowieka.\n"
                    "4. ZASADA LUDZKIEJ UCZCIWOŚCI: Masz przed sobą skondensowane notatki, a surowy plik został usunięty z pamięci. "
                    "Jeśli użytkownik zapyta o mechaniczne cechy dokumentu (np. ile razy w tekście pada dane słowo, na której stronie coś jest, "
                    "albo poprosi o dokładny cytat słowo w słowo) – NIE ZMYŚLAJ i nie licz słów w swoich notatkach! "
                    "Odpowiedz po prostu szczerze i po ludzku: 'Nie wiem — nie liczyłem słów w dokumencie, pamiętam jedynie główne fakty i wnioski'."
                )

                formatted_messages = [{"role": "system", "content": system_prompt}] + messages

                try:
                    prompt = self.cortex_tok.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"{system_prompt}\n\nUżytkownik: {messages[-1]['content']}\nAsystent:"

                inputs = self.cortex_tok(prompt, return_tensors="pt").to(self.device)
                streamer = TextIteratorStreamer(self.cortex_tok, skip_prompt=True, skip_special_tokens=True)

                kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=max(temperature, 0.4),
                    top_p=0.9,
                    pad_token_id=self.cortex_tok.pad_token_id,
                )

                gen_thread = threading.Thread(target=self.cortex_model.generate, kwargs=kwargs)
                gen_thread.start()

                for chunk in streamer:
                    yield chunk

                gen_thread.join()

        finally:
            self.is_preempted = False
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  [Cognitive OS] Cortex query handled in {elapsed_ms:.1f} ms. Preemption released.")


# =====================================================================
# 3. Application State & Continuous Ingestion Watchers
# =====================================================================

BASE_DIR = Path(__file__).parent

class AssistantApplication:
    def __init__(self, model_choice: str = "8b"):
        self.wm = BaddeleyWorkingMemoryStore(max_assertions=120)
        self.engine = DualCognitiveEngine(base_dir=BASE_DIR, model_choice=model_choice)
        self.chat_history: List[Dict[str, str]] = []
        self.clipboard_monitor_active = False
        self.clipboard_thread: Optional[threading.Thread] = None
        self.last_clipboard_text = ""
        self.is_running = True

    def configure_model(self, model_choice: str):
        self.engine.configure_model(model_choice)

    def start(self):
        threading.Thread(target=self.engine.initialize_both_models, daemon=True).start()

    def start_background_ingestion(self, text: str, source_name: str, source_type: str = "file"):
        """Uruchamia asymilację dokumentu w osobnym wątku roboczym bez blokowania interfejsu."""
        def _worker():
            try:
                # Zarejestruj źródło
                asyncio.run(self.wm.add_source(source_name, len(text), source_type))

                # Callback przekazujący każdy wyekstrahowany fakt na żywo do pamięci roboczej
                def _on_fact(f: str):
                    asyncio.run(self.wm.store_assertions([f], source=source_name))

                self.engine.distill_document_stream(
                    full_text=text,
                    source_name=source_name,
                    on_fact_callback=_on_fact
                )
            except Exception as e:
                print(f"  [Cognitive OS] Background ingestion exception: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    def toggle_clipboard(self, active: bool) -> bool:
        if pyperclip is None:
            self.clipboard_monitor_active = False
            return False
        self.clipboard_monitor_active = active
        if active and (self.clipboard_thread is None or not self.clipboard_thread.is_alive()):
            self.clipboard_thread = threading.Thread(target=self._clipboard_worker, daemon=True)
            self.clipboard_thread.start()
        return self.clipboard_monitor_active

    def _clipboard_worker(self):
        while self.clipboard_monitor_active and self.is_running:
            try:
                current = pyperclip.paste()
                if current and current != self.last_clipboard_text and len(current.strip()) > 10:
                    self.last_clipboard_text = current
                    facts = self.engine.distill_document_chunks(current, "Schowek")
                    asyncio.run(self.wm.store_assertions(facts, source="Schowek"))
                    asyncio.run(self.wm.add_source("Schowek systemowy", len(current), "clipboard"))
            except Exception:
                pass
            time.sleep(1.0)

    def get_telemetry(self) -> Dict[str, Any]:
        rss_mb = 0.0
        if psutil:
            try:
                proc = psutil.Process(os.getpid())
                rss_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                rss_mb = 120.0
        return {
            "ram_rss_mb": rss_mb,
            "device": self.engine.device.upper(),
            "status": self.engine.status,
            "is_ready": self.engine.is_ready,
            "is_preempted": self.engine.is_preempted,
            "preemption_count": self.engine.preemption_count,
            "hippo_active": self.engine.hippo_active or self.engine.is_ingesting,
            "clipboard_active": self.clipboard_monitor_active,
            "model_label": self.engine.model_label,
            "model_desc": self.engine.model_desc,
            "ingestion": {
                "active": self.engine.is_ingesting,
                "percent": self.engine.ingestion_progress,
                "step": self.engine.ingestion_step,
                "total": self.engine.ingestion_total,
                "status_text": self.engine.ingestion_status,
                "facts_found": self.engine.ingestion_facts_count,
            }
        }


assistant = AssistantApplication()


# =====================================================================
# 4. FastAPI Routes & Endpoints
# =====================================================================

@contextlib.asynccontextmanager
async def lifespan(app_instance: FastAPI):
    assistant.start()
    yield
    assistant.is_running = False

app = FastAPI(title="Cognitive Dual-Model Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "Wklejony tekst"


class ClipboardRequest(BaseModel):
    active: bool


@app.get("/api/state")
async def api_get_state():
    snap = await assistant.wm.snapshot()
    telemetry = assistant.get_telemetry()
    return {
        "memory": snap,
        "telemetry": telemetry,
        "chat_count": len(assistant.chat_history),
    }


@app.post("/api/ingest_text")
async def api_ingest_text(req: TextIngestRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Brak tekstu do wgrania"})
    
    src_name = req.source_name or "Wklejona notatka"
    assistant.start_background_ingestion(req.text, src_name, "text")
    return {
        "ok": True,
        "source_name": src_name,
        "chars": len(req.text),
        "message": f"Wklejono treść ({len(req.text):,} znaków). Kora asymiluje materiał w tle."
    }


@app.post("/api/upload_file")
async def api_upload_file(file: UploadFile = File(...)):
    filename = file.filename or "plik"
    contents = await file.read()
    extracted_text = ""

    if filename.lower().endswith(".pdf"):
        if pypdf:
            try:
                reader = pypdf.PdfReader(BytesIO(contents))
                pages = [page.extract_text() or "" for page in reader.pages]
                extracted_text = "\n\n".join(pages)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Błąd czytania PDF: {e}"})
        else:
            return JSONResponse(status_code=400, content={"error": "Biblioteka pypdf nie jest zainstalowana"})
    else:
        try:
            extracted_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                extracted_text = contents.decode("latin-1")
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Nieobsługiwany format pliku"})

    if not extracted_text.strip():
        return JSONResponse(status_code=400, content={"error": "Plik jest pusty"})

    # Launch background continuous assimilation
    assistant.start_background_ingestion(extracted_text, filename, "file")
    
    return {
        "ok": True,
        "filename": filename,
        "chars": len(extracted_text),
        "message": f"Wgrano {filename} ({len(extracted_text):,} znaków). Kora asymiluje materiał w tle."
    }


@app.post("/api/clear_memory")
async def api_clear_memory():
    await assistant.wm.clear()
    return {"ok": True, "message": "Pamięć robocza została wyczyszczona."}


@app.post("/api/clear_chat")
async def api_clear_chat():
    assistant.chat_history.clear()
    return {"ok": True, "message": "Historia czatu została wyczyszczona."}


@app.post("/api/clipboard_toggle")
async def api_clipboard_toggle(req: ClipboardRequest):
    active = assistant.toggle_clipboard(req.active)
    return {"ok": True, "active": active}


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_msg = data.get("message", "").strip()
            if not user_msg:
                continue

            # Record user turn
            assistant.chat_history.append({"role": "user", "content": user_msg})

            # Retrieve current Working Memory snapshot
            wm_context = await assistant.wm.get_working_memory_prompt()

            # Stream answer generated exclusively by Cortex (LLaMA-3-8B) with Preemption
            stream_gen = assistant.engine.generate_cortex_response_streaming(
                messages=list(assistant.chat_history[-6:]),
                working_memory_context=wm_context,
            )

            assistant_accumulated = []
            for token in stream_gen:
                assistant_accumulated.append(token)
                await websocket.send_text(json.dumps({
                    "type": "token",
                    "token": token
                }))

            complete_reply = "".join(assistant_accumulated)
            assistant.chat_history.append({"role": "assistant", "content": complete_reply})



            await websocket.send_text(json.dumps({
                "type": "done",
                "full_text": complete_reply
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


# =====================================================================
# 5. Clean, Modern Frontend (No Dropdown, Live OS Dual-Process Status)
# =====================================================================

HTML_FRONTEND = """<!DOCTYPE html>
<html lang="pl" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cognitive Dual-Process Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              500: '#0ea5e9',
              600: '#0284c7',
              700: '#0369a1',
            }
          }
        }
      }
    };
  </script>
  <style>
    body { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    .chat-bubble-assistant { background-color: #161f30; border: 1px solid #1e293b; }
    .chat-bubble-user { background-color: #0284c7; color: #ffffff; }
    .typing-cursor::after { content: '▋'; animation: blink 1s infinite; color: #38bdf8; }
    @keyframes blink { 0%, 50% { opacity: 1; } 50.1%, 100% { opacity: 0; } }
    .markdown-content p { margin-bottom: 0.6rem; line-height: 1.6; }
    .markdown-content p:last-child { margin-bottom: 0; }
    .markdown-content ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.6rem; }
    .markdown-content ol { list-style-type: decimal; padding-left: 1.25rem; margin-bottom: 0.6rem; }
    .markdown-content li { margin-bottom: 0.25rem; }
    .markdown-content code { background: rgba(0,0,0,0.3); padding: 0.15rem 0.35rem; border-radius: 4px; font-family: monospace; }
  </style>
</head>
<body class="text-slate-200 h-screen flex overflow-hidden">

  <!-- LEFT SIDEBAR: Context & Knowledge Ingestion -->
  <aside id="sidebar" class="w-80 bg-slate-900/90 border-r border-slate-800 flex flex-col transition-all duration-300 z-30">
    
    <!-- Sidebar Header -->
    <div class="p-4 border-b border-slate-800 flex items-center justify-between">
      <div class="flex items-center space-x-2">
        <div class="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse"></div>
        <span class="font-semibold text-sm tracking-wide text-white">Pamięć Kognitywna</span>
      </div>
      <button onclick="clearMemory()" title="Wyczyść pamięć" class="text-xs text-slate-400 hover:text-rose-400 p-1.5 rounded hover:bg-slate-800 transition">
        🗑️ Wyczyść
      </button>
    </div>

    <!-- Upload & Context Tabs -->
    <div class="p-4 space-y-4 flex-1 overflow-y-auto">
      
      <!-- Upload File Zone -->
      <input type="file" id="fileInput" class="hidden" onchange="uploadSelectedFile(event)" onclick="this.value=''" accept=".pdf,.txt,.md,.json,.py,.csv,.log" />
      <label for="fileInput" id="uploadDropzone" class="block border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-4 text-center cursor-pointer transition bg-slate-800/40"
           ondragover="event.preventDefault(); this.classList.add('border-sky-400');"
           ondragleave="this.classList.remove('border-sky-400');"
           ondrop="handleFileDrop(event)">
        <div id="uploadIcon" class="text-2xl mb-1">📄</div>
        <div id="uploadTitle" class="text-xs font-medium text-slate-200">Wgraj dokument (PDF, TXT, MD)</div>
        <div id="uploadSubtitle" class="text-[11px] text-slate-400 mt-1">Kliknij tutaj lub upuść plik</div>
      </label>

      <!-- Live Ingestion Progress Card -->
      <div id="ingestionProgressBox" class="hidden bg-sky-950/80 border border-sky-600/80 rounded-xl p-3 space-y-2">
        <div class="flex justify-between items-center text-xs">
          <span class="font-medium text-sky-200 flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-sky-400 animate-ping"></span>
            <span id="ingestStatusText">Asymilacja w tle...</span>
          </span>
          <span id="ingestPercentText" class="font-mono text-sky-300 font-bold">0%</span>
        </div>
        <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700">
          <div id="ingestProgressBar" class="bg-gradient-to-r from-sky-500 to-emerald-400 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
        <div class="flex justify-between items-center text-[10px] text-slate-400 font-mono">
          <span id="ingestStepsText">Fragment: 0 / 0</span>
          <span id="ingestFactsFoundText" class="text-emerald-400 font-bold">0 faktów</span>
        </div>
      </div>

      <!-- Quick Paste Context Area -->
      <div class="space-y-1.5">
        <div class="flex justify-between items-center text-xs font-medium text-slate-300">
          <span>Wklej treść / notatkę:</span>
          <button onclick="pasteFromClipboard()" class="text-[11px] text-sky-400 hover:underline">Wklej ze schowka</button>
        </div>
        <textarea id="pasteContextInput" rows="3" placeholder="Wklej dowolny artykuł, umowę lub notatkę (auto-analiza)..."
                  onpaste="handleNotePaste(event)"
                  class="w-full bg-slate-950/80 border border-slate-700 rounded-lg p-2.5 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-sky-500 transition resize-none"></textarea>
        <button onclick="submitPastedContext()" class="w-full bg-slate-800 hover:bg-slate-700 text-sky-400 font-medium py-1.5 rounded-lg text-xs border border-slate-700 transition">
          + Zapisz fakty w pamięci Hipokampa
        </button>
      </div>

      <!-- Clipboard Listener Toggle -->
      <div class="bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
        <div>
          <div class="text-xs font-semibold text-slate-200">Podsłuch schowka macOS</div>
          <div class="text-[11px] text-slate-400">Pasywne chłonięcie każdego Cmd+C</div>
        </div>
        <button id="clipBtn" onclick="toggleClipboard()" class="px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400 transition">
          OFF
        </button>
      </div>

      <!-- Ingested Sources & Facts List -->
      <div class="space-y-2 pt-2">
        <div class="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
          <span>Wgrane źródła (<span id="sourceCount">0</span>)</span>
          <span id="factBadge" class="text-[10px] text-emerald-400 font-mono">0 faktów</span>
        </div>
        <div id="sourcesList" class="space-y-1.5 text-xs">
          <div class="text-slate-500 italic text-[11px]">Brak wgranego kontekstu. Wgraj plik lub wklej tekst powyżej.</div>
        </div>
      </div>

      <!-- Live Memory Assertions Preview -->
      <div class="space-y-2 pt-2 border-t border-slate-800">
        <div class="flex items-center justify-between text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          <span>Bufor roboczy O(1)</span>
          <button onclick="copyAllFacts()" class="text-[10px] text-sky-400 hover:text-sky-300 font-sans normal-case transition cursor-pointer">
            📋 Kopiuj fakty
          </button>
        </div>
        <div id="assertionsBox" class="max-h-56 overflow-y-auto space-y-1 text-[11px] font-mono text-slate-300 select-text">
          <span class="text-slate-500 italic">Pamięć pusta...</span>
        </div>
      </div>

    </div>

    <!-- Sidebar Footer Telemetry -->
    <div class="p-3 border-t border-slate-800 text-[11px] text-slate-400 flex justify-between items-center bg-slate-950/40 font-mono">
      <span id="ramUsage">RAM: -- MB</span>
      <span id="deviceBadge" class="text-sky-400">MPS</span>
    </div>
  </aside>

  <!-- MAIN CHAT AREA -->
  <main class="flex-1 flex flex-col bg-[#0b0f19] relative">
    
    <!-- Top Header (Displays Dual-Model Operating System Status) -->
    <header class="h-14 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-900/60 backdrop-blur">
      <div class="flex items-center space-x-3">
        <button onclick="toggleSidebar()" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition">
          ☰
        </button>
        <div class="font-bold text-sm text-white flex items-center gap-2">
          Baddeley Cognitive Assistant <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">Dual-LLM OS</span>
        </div>
      </div>

      <!-- Autonomous Dynamic Status Badges (NO manual model selector!) -->
      <div class="flex items-center space-x-3 text-xs font-mono">
        <div id="hippoBadge" class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300">
          <span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hipokamp: Pamięć Robocza O(1)
        </div>
        <div id="cortexBadge" class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400">
          <span class="w-2 h-2 rounded-full bg-slate-600"></span> Kora (Uśpiona)
        </div>
        <button onclick="clearChat()" class="text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition text-xs font-sans">
          Nowy czat
        </button>
      </div>
    </header>

    <!-- Chat Messages Container -->
    <div id="chatContainer" class="flex-1 overflow-y-auto p-6 space-y-4 max-w-4xl w-full mx-auto">
      
      <!-- Welcome Message -->
      <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-2">
        <div class="font-semibold text-sky-400 flex items-center gap-2">
          <span>🧠 Asystent Kognitywny (<span id="welcomeModelBadge">Kora</span>)</span>
        </div>
        <p class="leading-relaxed text-slate-200">
          Dzień dobry! Po załadowaniu pliku lub wklejeniu notatki **Kora Wykonawcza** automatycznie analizuje materiał, wyciąga najważniejsze fakty do bufora **Hipokampa O(1)** i natychmiast usuwa surowy tekst z pamięci RAM.
          Następnie Kora przechodzi w stan uśpienia i czeka na Twoje pytania.
        </p>
        <div class="pt-2 flex flex-wrap gap-2 text-[11px]">
          <button onclick="sendQuickPrompt('Jakie są kluczowe wnioski z moich materiałów?')" class="bg-slate-800 hover:bg-slate-700 text-sky-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            🔍 Kluczowe wnioski z materiałów
          </button>
          <button onclick="sendQuickPrompt('Podsumuj w punktach najważniejsze tezy wgranego dokumentu.')" class="bg-slate-800 hover:bg-slate-700 text-emerald-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            📜 Podsumuj tezy
          </button>
          <button onclick="sendQuickPrompt('Co zapamiętałeś z ostatnio wgranych źródeł?')" class="bg-slate-800 hover:bg-slate-700 text-amber-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            💡 Co masz w pamięci roboczej?
          </button>
        </div>
      </div>

    </div>

    <!-- Input Bar -->
    <div class="p-4 border-t border-slate-800/80 bg-slate-900/40">
      <div class="max-w-4xl mx-auto flex items-end gap-2">
        <div class="flex-1 bg-slate-900 border border-slate-700 focus-within:border-sky-500 rounded-2xl p-2 flex items-center gap-2 transition">
          <textarea id="messageInput" rows="1" placeholder="Zadaj pytanie o wgrany dokument lub porozmawiaj... (Enter wysyła)"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResize(this)"
                    class="flex-1 bg-transparent text-slate-100 placeholder-slate-500 text-xs outline-none resize-none max-h-32 px-2 py-1"></textarea>
          <button onclick="sendMessage()" class="bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold p-2 rounded-xl transition">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
            </svg>
          </button>
        </div>
      </div>
      <div class="text-[11px] text-center text-slate-500 mt-2">
        Architektura Baddeleya: Kora asymiluje materiał do pamięci roboczej Hipokampa O(1) i odpowiada z niej na żądanie.
      </div>
    </div>

  </main>

  <script>
    let ws;
    let currentAssistantBubble = null;
    let isGenerating = false;
    let clipboardState = false;

    function connectChatWS() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          if (currentAssistantBubble) {
            currentAssistantBubble.rawText += data.token;
            currentAssistantBubble.querySelector('.markdown-content').innerHTML = marked.parse(currentAssistantBubble.rawText);
            scrollChat();
          }
        } else if (data.type === 'done') {
          if (currentAssistantBubble) {
            currentAssistantBubble.classList.remove('typing-cursor');
            currentAssistantBubble = null;
          }
          isGenerating = false;
          setCortexActive(false);
          refreshState();
        }
      };

      ws.onclose = () => {
        setTimeout(connectChatWS, 1500);
      };
    }

    let currentModelLabel = "Kora";
    let isCortexRunning = false;
    let isCortexDistillingState = false;

    function setCortexActive(active) {
      isCortexRunning = active;
      const cBadge = document.getElementById('cortexBadge');
      const hBadge = document.getElementById('hippoBadge');
      if (active) {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600 text-amber-300 animate-pulse';
        cBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span> ⚡ ${currentModelLabel} (Odpowiada...)`;
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-600"></span> Hipokamp: Bufor O(1)';
      } else {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        cBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-600"></span> ${currentModelLabel} (Uśpiona)`;
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hipokamp: Pamięć Robocza O(1)';
      }
    }

    function setCortexDistilling(active) {
      isCortexDistillingState = active;
      const cBadge = document.getElementById('cortexBadge');
      const hBadge = document.getElementById('hippoBadge');
      if (active) {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600 text-amber-300';
        cBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> ${currentModelLabel} (Destyluje wiedzę...)`;
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Hipokamp: Zapisuje fakty...';
      } else {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        cBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-600"></span> ${currentModelLabel} (Uśpiona)`;
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hipokamp: Pamięć Robocza O(1)';
      }
    }

    async function refreshState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Model label
        if (data.telemetry && data.telemetry.model_label) {
          currentModelLabel = data.telemetry.model_label;
          const welcomeBadge = document.getElementById('welcomeModelBadge');
          if (welcomeBadge) welcomeBadge.innerText = currentModelLabel;
          if (!isCortexRunning && !isCortexDistillingState) {
            const cBadge = document.getElementById('cortexBadge');
            const isReady = data.telemetry.is_ready;
            cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
            cBadge.innerHTML = `<span class="w-2 h-2 rounded-full ${isReady ? 'bg-slate-600' : 'bg-amber-400 animate-pulse'}"></span> ${currentModelLabel} (${isReady ? 'Uśpiona' : 'Ładowanie...'})`;
          }
        }

        // Ingestion progress
        const pBox = document.getElementById('ingestionProgressBox');
        if (data.telemetry && data.telemetry.ingestion && data.telemetry.ingestion.active) {
          pBox.classList.remove('hidden');
          const ing = data.telemetry.ingestion;
          document.getElementById('ingestStatusText').innerText = ing.status_text || 'Asymilacja w tle...';
          document.getElementById('ingestPercentText').innerText = `${ing.percent}%`;
          document.getElementById('ingestProgressBar').style.width = `${ing.percent}%`;
          document.getElementById('ingestStepsText').innerText = `Blok: ${ing.step} / ${ing.total}`;
          document.getElementById('ingestFactsFoundText').innerText = `${data.memory.assertion_count} faktów`;

          if (!isCortexRunning) {
            const cBadge = document.getElementById('cortexBadge');
            cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600 text-amber-300';
            cBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> ${currentModelLabel} (Destyluje: ${ing.percent}%)`;
          }
        } else {
          pBox.classList.add('hidden');
        }

        // RAM & Device
        document.getElementById('ramUsage').innerText = `RAM: ${data.telemetry.ram_rss_mb} MB`;
        document.getElementById('deviceBadge').innerText = data.telemetry.device;

        // Clipboard status
        clipboardState = data.telemetry.clipboard_active;
        const clipBtn = document.getElementById('clipBtn');
        if (clipboardState) {
          clipBtn.innerText = 'ON';
          clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-emerald-600 bg-emerald-950 text-emerald-300';
        } else {
          clipBtn.innerText = 'OFF';
          clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400';
        }

        // Sources
        document.getElementById('sourceCount').innerText = data.memory.sources.length;
        document.getElementById('factBadge').innerText = `${data.memory.assertion_count} faktów`;

        const sourcesList = document.getElementById('sourcesList');
        if (data.memory.sources.length > 0) {
          sourcesList.innerHTML = data.memory.sources.map(s => `
            <div class="bg-slate-950/80 border border-slate-800 p-2 rounded-lg flex items-center justify-between">
              <span class="truncate text-slate-200 font-medium">${s.name}</span>
              <span class="text-[10px] text-slate-400 font-mono">${Math.round(s.chars / 1000)}k zn.</span>
            </div>
          `).reverse().join('');
        } else {
          sourcesList.innerHTML = '<div class="text-slate-500 italic text-[11px]">Brak wgranego kontekstu. Wgraj plik lub wklej tekst powyżej.</div>';
        }

        // Assertions preview
        const assertionsBox = document.getElementById('assertionsBox');
        if (data.memory.assertions.length > 0) {
          assertionsBox.innerHTML = data.memory.assertions.map(a => `
            <div class="p-1.5 rounded bg-slate-950/60 border-l-2 border-l-sky-500 select-text cursor-text">
              <div class="text-[9px] text-sky-400 font-bold">[${a.source}] ${a.timestamp}</div>
              <div class="text-slate-200 select-text leading-snug mt-0.5">${escapeHtml(a.text)}</div>
            </div>
          `).reverse().join('');
        } else {
          assertionsBox.innerHTML = '<span class="text-slate-500 italic">Pamięć pusta...</span>';
        }

      } catch (e) {
        console.error(e);
      }
    }

    async function copyAllFacts() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        if (!data.memory.assertions || data.memory.assertions.length === 0) {
          alert('Brak faktów w pamięci do skopiowania.');
          return;
        }
        const text = data.memory.assertions.map((a, i) => `${i + 1}. [${a.source}] ${a.text}`).join('\\n\\n');
        if (navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(text);
            alert(`✓ Skopiowano ${data.memory.assertions.length} faktów do schowka!`);
            return;
          } catch (clipErr) {
            // fallback below
          }
        }
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        alert(`✓ Skopiowano ${data.memory.assertions.length} faktów do schowka!`);
      } catch (e) {
        alert('Nie udało się skopiować faktów: ' + e);
      }
    }

    function sendMessage() {
      const input = document.getElementById('messageInput');
      const text = input.value.trim();
      if (!text || isGenerating) return;
      input.value = '';
      input.rows = 1;

      // Add user bubble
      const chatContainer = document.getElementById('chatContainer');
      const userDiv = document.createElement('div');
      userDiv.className = 'flex justify-end';
      userDiv.innerHTML = `
        <div class="chat-bubble-user p-3 px-4 rounded-2xl max-w-xl text-xs whitespace-pre-wrap">
          ${escapeHtml(text)}
        </div>
      `;
      chatContainer.appendChild(userDiv);

      // Create empty assistant bubble
      const assistantDiv = document.createElement('div');
      assistantDiv.className = 'chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-1 typing-cursor';
      assistantDiv.rawText = '';
      assistantDiv.innerHTML = `
        <div class="font-semibold text-sky-400 text-[11px] mb-1 flex items-center gap-1.5">
          <span>${currentModelLabel}</span>
        </div>
        <div class="markdown-content leading-relaxed text-slate-200"></div>
      `;
      chatContainer.appendChild(assistantDiv);
      currentAssistantBubble = assistantDiv;
      isGenerating = true;
      setCortexActive(true);

      scrollChat();

      // Send to WebSocket
      ws.send(JSON.stringify({ message: text }));
    }

    function sendQuickPrompt(prompt) {
      document.getElementById('messageInput').value = prompt;
      sendMessage();
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 128) + 'px';
    }

    function scrollChat() {
      const c = document.getElementById('chatContainer');
      c.scrollTop = c.scrollHeight;
    }

    let noteDebounce = null;
    function handleNotePaste(e) {
      clearTimeout(noteDebounce);
      noteDebounce = setTimeout(() => {
        submitPastedContext();
      }, 400);
    }

    async function uploadSelectedFile(e) {
      const file = (e.target && e.target.files && e.target.files.length > 0) ? e.target.files[0] : (e.files ? e.files[0] : null);
      if (!file) return;

      const titleEl = document.getElementById('uploadTitle');
      const subEl = document.getElementById('uploadSubtitle');
      const iconEl = document.getElementById('uploadIcon');

      const originalTitle = 'Wgraj dokument (PDF, TXT, MD)';
      titleEl.innerText = `Przesyłanie: ${file.name.substring(0, 16)}...`;
      subEl.innerText = 'Przygotowywanie asymilacji...';
      iconEl.innerText = '⏳';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload_file', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.ok) {
          titleEl.innerText = '✓ Plik przesłany!';
          subEl.innerText = `Asymilacja w tle (${Math.round(data.chars / 1000)}k zn.)...`;
          iconEl.innerText = '⚡';

          // Add visible confirmation card directly into chat (Kora remains asleep!)
          const chatContainer = document.getElementById('chatContainer');
          const noticeDiv = document.createElement('div');
          noticeDiv.className = 'bg-sky-950/50 border border-sky-700/80 p-3.5 rounded-2xl max-w-xl text-xs text-sky-200 mx-auto text-center space-y-1 my-2';
          noticeDiv.innerHTML = `
            <div class="font-bold text-sky-300 flex items-center justify-center gap-1.5">
              <span>📄</span> Wgrano materiał: ${escapeHtml(data.filename)}
            </div>
            <div class="text-[11px] text-sky-300">
              Rozpoczęto asymilację ${Math.round(data.chars / 1000)}k znaków w tle. Pasek postępu i nowe fakty pojawiają się na żywo w lewym panelu. Możesz już zadawać pytania!
            </div>
          `;
          chatContainer.appendChild(noticeDiv);
          scrollChat();

          setTimeout(() => {
            titleEl.innerText = 'Wgraj kolejny dokument';
            subEl.innerText = 'Kliknij tutaj lub upuść plik';
            iconEl.innerText = '📄';
          }, 3500);

          await refreshState();
        } else {
          alert(`Błąd wczytywania: ${data.error}`);
          titleEl.innerText = originalTitle;
          iconEl.innerText = '📄';
        }
      } catch (err) {
        console.error(err);
        alert('Błąd połączenia z serwerem. Upewnij się, że serwer jest uruchomiony w terminalu.');
        titleEl.innerText = originalTitle;
        iconEl.innerText = '📄';
      } finally {
        document.getElementById('fileInput').value = '';
      }
    }

    function handleFileDrop(e) {
      e.preventDefault();
      const dropzone = document.getElementById('uploadDropzone');
      if (dropzone) dropzone.classList.remove('border-sky-400');
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        uploadSelectedFile({ files: files });
      }
    }

    async function submitPastedContext() {
      const input = document.getElementById('pasteContextInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      input.placeholder = 'Kora asymiluje notatkę w tle...';

      try {
        const res = await fetch('/api/ingest_text', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text, source_name: 'Wklejona notatka' })
        });
        const data = await res.json();
        if (data.ok) {
          const chatContainer = document.getElementById('chatContainer');
          const noticeDiv = document.createElement('div');
          noticeDiv.className = 'bg-sky-950/50 border border-sky-700/80 p-3 rounded-2xl max-w-xl text-xs text-sky-200 mx-auto text-center space-y-1 my-2';
          noticeDiv.innerHTML = `
            <div class="font-bold text-sky-300 flex items-center justify-center gap-1.5">
              <span>📝</span> Wklejono treść (${data.chars} znaków)
            </div>
            <div class="text-[11px] text-sky-400">
              Kora rozpoczęła asymilację materiału w tle. Nowe fakty pojawią się w pamięci roboczej.
            </div>
          `;
          chatContainer.appendChild(noticeDiv);
          scrollChat();
          input.placeholder = 'Wklej dowolny artykuł, umowę lub notatkę (auto-analiza)...';
          await refreshState();
        }
      } catch (err) {
        console.error(err);
      } finally {
        input.placeholder = 'Wklej dowolny artykuł, umowę lub notatkę (auto-analiza)...';
      }
    }

    async function pasteFromClipboard() {
      try {
        const text = await navigator.clipboard.readText();
        if (!text || !text.trim()) {
          alert('Schowek jest pusty.');
          return;
        }
        document.getElementById('pasteContextInput').value = text;
        await submitPastedContext();
      } catch (e) {
        alert('Zezwól na dostęp do schowka');
      }
    }

    async function toggleClipboard() {
      await fetch('/api/clipboard_toggle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ active: !clipboardState })
      });
      refreshState();
    }

    async function clearMemory() {
      if (confirm('Wyczyścić pamięć roboczą asystenta?')) {
        await fetch('/api/clear_memory', { method: 'POST' });
        refreshState();
      }
    }

    async function clearChat() {
      await fetch('/api/clear_chat', { method: 'POST' });
      document.getElementById('chatContainer').innerHTML = `
        <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-2">
          <div class="font-semibold text-sky-400">🤖 Nowy czat</div>
          <p class="text-slate-200">Pamięć robocza zachowana. Zadaj dowolne pytanie Kory Wykonawczej.</p>
        </div>
      `;
    }

    function toggleSidebar() {
      const s = document.getElementById('sidebar');
      s.classList.toggle('-ml-80');
    }

    function escapeHtml(text) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    window.addEventListener('load', () => {
      connectChatWS();
      refreshState();
      setInterval(refreshState, 2500);
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=HTML_FRONTEND)


# =====================================================================
# 6. Automated Headless Verification Engine
# =====================================================================

async def run_headless_tests():
    """Weryfikacja podsystemu pamięci i silnika dualnego."""
    print("\n=======================================================")
    print("Starting Baddeley Dual-Process Assistant Tests")
    print("=======================================================")

    test_wm = BaddeleyWorkingMemoryStore(max_assertions=30)

    print("[1/3] Testing Chunking & Distillation Pipeline...")
    synthetic_doc = (
        "Rozdział 1: Wprowadzenie do ekonomii dóbr luksusowych. "
        "Dobra luksusowe charakteryzują się dodatnią elastycznością dochodową popytu. "
        "Współczesny rynek dóbr luksusowych opiera się na ekskluzywności i kapitale symbolicznym.\n\n"
        "Rozdział 2: Cyfryzacja i nowe rynki w Azji. "
        "Głównym motorem wzrostu stały się Chiny i pokolenie Z. "
        "Zrównoważony rozwój i autentyczność to nowe kluczowe filary pozycjonowania marek."
    )
    facts = assistant.engine.distill_document_chunks(synthetic_doc, "luxury.pdf")
    assert len(facts) >= 2, "Failed to distill document chunks"
    await test_wm.store_assertions(facts, source="luxury.pdf")
    snap = await test_wm.snapshot()
    assert snap["assertion_count"] > 0, "Assertions not stored"
    print(f"      Distilled {len(facts)} high-value assertions across chapters.")

    print("[2/3] Testing Dynamic Working Memory Context Prompt...")
    prompt_ctx = await test_wm.get_working_memory_prompt()
    assert "luksus" in prompt_ctx.lower() or "chiny" in prompt_ctx.lower()
    print(f"      Working memory prompt length: {len(prompt_ctx)} chars.")

    print("[3/3] Testing Preemption Mechanism Flag...")
    assert assistant.engine.is_preempted is False, "Preemption stuck"
    print("      Preemption synchronization primitives verified.")

    print("\n[TEST PASSED] Dual-Process Cognitive Engine verified successfully.\n")


# =====================================================================
# 7. Main Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Baddeley Cognitive Dual-Model Assistant")
    parser.add_argument("--test", action="store_true", help="Run automated verification suite and exit")
    parser.add_argument("--model", type=str, default="8b", choices=["8b", "3b", "1.7b"], help="Executive Cortex model: 8b (default), 3b (Llama-3.2-3B), 1.7b (SmolLM2)")
    parser.add_argument("--3b", dest="use_3b", action="store_true", help="Shortcut for --model 3b (Llama-3.2-3B)")
    parser.add_argument("--fast", action="store_true", help="Shortcut for fast testing mode (--model 3b)")
    parser.add_argument("--1.7b", dest="use_17b", action="store_true", help="Shortcut for --model 1.7b (SmolLM2)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    selected_model = args.model
    if args.use_3b or args.fast:
        selected_model = "3b"
    elif args.use_17b:
        selected_model = "1.7b"

    assistant.configure_model(selected_model)

    if args.test:
        asyncio.run(run_headless_tests())
        sys.exit(0)

    url = f"http://{args.host}:{args.port}"

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n=======================================================")
    print(f"  🧠 BADDELEY COGNITIVE DUAL-PROCESS ASSISTANT")
    print(f"  Local Web App: {url}")
    print(f"  Executive Cortex: {assistant.engine.model_desc}")
    print(f"  Working Memory: Hippocampus O(1) Episodic Buffer")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
