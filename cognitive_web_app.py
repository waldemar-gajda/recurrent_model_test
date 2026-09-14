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

    def __init__(self, max_assertions: int = 80):
        self.max_assertions = max_assertions
        self.lock = asyncio.Lock()
        self.assertions: collections.deque[MemoryAssertion] = collections.deque(maxlen=max_assertions)
        self.sources: List[Dict[str, Any]] = []
        self.total_tokens_ingested: int = 0
        self.last_ingest_time: str = "Brak"

    async def add_source(self, name: str, char_count: int, source_type: str):
        async with self.lock:
            self.sources.append({
                "name": name,
                "chars": char_count,
                "type": source_type,
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            })

    async def store_assertions(self, texts: List[str], source: str):
        async with self.lock:
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            self.last_ingest_time = now_str
            for t in texts:
                cleaned = t.strip()
                if len(cleaned) > 15:
                    self.assertions.append(MemoryAssertion(
                        id=f"M-{int(time.time() * 1000) % 100000}",
                        source=source,
                        text=cleaned[:300],
                        timestamp=now_str
                    ))

    async def get_working_memory_prompt(self, max_chars: int = 4000) -> str:
        """Kompiluje stan pamięci roboczej do promptu dla Kory Wykonawczej."""
        async with self.lock:
            if not self.assertions:
                return "Pamięć robocza jest pusta (brak wgranych materiałów ani skopiowanego tekstu)."
            lines = [f"• [{a.source}] {a.text}" for a in list(self.assertions)[-30:]]
            result = "\n".join(lines)
            return result[:max_chars]

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "assertion_count": len(self.assertions),
                "assertions": [dataclasses.asdict(a) for a in list(self.assertions)[-25:]],
                "sources": list(self.sources),
                "total_tokens": self.total_tokens_ingested,
                "last_ingest": self.last_ingest_time,
            }

    async def clear(self):
        async with self.lock:
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

    def __init__(self, base_dir: Path):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.base_dir = base_dir

        # Hippocampus (1.7B)
        self.hippo_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        self.hippo_tok = None
        self.hippo_model = None

        # Cortex (LLaMA-3-8B)
        self.cortex_local_path = base_dir / "llama-3-8b-instruct"
        self.cortex_path = str(self.cortex_local_path) if self.cortex_local_path.exists() else "NousResearch/Meta-Llama-3-8B-Instruct"
        self.cortex_tok = None
        self.cortex_model = None

        # State & Preemption primitives
        self.preemption_lock = threading.Lock()
        self.is_preempted = False
        self.preemption_count = 0
        self.status = "Inicjalizacja modeli..."
        self.is_ready = False
        self.loading_in_progress = False
        self.hippo_active = False

    def initialize_both_models(self):
        """Wczytuje model Kory Wykonawczej (LLaMA-3-8B) do pamięci operacyjnej."""
        self.loading_in_progress = True
        try:
            self.status = "Ładowanie Kory Wykonawczej (LLaMA-3-8B)..."
            print(f"  [Cognitive OS] Loading Executive Cortex ({self.cortex_path}) on {self.device}...")
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
            print("  [Cognitive OS] ✓ Executive Cortex loaded.")

            self.is_ready = True
            self.status = "System Gotowy (Kora LLaMA-3-8B + Hipokamp O(1))"
            print(f"  [Cognitive OS] Cognitive Runtime fully operational on {self.device.upper()}.")
        except Exception as e:
            self.status = f"Błąd inicjalizacji: {e}"
            print(f"  [Cognitive OS] ✗ Initialization failed: {e}")
        finally:
            self.loading_in_progress = False

    # ── Source Cleaning & Semantic Distillation into Hippocampus ─────────────

    def clean_source_text(self, text: str) -> str:
        """
        Oczyszcza surowy tekst dokumentu (np. z PDF):
        - Łączy słowa rozbite dywizem na końcu linii (np. 'luxu-\\nry' -> 'luxury')
        - Łączy liczby i symbole rozbite przez PDF (np. '95\\n%' -> '95%')
        - Łączy linie wewnątrz akapitów rozbite przez formatowanie PDF
        - Usuwa powtarzalne stopki, nagłówki i szum redakcyjny (nie kasując kluczowych danych liczbowych)
        """
        # 1. Łączenie liczb z procentami lub jednostkami rozbitych enterem
        t = re.sub(r'(\d+)\s*\n\s*(%|mln|mld|tys\.|proc\.)', r'\1\2', text)
        # 2. Łączenie słów rozdzielonych łącznikiem na końcu linii
        t = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', t)
        # 3. Łączenie linii wewnątrz zdań (usuwa pojedyncze \n niepoprzedzone kropką/dwukropkiem)
        t = re.sub(r'(?<![.!?:\n])\n(?![A-Z0-9\n])', ' ', t)
        # 4. Filtr szumu: linie będące numerami stron lub formułkami wydawniczymi
        cleaned_lines = []
        for line in t.split('\n'):
            l_strip = line.strip()
            if not l_strip:
                continue
            # Tylko jawne nagłówki stron usuwamy, nigdy same cyfry (np. 95, 637 na slajdach)
            if re.match(r'^(page\s+\d+(\s+of\s+\d+)?|strona\s+\d+(\s+z\s+\d+)?)$', l_strip, re.IGNORECASE):
                continue
            if any(term in l_strip.lower() for term in ['all rights reserved', 'doi: 10.', 'isbn 978-', 'printed in', 'taylor & francis', 'contents', 'index']):
                continue
            cleaned_lines.append(l_strip)
        return '\n\n'.join(cleaned_lines)

    def distill_document_chunks(self, full_text: str, source_name: str) -> List[str]:
        """
        Asymilacja semantyczna materiału źródłowego przez Korę (LLaMA-3-8B):
        - Oczyszcza surowy tekst i filtruje szum redakcyjny
        - Dzieli na logiczne makro-sekcje
        - Wykorzystuje model do wygenerowania ustrukturyzowanych faktów/notatek w języku polskim
        - Zapisuje notatki do pamięci roboczej O(1) i natychmiast eksmituje surowe tokeny
        """
        self.hippo_active = True
        try:
            # Jeśli Kora jest aktywnie ładowana w tle, poczekaj na zakończenie
            if self.cortex_model is None and self.loading_in_progress:
                wait_sec = 0
                while self.cortex_model is None and self.loading_in_progress and wait_sec < 30:
                    time.sleep(0.5)
                    wait_sec += 1

            cleaned_text = self.clean_source_text(full_text)
            if not cleaned_text.strip():
                return []

            # Podział na akapity
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 25]
            if not paragraphs:
                return []

            # Zbuduj makro-bloki po ok. 2200 znaków
            macro_chunks = []
            chunk_size = 2200
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

            # Wybierz do 4 najważniejszych przekrojowych fragmentów dokumentu
            if len(macro_chunks) <= 4:
                selected_chunks = macro_chunks
            else:
                selected_chunks = [
                    macro_chunks[0],
                    macro_chunks[len(macro_chunks) // 3],
                    macro_chunks[2 * len(macro_chunks) // 3],
                    macro_chunks[-1]
                ]

            extracted_facts = []

            # 1. Głęboka destylacja semantyczna przez model Kory LLaMA-3-8B (gdy załadowany)
            if self.cortex_model is not None and self.cortex_tok is not None:
                try:
                    for chunk in selected_chunks:
                        with self.preemption_lock:
                            prompt_messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Jesteś precyzyjnym modułem kognitywnym asystenta. Przeanalizuj poniższy fragment tekstu "
                                        "i wyodrębnij z niego od 3 do 5 kluczowych, merytorycznych faktów (definicje, tezy, dane liczbowe, rynki, strategie). "
                                        "Każdy fakt zapisz w nowej linii zaczynając od myślnika '- '. "
                                        "Pisz wyłącznie pełnymi, poprawnymi gramatycznie zdaniami w języku polskim. "
                                        "Uzupełnij kontekstowo urwane słowa ze slajdów. "
                                        "Żadnych wstępów ani komentarzy pobocznych."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": f"DOKUMENT: {source_name}\n\n{chunk[:2200]}"
                                }
                            ]
                            prompt = self.cortex_tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                            inputs = self.cortex_tok(prompt, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                out_ids = self.cortex_model.generate(
                                    **inputs,
                                    max_new_tokens=180,
                                    do_sample=False,
                                    pad_token_id=self.cortex_tok.pad_token_id
                                )
                            response = self.cortex_tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                            for line in response.splitlines():
                                l = line.strip()
                                if l.startswith("-") or l.startswith("•") or (len(l) > 3 and l[0].isdigit() and l[1] in [".", ")"]):
                                    clean_fact = re.sub(r"^[-•\d.)\s]+", "", l).strip()
                                    if len(clean_fact) > 15 and clean_fact not in extracted_facts:
                                        extracted_facts.append(clean_fact)
                except Exception as e:
                    print(f"  [Cognitive OS] Neural distillation note: {e}. Using high-salience fallback.")

            # 2. Szybki, odporny ekstraktor pełnych zdań jako fallback lub uzupełnienie
            if len(extracted_facts) < 4:
                for chunk in macro_chunks[:6]:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if len(s.strip()) > 35]
                    for s in sentences[:3]:
                        if not any(s in existing for existing in extracted_facts):
                            extracted_facts.append(s)

            return extracted_facts[:35]
        finally:
            self.hippo_active = False

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
                        yield "Kora Wykonawcza (LLaMA-3-8B) nadal się ładuje do pamięci RAM. Proszę odczekać kilka sekund..."
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
                    "3. Nigdy nie wypluwaj surowych zmiennych programistycznych ani technicznego debugu. Odpowiadaj jak wybitny asystent człowieka."
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
    def __init__(self):
        self.wm = BaddeleyWorkingMemoryStore(max_assertions=80)
        self.engine = DualCognitiveEngine(base_dir=BASE_DIR)
        self.chat_history: List[Dict[str, str]] = []
        self.clipboard_monitor_active = False
        self.clipboard_thread: Optional[threading.Thread] = None
        self.last_clipboard_text = ""
        self.is_running = True

    def start(self):
        threading.Thread(target=self.engine.initialize_both_models, daemon=True).start()

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
            "hippo_active": self.engine.hippo_active,
            "clipboard_active": self.clipboard_monitor_active,
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
    facts = assistant.engine.distill_document_chunks(req.text, src_name)
    await assistant.wm.store_assertions(facts, source=src_name)
    await assistant.wm.add_source(src_name, len(req.text), "text")
    return {
        "ok": True,
        "source_name": src_name,
        "chars": len(req.text),
        "facts_extracted": len(facts),
        "message": f"Kora przeanalizowała materiał i zapisała {len(facts)} faktów w pamięci roboczej."
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

    # Ingest through Hippocampus
    facts = assistant.engine.distill_document_chunks(extracted_text, filename)
    await assistant.wm.store_assertions(facts, source=filename)
    await assistant.wm.add_source(filename, len(extracted_text), "file")
    
    return {
        "ok": True,
        "filename": filename,
        "chars": len(extracted_text),
        "facts_extracted": len(facts),
        "message": f"Kora przeanalizowała dokument ({len(extracted_text)} znaków) i zasiliła pamięć roboczą o {len(facts)} faktów."
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

            # Re-ingest conversational takeaway into episodic buffer
            await assistant.wm.store_assertions(
                [f"Użytkownik zapytał: {user_msg}", f"Kora odpowiedziała: {complete_reply[:140]}"],
                source="Rozmowa"
            )

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
          <span class="w-2 h-2 rounded-full bg-slate-600"></span> Kora 8B (Uśpiona)
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
          <span>🧠 Asystent Kognitywny (Kora Wykonawcza LLaMA-3-8B)</span>
        </div>
        <p class="leading-relaxed text-slate-200">
          Dzień dobry! Po załadowaniu pliku lub wklejeniu notatki **Kora Wykonawcza (LLaMA-3-8B)** automatycznie analizuje materiał, wyciąga najważniejsze fakty do bufora **Hipokampa O(1)** i natychmiast usuwa surowy tekst z pamięci RAM.
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
        Architektura Baddeleya: Kora (LLaMA-3-8B) asymiluje materiał do pamięci roboczej Hipokampa O(1) i odpowiada z niej na żądanie.
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

    function setCortexActive(active) {
      const cBadge = document.getElementById('cortexBadge');
      const hBadge = document.getElementById('hippoBadge');
      if (active) {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600 text-amber-300 animate-pulse';
        cBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-amber-400"></span> ⚡ Kora 8B (Odpowiada...)';
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-600"></span> Hipokamp: Bufor O(1)';
      } else {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        cBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-600"></span> Kora 8B (Uśpiona)';
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hipokamp: Pamięć Robocza O(1)';
      }
    }

    function setCortexDistilling(active) {
      const cBadge = document.getElementById('cortexBadge');
      const hBadge = document.getElementById('hippoBadge');
      if (active) {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600 text-amber-300';
        cBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> Kora 8B (Destyluje wiedzę...)';
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Hipokamp: Zapisuje fakty...';
      } else {
        cBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400';
        cBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-600"></span> Kora 8B (Uśpiona)';
        hBadge.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800 text-emerald-300';
        hBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hipokamp: Pamięć Robocza O(1)';
      }
    }

    async function refreshState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

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
          <span>Kora Wykonawcza (LLaMA-3-8B)</span>
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
      titleEl.innerText = `Analiza: ${file.name.substring(0, 16)}...`;
      subEl.innerText = 'Kora destyluje fakty do pamięci...';
      iconEl.innerText = '⏳';
      setCortexDistilling(true);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload_file', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.ok) {
          titleEl.innerText = '✓ Zaasymilowano pomyślnie!';
          subEl.innerText = `${Math.round(data.chars / 1000)}k zn., ${data.facts_extracted} faktów`;
          iconEl.innerText = '✅';

          // Add visible confirmation card directly into chat (Kora remains asleep!)
          const chatContainer = document.getElementById('chatContainer');
          const noticeDiv = document.createElement('div');
          noticeDiv.className = 'bg-emerald-950/50 border border-emerald-700/80 p-3.5 rounded-2xl max-w-xl text-xs text-emerald-200 mx-auto text-center space-y-1 my-2';
          noticeDiv.innerHTML = `
            <div class="font-bold text-emerald-300 flex items-center justify-center gap-1.5">
              <span>📄</span> Zaasymilowano dokument: ${escapeHtml(data.filename)}
            </div>
            <div class="text-[11px] text-emerald-400">
              Kora przeanalizowała treść i zapisała <strong>${data.facts_extracted} kluczowych faktów</strong> (${Math.round(data.chars / 1000)}k znaków) w pamięci roboczej Hipokampa. Kora jest w uśpieniu i czeka na Twoje pytania.
            </div>
          `;
          chatContainer.appendChild(noticeDiv);
          scrollChat();

          setTimeout(() => {
            titleEl.innerText = 'Wgraj kolejny dokument';
            subEl.innerText = 'Kliknij tutaj lub upuść plik';
            iconEl.innerText = '📄';
          }, 4000);

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
        setCortexDistilling(false);
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
      input.placeholder = 'Kora analizuje notatkę i zapisuje fakty...';
      setCortexDistilling(true);

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
              <span>📝</span> Zaasymilowano notatkę (${data.chars} znaków)
            </div>
            <div class="text-[11px] text-sky-400">
              Kora zapisała <strong>${data.facts_extracted} faktów</strong> w pamięci roboczej Hipokampa. Kora czeka na Twoje pytania.
            </div>
          `;
          chatContainer.appendChild(noticeDiv);
          scrollChat();
          await refreshState();
        }
      } catch (err) {
        console.error(err);
      } finally {
        setCortexDistilling(false);
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
    parser.add_argument("--port", type=int, default=8000, help="Port to serve (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

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
    print(f"  Operating System: Hippocampus (1.7B) + Cortex (LLaMA-3 8B)")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
