#!/usr/bin/env python3
"""
cognitive_web_app.py — Baddeley Cognitive Working Memory Assistant & Agent
========================================================================
A modern, intuitive AI assistant powered by Baddeley Cognitive Working Memory.
Features:
- Natural Conversational Agent (real neural generation with streaming tokens).
- Working Memory Context Injection (remembers uploaded files, notes, clipboard).
- Drag-and-Drop File Upload (PDF, TXT, MD, JSON, Python, CSV, Log).
- Passive macOS Clipboard Monitor (opt-in toggle).
- Clean, clutter-free modern interface (ChatGPT / Claude style).
- Model switching (SmolLM2-1.7B for fast responses, LLaMA-3-8B for frontier reasoning).
- Headless verification: `python3 cognitive_web_app.py --test`.
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
# 1. Cognitive Working Memory Engine (O(1) Bounded State)
# =====================================================================

@dataclasses.dataclass
class WorkingMemoryFact:
    fact_id: str
    source: str
    text: str
    timestamp: str


class CognitiveWorkingMemory:
    """Maintains an entity-scoped, bounded working memory of active context."""

    def __init__(self, max_facts: int = 50):
        self.max_facts = max_facts
        self.lock = asyncio.Lock()
        self.facts: collections.deque[WorkingMemoryFact] = collections.deque(maxlen=max_facts)
        self.uploaded_sources: List[Dict[str, Any]] = []
        self.total_tokens_ingested: int = 0

    async def ingest(self, text: str, source: str = "tekst"):
        """Ingest and distill incoming context."""
        cleaned = text.strip()
        if not cleaned:
            return

        async with self.lock:
            # Approximate token count
            approx_tokens = len(cleaned.split())
            self.total_tokens_ingested += approx_tokens

            # Extract distinct factual sentences
            sentences = [s.strip() for s in re.split(r"[.\n;]+", cleaned) if len(s.strip()) > 15]
            if not sentences:
                sentences = [cleaned[:160]]

            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            for sent in sentences[:10]:
                fact = WorkingMemoryFact(
                    fact_id=f"F-{int(time.time() * 1000) % 100000}",
                    source=source,
                    text=sent[:240],
                    timestamp=now_str,
                )
                self.facts.append(fact)

    async def add_source_record(self, name: str, char_count: int, source_type: str):
        async with self.lock:
            self.uploaded_sources.append({
                "name": name,
                "chars": char_count,
                "type": source_type,
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            })

    async def get_context_for_prompt(self, max_chars: int = 2500) -> str:
        """Returns summarized working memory string for neural prompt injection."""
        async with self.lock:
            if not self.facts:
                return ""
            lines = [f"• [{f.source}] {f.text}" for f in list(self.facts)[-15:]]
            result = "\n".join(lines)
            return result[:max_chars]

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "fact_count": len(self.facts),
                "facts": [dataclasses.asdict(f) for f in list(self.facts)[-20:]],
                "sources": list(self.uploaded_sources),
                "total_tokens": self.total_tokens_ingested,
            }

    async def clear(self):
        async with self.lock:
            self.facts.clear()
            self.uploaded_sources.clear()


# =====================================================================
# 2. Neural Model Manager (Real Autoregressive LLM Inference)
# =====================================================================

class NeuralModelManager:
    """Manages weights, caching, and token generation on Apple Silicon MPS / CPU."""

    MODELS_CONFIG = {
        "SmolLM2-1.7B": {
            "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "desc": "Zrównoważony i szybki (rekomendowany)",
            "ram": "~3.4 GB",
        },
        "LLaMA-3-8B": {
            "path": "./llama-3-8b-instruct",
            "fallback_path": "NousResearch/Meta-Llama-3-8B-Instruct",
            "desc": "Zaawansowane rozumowanie (najwyższa jakość)",
            "ram": "~16 GB",
        },
        "SmolLM2-135M": {
            "path": "./smollm2-135m-instruct",
            "fallback_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "desc": "Ultralekki model kieszonkowy",
            "ram": "~250 MB",
        },
    }

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.current_model_name = "SmolLM2-1.7B"
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        self.load_status = "Inicjalizacja..."
        self._lock = threading.Lock()

    def get_model_options(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": k,
                "desc": v["desc"],
                "ram": v["ram"],
                "active": k == self.current_model_name
            }
            for k, v in self.MODELS_CONFIG.items()
        ]

    def load_model(self, model_name: str):
        """Loads or hot-swaps model into memory."""
        with self._lock:
            if self.model is not None and self.current_model_name == model_name:
                return

            self.is_loading = True
            self.load_status = f"Ładowanie modelu {model_name}..."
            try:
                # Free previous model memory
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    self.model = None
                    self.tokenizer = None
                    if self.device == "mps":
                        torch.mps.empty_cache()

                cfg = self.MODELS_CONFIG.get(model_name, self.MODELS_CONFIG["SmolLM2-1.7B"])
                path = cfg["path"]

                # Check if local directory exists, else fallback to HF Hub
                if not Path(path).exists() and "fallback_path" in cfg:
                    path = cfg["fallback_path"]

                print(f"  [Neural Engine] Loading {model_name} from '{path}' on {self.device}...")
                tok = AutoTokenizer.from_pretrained(path)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                # Use bfloat16 for 1.7B and 8B on MPS
                dtype = torch.bfloat16 if self.device == "mps" and "135M" not in model_name else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                model.eval()

                self.tokenizer = tok
                self.model = model
                self.current_model_name = model_name
                self.load_status = f"Gotowy ({model_name} na {self.device.upper()})"
                print(f"  [Neural Engine] ✓ {model_name} loaded successfully.")
            except Exception as e:
                self.load_status = f"Błąd ładowania: {e}"
                print(f"  [Neural Engine] ✗ Failed to load {model_name}: {e}")
            finally:
                self.is_loading = False

    def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        working_memory_context: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        if self.model is None or self.tokenizer is None:
            wait_count = 0
            while self.is_loading and wait_count < 30:
                time.sleep(0.5)
                wait_count += 1
            if self.model is None or self.tokenizer is None:
                self.load_model(self.current_model_name)

        if self.model is None or self.tokenizer is None:
            yield "Przepraszam, model nie mógł zostać załadowany. Sprawdź logi serwera."
            return

        # Prepare system prompt with working memory
        system_instruction = (
            "Jesteś pomocnym, inteligentnym asystentem AI wyposażonym w pamięć roboczą (Working Memory).\n"
            "Odpowiadaj naturalnie, wyczerpująco i uprzejmie w języku użytkownika (domyślnie po polsku).\n"
        )
        if working_memory_context:
            system_instruction += (
                "\n--- PAMIĘĆ ROBOCZA (FAKTY Z WGRANYCH DOKUMENTÓW, SCHOWKA I ROZMOWY) ---\n"
                f"{working_memory_context}\n"
                "------------------------------------------------------------------------\n"
                "Instrukcja: Jeśli pytanie użytkownika dotyczy powyższego kontekstu lub dokumentów, "
                "wykorzystaj te informacje, aby precyzyjnie i zgodnie z faktami odpowiedzieć na pytanie. "
                "Jeśli to zwykła rozmowa lub pytanie ogólne, odpowiedz płynnie z własnej wiedzy."
            )

        full_messages = [{"role": "system", "content": system_instruction}] + messages

        try:
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models with non-standard chat templates
            prompt = f"{system_instruction}\n\nUser: {messages[-1]['content']}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.1),
            temperature=max(temperature, 0.2),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen_thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        gen_thread.start()

        for chunk in streamer:
            yield chunk

        gen_thread.join()


# =====================================================================
# 3. Application State & Continuous Ingestion Watchers
# =====================================================================

class AssistantApplication:
    def __init__(self):
        self.wm = CognitiveWorkingMemory()
        self.model_mgr = NeuralModelManager()
        self.chat_history: List[Dict[str, str]] = []
        self.clipboard_monitor_active = False
        self.clipboard_thread: Optional[threading.Thread] = None
        self.last_clipboard_text = ""
        self.start_time = time.time()
        self.is_running = True

    def start(self):
        # Asynchronously load default model SmolLM2-1.7B
        threading.Thread(target=self.model_mgr.load_model, args=("SmolLM2-1.7B",), daemon=True).start()

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
                if current and current != self.last_clipboard_text and len(current.strip()) > 5:
                    self.last_clipboard_text = current
                    asyncio.run(self.wm.ingest(current, source="schowek"))
                    asyncio.run(self.wm.add_source_record("Schowek systemowy", len(current), "clipboard"))
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
            "clipboard_active": self.clipboard_monitor_active,
            "current_model": self.model_mgr.current_model_name,
            "model_status": self.model_mgr.load_status,
            "is_loading": self.model_mgr.is_loading,
            "device": self.model_mgr.device.upper(),
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

app = FastAPI(title="Cognitive Assistant", lifespan=lifespan)

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


class SwitchModelRequest(BaseModel):
    model_name: str


class ClipboardRequest(BaseModel):
    active: bool


@app.get("/api/state")
async def api_get_state():
    snap = await assistant.wm.snapshot()
    telemetry = assistant.get_telemetry()
    models = assistant.model_mgr.get_model_options()
    return {
        "memory": snap,
        "telemetry": telemetry,
        "models": models,
        "chat_count": len(assistant.chat_history),
    }


@app.post("/api/ingest_text")
async def api_ingest_text(req: TextIngestRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Brak tekstu do wgrania"})
    await assistant.wm.ingest(req.text, source=req.source_name or "Wklejony tekst")
    await assistant.wm.add_source_record(req.source_name or "Wklejony tekst", len(req.text), "text")
    return {"ok": True, "message": f"Wgrano {len(req.text)} znaków do pamięci roboczej."}


@app.post("/api/upload_file")
async def api_upload_file(file: UploadFile = File(...)):
    filename = file.filename or "plik"
    contents = await file.read()
    extracted_text = ""

    if filename.lower().endswith(".pdf"):
        if pypdf:
            try:
                reader = pypdf.PdfReader(BytesIO(contents))
                extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Błąd czytania PDF: {e}"})
        else:
            return JSONResponse(status_code=400, content={"error": "Biblioteka pypdf nie jest zainstalowana"})
    else:
        # Text, Markdown, Code, JSON, Log
        try:
            extracted_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                extracted_text = contents.decode("latin-1")
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Nieobsługiwany format pliku"})

    if not extracted_text.strip():
        return JSONResponse(status_code=400, content={"error": "Plik jest pusty"})

    await assistant.wm.ingest(extracted_text, source=filename)
    await assistant.wm.add_source_record(filename, len(extracted_text), "file")
    return {
        "ok": True,
        "filename": filename,
        "chars": len(extracted_text),
        "message": f"Pomyślnie zindeksowano plik '{filename}' ({len(extracted_text)} znaków) w pamięci roboczej."
    }


@app.post("/api/clear_memory")
async def api_clear_memory():
    await assistant.wm.clear()
    return {"ok": True, "message": "Pamięć robocza została wyczyszczona."}


@app.post("/api/clear_chat")
async def api_clear_chat():
    assistant.chat_history.clear()
    return {"ok": True, "message": "Historia czatu została wyczyszczona."}


@app.post("/api/switch_model")
async def api_switch_model(req: SwitchModelRequest):
    threading.Thread(target=assistant.model_mgr.load_model, args=(req.model_name,), daemon=True).start()
    return {"ok": True, "message": f"Rozpoczęto ładowanie modelu {req.model_name}"}


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

            # Append user message
            assistant.chat_history.append({"role": "user", "content": user_msg})

            # Retrieve dynamic working memory context
            wm_context = await assistant.wm.get_context_for_prompt()

            # Stream response
            stream_gen = assistant.model_mgr.generate_streaming(
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

            # Auto-ingest query & response summary into working memory
            await assistant.wm.ingest(f"Użytkownik: {user_msg}\nAsystent: {complete_reply[:120]}", source="rozmowa")

            await websocket.send_text(json.dumps({
                "type": "done",
                "full_text": complete_reply
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


# =====================================================================
# 5. Clean, Modern Frontend (ChatGPT / Claude Style SPA)
# =====================================================================

HTML_FRONTEND = """<!DOCTYPE html>
<html lang="pl" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cognitive AI Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#f0f9ff',
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
    .markdown-content p { margin-bottom: 0.5rem; line-height: 1.5; }
    .markdown-content p:last-child { margin-bottom: 0; }
    .markdown-content ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.5rem; }
    .markdown-content code { background: rgba(0,0,0,0.3); padding: 0.15rem 0.35rem; border-radius: 4px; font-family: monospace; }
  </style>
</head>
<body class="text-slate-200 h-screen flex overflow-hidden">

  <!-- LEFT SIDEBAR: Context & Knowledge Manager -->
  <aside id="sidebar" class="w-80 bg-slate-900/90 border-r border-slate-800 flex flex-col transition-all duration-300 z-30">
    
    <!-- Sidebar Header -->
    <div class="p-4 border-b border-slate-800 flex items-center justify-between">
      <div class="flex items-center space-x-2">
        <div class="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse"></div>
        <span class="font-semibold text-sm tracking-wide text-white">Pamięć Robocza</span>
      </div>
      <button onclick="clearMemory()" title="Wyczyść pamięć" class="text-xs text-slate-400 hover:text-rose-400 p-1.5 rounded hover:bg-slate-800 transition">
        🗑️ Wyczyść
      </button>
    </div>

    <!-- Upload & Context Tabs -->
    <div class="p-4 space-y-4 flex-1 overflow-y-auto">
      
      <!-- Upload File Zone -->
      <div class="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-4 text-center cursor-pointer transition bg-slate-800/40"
           onclick="document.getElementById('fileInput').click()"
           ondragover="event.preventDefault()"
           ondrop="handleFileDrop(event)">
        <input type="file" id="fileInput" class="hidden" onchange="uploadSelectedFile(event)" accept=".pdf,.txt,.md,.json,.py,.csv,.log" />
        <div class="text-2xl mb-1">📄</div>
        <div class="text-xs font-medium text-slate-200">Wgraj dokument lub plik</div>
        <div class="text-[11px] text-slate-400 mt-1">PDF, TXT, MD, JSON, Python</div>
      </div>

      <!-- Quick Paste Context Area -->
      <div class="space-y-1.5">
        <div class="flex justify-between items-center text-xs font-medium text-slate-300">
          <span>Wklej treść / notatkę:</span>
          <button onclick="pasteFromClipboard()" class="text-[11px] text-sky-400 hover:underline">Wklej</button>
        </div>
        <textarea id="pasteContextInput" rows="3" placeholder="Wklej dowolny artykuł, umowę, kod lub notatki..."
                  class="w-full bg-slate-950/80 border border-slate-700 rounded-lg p-2.5 text-xs text-slate-200 placeholder-slate-500 outline-none focus:border-sky-500 transition resize-none"></textarea>
        <button onclick="submitPastedContext()" class="w-full bg-slate-800 hover:bg-slate-700 text-sky-400 font-medium py-1.5 rounded-lg text-xs border border-slate-700 transition">
          + Dodaj do kontekstu asystenta
        </button>
      </div>

      <!-- Clipboard Listener Toggle -->
      <div class="bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
        <div>
          <div class="text-xs font-semibold text-slate-200">Podsłuch schowka</div>
          <div class="text-[11px] text-slate-400">Automatycznie chłonie kopiowany tekst</div>
        </div>
        <button id="clipBtn" onclick="toggleClipboard()" class="px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400 transition">
          OFF
        </button>
      </div>

      <!-- Ingested Sources & Facts List -->
      <div class="space-y-2 pt-2">
        <div class="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
          <span>Aktywny kontekst (<span id="sourceCount">0</span>)</span>
          <span id="factBadge" class="text-[10px] text-emerald-400 font-mono">0 faktów</span>
        </div>
        <div id="sourcesList" class="space-y-1.5 text-xs">
          <div class="text-slate-500 italic text-[11px]">Brak wgranego kontekstu. Wgraj plik lub wklej tekst powyżej.</div>
        </div>
      </div>

    </div>

    <!-- Sidebar Footer Telemetry -->
    <div class="p-3 border-t border-slate-800 text-[11px] text-slate-400 flex justify-between items-center bg-slate-950/40">
      <span id="ramUsage">RAM: -- MB</span>
      <span id="deviceBadge" class="font-mono text-sky-400">MPS</span>
    </div>
  </aside>

  <!-- MAIN CHAT AREA -->
  <main class="flex-1 flex flex-col bg-[#0b0f19] relative">
    
    <!-- Top Header -->
    <header class="h-14 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-900/60 backdrop-blur">
      <div class="flex items-center space-x-3">
        <button onclick="toggleSidebar()" class="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-800 transition">
          ☰
        </button>
        <div class="font-bold text-sm text-white flex items-center gap-2">
          Cognitive Assistant <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">O(1) Memory</span>
        </div>
      </div>

      <!-- Model Selector & Status -->
      <div class="flex items-center space-x-3 text-xs">
        <span id="statusIndicator" class="text-slate-400 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400"></span> Gotowy
        </span>
        <select id="modelSelect" onchange="switchModel(this.value)" class="bg-slate-800 text-sky-300 border border-slate-700 rounded-lg px-2.5 py-1 text-xs outline-none focus:border-sky-500 cursor-pointer">
          <option value="SmolLM2-1.7B" selected>SmolLM2-1.7B (Szybki)</option>
          <option value="LLaMA-3-8B">LLaMA-3-8B (Mocny)</option>
          <option value="SmolLM2-135M">SmolLM2-135M (Lekki)</option>
        </select>
        <button onclick="clearChat()" class="text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition">
          Nowy czat
        </button>
      </div>
    </header>

    <!-- Chat Messages Container -->
    <div id="chatContainer" class="flex-1 overflow-y-auto p-6 space-y-4 max-w-4xl w-full mx-auto">
      
      <!-- Welcome Message -->
      <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-2">
        <div class="font-semibold text-sky-400 flex items-center gap-2">
          <span>🤖 Asystent Kognitywny</span>
        </div>
        <p class="leading-relaxed text-slate-200">
          Cześć! Jestem Twoim lokalnym asystentem AI. Posiadam aktywną pamięć roboczą – możesz wgrać dokument (PDF, TXT, notatki), wkleić artykuł po lewej stronie, lub włączyć podsłuch schowka.
        </p>
        <div class="pt-2 flex flex-wrap gap-2 text-[11px]">
          <button onclick="sendQuickPrompt('Kim jesteś i jak działasz?')" class="bg-slate-800 hover:bg-slate-700 text-sky-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            💡 Kim jesteś?
          </button>
          <button onclick="sendQuickPrompt('Podsumuj wgrany kontekst.')" class="bg-slate-800 hover:bg-slate-700 text-emerald-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            📜 Podsumuj kontekst
          </button>
          <button onclick="sendQuickPrompt('Jakie są kluczowe wnioski z moich materiałów?')" class="bg-slate-800 hover:bg-slate-700 text-amber-300 px-2.5 py-1 rounded-lg border border-slate-700 transition">
            🔍 Kluczowe wnioski
          </button>
        </div>
      </div>

    </div>

    <!-- Input Bar -->
    <div class="p-4 border-t border-slate-800/80 bg-slate-900/40">
      <div class="max-w-4xl mx-auto flex items-end gap-2">
        <div class="flex-1 bg-slate-900 border border-slate-700 focus-within:border-sky-500 rounded-2xl p-2 flex items-center gap-2 transition">
          <textarea id="messageInput" rows="1" placeholder="Napisz wiadomość lub zadaj pytanie o wgrany kontekst... (Enter aby wysłać)"
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
        Wszystkie obliczenia i modele działają w 100% lokalnie na Twoim komputerze Mac.
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
          refreshState();
        }
      };

      ws.onclose = () => {
        setTimeout(connectChatWS, 1500);
      };
    }

    async function refreshState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // RAM & Device
        document.getElementById('ramUsage').innerText = `RAM: ${data.telemetry.ram_rss_mb} MB`;
        document.getElementById('deviceBadge').innerText = data.telemetry.device;

        // Model Status
        const statusEl = document.getElementById('statusIndicator');
        if (data.telemetry.is_loading) {
          statusEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> ${data.telemetry.model_status}`;
        } else {
          statusEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> ${data.telemetry.model_status}`;
        }

        // Clipboard
        clipboardState = data.telemetry.clipboard_active;
        const clipBtn = document.getElementById('clipBtn');
        if (clipboardState) {
          clipBtn.innerText = 'ON';
          clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-emerald-600 bg-emerald-950 text-emerald-300';
        } else {
          clipBtn.innerText = 'OFF';
          clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-800 text-slate-400';
        }

        // Sources & Facts
        document.getElementById('sourceCount').innerText = data.memory.sources.length;
        document.getElementById('factBadge').innerText = `${data.memory.fact_count} faktów`;

        const sourcesList = document.getElementById('sourcesList');
        if (data.memory.sources.length > 0) {
          sourcesList.innerHTML = data.memory.sources.map(s => `
            <div class="bg-slate-950/80 border border-slate-800 p-2 rounded-lg flex items-center justify-between">
              <span class="truncate text-slate-200">${s.name}</span>
              <span class="text-[10px] text-slate-400">${Math.round(s.chars / 1000)}k zn.</span>
            </div>
          `).reverse().join('');
        } else {
          sourcesList.innerHTML = '<div class="text-slate-500 italic text-[11px]">Brak wgranego kontekstu. Wgraj plik lub wklej tekst powyżej.</div>';
        }
      } catch (e) {
        console.error(e);
      }
    }

    function sendMessage() {
      const input = document.getElementById('messageInput');
      const text = input.value.trim();
      if (!text || isGenerating) return;
      input.value = '';
      input.rows = 1;

      // Add user message
      const chatContainer = document.getElementById('chatContainer');
      const userDiv = document.createElement('div');
      userDiv.className = 'flex justify-end';
      userDiv.innerHTML = `
        <div class="chat-bubble-user p-3 px-4 rounded-2xl max-w-xl text-xs whitespace-pre-wrap">
          ${escapeHtml(text)}
        </div>
      `;
      chatContainer.appendChild(userDiv);

      // Create empty assistant message with typing effect
      const assistantDiv = document.createElement('div');
      assistantDiv.className = 'chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-1 typing-cursor';
      assistantDiv.rawText = '';
      assistantDiv.innerHTML = `
        <div class="font-semibold text-sky-400 text-[11px] mb-1">Asystent</div>
        <div class="markdown-content leading-relaxed text-slate-200"></div>
      `;
      chatContainer.appendChild(assistantDiv);
      currentAssistantBubble = assistantDiv;
      isGenerating = true;

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

    async function uploadSelectedFile(e) {
      const file = e.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/upload_file', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.ok) {
          alert(`✓ ${data.message}`);
          refreshState();
        } else {
          alert(`Błąd: ${data.error}`);
        }
      } catch (err) {
        alert('Wystąpił błąd podczas wysyłania pliku');
      }
    }

    function handleFileDrop(e) {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        uploadSelectedFile({ target: { files } });
      }
    }

    async function submitPastedContext() {
      const input = document.getElementById('pasteContextInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      const res = await fetch('/api/ingest_text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text, source_name: 'Wklejona notatka' })
      });
      const data = await res.json();
      if (data.ok) {
        refreshState();
      }
    }

    async function pasteFromClipboard() {
      try {
        const text = await navigator.clipboard.readText();
        document.getElementById('pasteContextInput').value = text;
      } catch (e) {
        alert('Zezwól przeglądarce na dostęp do schowka');
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

    async function switchModel(name) {
      await fetch('/api/switch_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ model_name: name })
      });
      refreshState();
    }

    async function clearMemory() {
      if (confirm('Czy na pewno chcesz wyczyścić pamięć roboczą asystenta?')) {
        await fetch('/api/clear_memory', { method: 'POST' });
        refreshState();
      }
    }

    async function clearChat() {
      await fetch('/api/clear_chat', { method: 'POST' });
      document.getElementById('chatContainer').innerHTML = `
        <div class="chat-bubble-assistant p-4 rounded-2xl max-w-2xl text-xs space-y-2">
          <div class="font-semibold text-sky-400">🤖 Nowy czat</div>
          <p class="text-slate-200">Pamięć robocza zachowana. O czym chcesz porozmawiać?</p>
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
      setInterval(refreshState, 3000);
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=HTML_FRONTEND)


# =====================================================================
# 6. Headless Verification Engine
# =====================================================================

async def run_headless_tests():
    """Verifies that the cognitive memory and neural model interfaces work."""
    print("\n=======================================================")
    print("Starting Baddeley Cognitive Working Memory Assistant Tests")
    print("=======================================================")

    test_wm = CognitiveWorkingMemory()

    print("[1/4] Testing Working Memory Ingestion & Distillation...")
    await test_wm.ingest("Pre-money wycena została ustalona na 45,000,000 EUR. Klucz: TITAN-9901.")
    await test_wm.ingest("Wdrożenie na klastrze produkcyjnym zaplanowano na piątek.")
    snap = await test_wm.snapshot()
    assert snap["fact_count"] > 0, "Facts were not ingested"
    print(f"      Distilled {snap['fact_count']} facts into working memory.")

    print("[2/4] Testing Dynamic Prompt Context Generation...")
    ctx = await test_wm.get_context_for_prompt()
    assert "45,000,000" in ctx or "TITAN" in ctx or "klastrze" in ctx
    print(f"      Generated prompt context preview ({len(ctx)} chars):\n      {ctx[:120]}...")

    print("[3/4] Testing Neural Model Manager Configuration...")
    mgr = NeuralModelManager()
    opts = mgr.get_model_options()
    assert len(opts) >= 3, "Model options missing"
    print(f"      Available models: {[o['id'] for o in opts]} on device: {mgr.device.upper()}")

    print("[4/4] Testing Memory Reset...")
    await test_wm.clear()
    snap_after = await test_wm.snapshot()
    assert snap_after["fact_count"] == 0, "Memory reset failed"
    print("      Memory cleared cleanly.")

    print("\n[TEST PASSED] All cognitive assistant tests completed successfully.\n")


# =====================================================================
# 7. Main Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Baddeley Cognitive Working Memory Assistant")
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
        time.sleep(1.2)
        webbrowser.open(url)

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n=======================================================")
    print(f"  🧠 BADDELEY COGNITIVE WORKING MEMORY ASSISTANT")
    print(f"  Local Web App: {url}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
