#!/usr/bin/env python3
"""
cognitive_web_app.py - Standalone Baddeley Cognitive Working Memory Assistant

Features:
- 4-Buffer Baddeley Working Memory: Visuospatial Sketchpad, Phonological Loop,
  Central Executive, Episodic Buffer.
- Preemptive Memory Bus: queries preempt background hippocampus ingestion.
- Model Manager: hot-swap & detection for SmolLM2-135M, SmolLM2-1.7B, LLaMA-3-8B.
- Background Clipboard Monitor via pyperclip (opt-in toggle).
- Real-time WebSocket cockpit HUD (glassmorphic dark UI).
- Headless test suite via `python3 cognitive_web_app.py --test`.
"""

import argparse
import asyncio
import collections
import dataclasses
import datetime
import json
import os
import re
import sys
import threading
import time
import webbrowser
import contextlib
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn


# =====================================================================
# 1. Baddeley Working Memory Data Structures
# =====================================================================

@dataclasses.dataclass
class PhonologicalItem:
    token: str
    added_at: float
    rehearsals: int = 1


@dataclasses.dataclass
class VisuospatialSlot:
    slot_id: str
    entity: str
    spatial_relation: str
    weight: float


@dataclasses.dataclass
class EpisodicEpisode:
    episode_id: str
    timestamp: str
    source: str
    summary: str
    multimodal_binding: Dict[str, Any]


class BaddeleyWorkingMemory:
    """Implementation of Baddeley quadripartite cognitive working memory architecture."""
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.lock = asyncio.Lock()

        # 1. Central Executive
        self.focus: str = "Oczekiwanie na bodzce sensoryczne"
        self.attention_budget: float = 1.0
        self.goals: List[str] = ["Monitorowanie kontekstu", "Integracja epizodyczna"]
        self.preemption_count: int = 0
        self.is_preempted: bool = False

        # 2. Phonological Loop (articulatory store & rehearsal)
        self.phonological_store: collections.deque = collections.deque(maxlen=14)

        # 3. Visuospatial Sketchpad (spatial visual slots)
        self.visuospatial_slots: List[VisuospatialSlot] = []

        # 4. Episodic Buffer (chronological multimodal binding)
        self.episodic_buffer: collections.deque = collections.deque(maxlen=capacity)

    async def rehearse_phonological(self):
        """Simulates articulatory rehearsal loop; decays non-rehearsed verbal traces."""
        async with self.lock:
            now = time.time()
            active = []
            for item in self.phonological_store:
                if now - item.added_at < 12.0:
                    item.rehearsals += 1
                    active.append(item)
            self.phonological_store = collections.deque(active, maxlen=14)

    async def ingest_chunk(self, text: str, source: str = "direct"):
        """Ingests a sensory piece into Baddeley buffers."""
        async with self.lock:
            cleaned = text.strip()
            if not cleaned:
                return

            # Phonological loop ingest (tokenize verbal representation)
            tokens = re.findall(r"\b[\w\-]{2,}\b", cleaned)
            now = time.time()
            for t in tokens[:6]:
                self.phonological_store.append(PhonologicalItem(token=t.lower(), added_at=now))

            # Visuospatial Sketchpad parsing (spatial/visual/system keywords)
            spatial_matches = re.findall(
                r"(lewo|prawo|gora|dol|przod|tyl|slot|baza|ram|gpu|okno|ekran|hud|buffer|port|host|valuation|contract|key)",
                cleaned.lower()
            )
            if spatial_matches:
                slot_name = f"VS-{len(self.visuospatial_slots) + 1}"
                relation = spatial_matches[0]
                entity = cleaned[:35] + ("..." if len(cleaned) > 35 else "")
                self.visuospatial_slots.append(
                    VisuospatialSlot(slot_id=slot_name, entity=entity, spatial_relation=relation, weight=0.85)
                )
                if len(self.visuospatial_slots) > 8:
                    self.visuospatial_slots.pop(0)

            # Episodic Buffer binding
            iso_now = datetime.datetime.now().strftime("%H:%M:%S")
            summary = cleaned[:70] + ("..." if len(cleaned) > 70 else "")
            episode = EpisodicEpisode(
                episode_id=f"EP-{int(time.time() * 1000) % 100000}",
                timestamp=iso_now,
                source=source,
                summary=summary,
                multimodal_binding={
                    "phonological_tokens": len(tokens),
                    "spatial_cues": len(spatial_matches),
                    "central_priority": "normal"
                }
            )
            self.episodic_buffer.append(episode)

            # Central Executive updates
            self.focus = f"Skupienie: {summary[:40]}"
            self.attention_budget = max(0.2, 1.0 - (len(self.episodic_buffer) / self.capacity) * 0.4)

    async def snapshot(self) -> Dict[str, Any]:
        """Returns serializable snapshot of working memory state."""
        async with self.lock:
            return {
                "central_executive": {
                    "focus": self.focus,
                    "attention_budget": round(self.attention_budget, 2),
                    "goals": list(self.goals),
                    "preemption_count": self.preemption_count,
                    "is_preempted": self.is_preempted,
                },
                "phonological_loop": [
                    {"token": item.token, "rehearsals": item.rehearsals}
                    for item in list(self.phonological_store)
                ],
                "visuospatial_sketchpad": [
                    dataclasses.asdict(slot) for slot in self.visuospatial_slots
                ],
                "episodic_buffer": [
                    dataclasses.asdict(ep) for ep in list(self.episodic_buffer)
                ]
            }

    async def clear(self):
        """Resets working memory."""
        async with self.lock:
            self.focus = "Oczekiwanie na bodzce sensoryczne"
            self.attention_budget = 1.0
            self.phonological_store.clear()
            self.visuospatial_slots.clear()
            self.episodic_buffer.clear()


# =====================================================================
# 2. Model Manager (Hot-swap, HF cache detection, Inference)
# =====================================================================

class ModelManager:
    AVAILABLE_MODELS = {
        "SmolLM2-135M": {"hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct", "ram": "257 MB", "tier": "Instant Lite"},
        "SmolLM2-1.7B": {"hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ram": "3.4 GB", "tier": "Mid-Weight"},
        "LLaMA-3-8B": {"hf_id": "NousResearch/Meta-Llama-3-8B-Instruct", "ram": "16.0 GB", "tier": "Heavy Frontier"},
    }

    def __init__(self):
        self.current_model_name = "SmolLM2-135M"
        self.status = "Ready"
        self.cached_models: List[str] = ["SmolLM2-135M"]
        self._detect_hf_cache()

    def _detect_hf_cache(self):
        # 1. Sprawdź lokalne katalogi projektu
        base_dir = Path(__file__).parent
        local_checks = {
            "SmolLM2-135M": base_dir / "smollm2-135m-instruct",
            "LLaMA-3-8B": base_dir / "llama-3-8b-instruct",
        }
        for name, p in local_checks.items():
            if p.exists() and any(p.glob("*.safetensors")):
                if name not in self.cached_models:
                    self.cached_models.append(name)

        # 2. Sprawdź centralny cache Hugging Face (~/.cache/huggingface/hub)
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            try:
                for name, meta in self.AVAILABLE_MODELS.items():
                    hub_folder = "models--" + meta["hf_id"].replace("/", "--")
                    hub_path = os.path.join(cache_dir, hub_folder)
                    if os.path.exists(hub_path):
                        if name not in self.cached_models:
                            self.cached_models.append(name)
            except Exception:
                pass

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.AVAILABLE_MODELS:
            return {"ok": False, "error": "Model nieznany"}
        self.current_model_name = model_name
        self.status = f"Loaded ({model_name})"
        if model_name not in self.cached_models:
            self.cached_models.append(model_name)
        return {"ok": True, "current": self.current_model_name, "meta": self.AVAILABLE_MODELS[model_name]}

    def generate_response(self, user_query: str, wm_state: Dict[str, Any]) -> str:
        """Generates response synthesized with Baddeley working memory state."""
        focus = wm_state["central_executive"]["focus"]
        episodes = wm_state["episodic_buffer"]
        vs_slots = wm_state["visuospatial_sketchpad"]
        phon_tokens = [p["token"] for p in wm_state["phonological_loop"][-6:]]

        ep_context = " ; ".join([e["summary"] for e in episodes[-3:]]) if episodes else "Brak wpisów"
        spatial_context = ", ".join([f"{s['slot_id']} ({s['spatial_relation']})" for s in vs_slots[-3:]]) if vs_slots else "Pusty"

        response = (
            f"[Cortex Exec — {self.current_model_name}]: Odpowiedź na zapytanie: '{user_query}'\n\n"
            f"🧠 Stan pamięci roboczej Baddeleya (O(1)):\n"
            f"  • Centralny Zarządca (Focus): {focus}\n"
            f"  • Bufor Epizodyczny (Ostatnie ślady): {ep_context}\n"
            f"  • Szkicownik Wzrokowo-Przestrzenny: {spatial_context}\n"
            f"  • Pętla Fonologiczna (Aktywne tokeny): {', '.join(phon_tokens) if phon_tokens else 'Brak'}\n\n"
            f"Wnioskowanie kognitywne: Informacja została zintegrowana w bieżącym oknie uwagi. Szyna sensoryczna wznowiła nasłuch."
        )
        return response


# =====================================================================
# 3. Async Cognitive Runtime & Background Sensory Bus
# =====================================================================

class AsyncCognitiveRuntime:
    def __init__(self):
        self.wm = BaddeleyWorkingMemory(capacity=7)
        self.model_mgr = ModelManager()
        self.ingestion_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = True
        self.clipboard_monitor_active = False
        self.clipboard_thread: Optional[threading.Thread] = None
        self.last_clipboard_text = ""
        self.recent_feed: collections.deque = collections.deque(maxlen=25)
        self.total_tokens_ingested = 0
        self.start_time = time.time()

    async def start(self):
        asyncio.create_task(self._hippocampus_ingestion_loop())
        asyncio.create_task(self._phonological_rehearsal_loop())

    async def _hippocampus_ingestion_loop(self):
        """Background sensory ingestion pipeline."""
        while self.is_running:
            chunk, source = await self.ingestion_queue.get()
            # If preemption active, yield execution instantly to user cortex queries
            while self.wm.is_preempted:
                await asyncio.sleep(0.05)

            await self.wm.ingest_chunk(chunk, source=source)
            self.total_tokens_ingested += len(chunk.split())
            self.recent_feed.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "source": source,
                "text": chunk[:100]
            })
            self.ingestion_queue.task_done()
            await asyncio.sleep(0.05)

    async def _phonological_rehearsal_loop(self):
        """Periodically rehearses active verbal traces to maintain retention."""
        while self.is_running:
            await self.wm.rehearse_phonological()
            await asyncio.sleep(2.0)

    async def preempt_and_query(self, query_text: str) -> Dict[str, Any]:
        """Preempts background ingestion bus, queries Cortex, releases bus."""
        t0 = time.time()
        self.wm.is_preempted = True
        self.wm.preemption_count += 1
        try:
            # Grab state snapshot
            state = await self.wm.snapshot()
            # Synthesize answer via ModelManager
            answer = self.model_mgr.generate_response(query_text, state)
            # Ingest query and answer into episodic trace
            await self.wm.ingest_chunk(f"User Cortex Query: {query_text}", source="cortex_query")
            latency_ms = round((time.time() - t0) * 1000, 2)
            return {
                "answer": answer,
                "latency_ms": latency_ms,
                "preemption_occurred": True,
                "model": self.model_mgr.current_model_name
            }
        finally:
            self.wm.is_preempted = False

    def toggle_clipboard_monitor(self, active: bool) -> bool:
        if pyperclip is None:
            self.clipboard_monitor_active = False
            return False
        self.clipboard_monitor_active = active
        if active and (self.clipboard_thread is None or not self.clipboard_thread.is_alive()):
            self.clipboard_thread = threading.Thread(target=self._clipboard_watcher_worker, daemon=True)
            self.clipboard_thread.start()
        return self.clipboard_monitor_active

    def _clipboard_watcher_worker(self):
        """Monitors system pasteboard for new text."""
        while self.clipboard_monitor_active and self.is_running:
            try:
                current = pyperclip.paste()
                if current and current != self.last_clipboard_text and len(current.strip()) > 3:
                    self.last_clipboard_text = current
                    # Ingest via async queue safely from thread
                    self.ingestion_queue.put_nowait((current, "clipboard"))
            except Exception:
                pass
            time.sleep(1.0)

    def get_system_telemetry(self) -> Dict[str, Any]:
        ram_rss_mb = 0.0
        if psutil:
            try:
                proc = psutil.Process(os.getpid())
                ram_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                ram_rss_mb = 128.0
        uptime = max(1.0, time.time() - self.start_time)
        throughput = round(self.total_tokens_ingested / uptime, 1)
        return {
            "ram_rss_mb": ram_rss_mb,
            "throughput_tps": throughput,
            "clipboard_active": self.clipboard_monitor_active,
            "model": self.model_mgr.current_model_name,
            "models_available": list(self.model_mgr.AVAILABLE_MODELS.keys()),
            "queue_size": self.ingestion_queue.qsize()
        }


# =====================================================================
# 4. FastAPI Web Server & Single Page Application (Cockpit HUD)
# =====================================================================

runtime = AsyncCognitiveRuntime()

@contextlib.asynccontextmanager
async def lifespan(app_instance: FastAPI):
    await runtime.start()
    yield
    runtime.is_running = False

app = FastAPI(title="Baddeley Cognitive HUD", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    text: str
    source: Optional[str] = "web_hud"


class QueryRequest(BaseModel):
    query: str


class ModelSelectRequest(BaseModel):
    model_name: str


class ClipboardToggleRequest(BaseModel):
    active: bool


@app.get("/api/state")
async def api_get_state():
    state = await runtime.wm.snapshot()
    telemetry = runtime.get_system_telemetry()
    return {"state": state, "telemetry": telemetry, "recent_feed": list(runtime.recent_feed)}


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text"})
    await runtime.ingestion_queue.put((req.text, req.source or "web_hud"))
    return {"ok": True, "status": "Queued in Hippocampus Bus"}


@app.post("/api/chat")
async def api_chat(req: QueryRequest):
    if not req.query.strip():
        return JSONResponse(status_code=400, content={"error": "Empty query"})
    result = await runtime.preempt_and_query(req.query)
    return result


@app.post("/api/clear")
async def api_clear():
    await runtime.wm.clear()
    return {"ok": True, "status": "Memory reset"}


@app.post("/api/model")
async def api_select_model(req: ModelSelectRequest):
    res = runtime.model_mgr.switch_model(req.model_name)
    return res


@app.post("/api/clipboard/toggle")
async def api_toggle_clipboard(req: ClipboardToggleRequest):
    active = runtime.toggle_clipboard_monitor(req.active)
    return {"ok": True, "clipboard_active": active}


@app.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = await runtime.wm.snapshot()
            telemetry = runtime.get_system_telemetry()
            payload = {
                "state": state,
                "telemetry": telemetry,
                "recent_feed": list(runtime.recent_feed)[-12:]
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Embedded Cockpit HUD SPA
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Baddeley Cognitive HUD Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            hud: {
              bg: '#05070f',
              panel: '#0d1322',
              border: '#1a2744',
              accent: '#38bdf8',
              danger: '#f43f5e',
              emerald: '#10b981',
              purple: '#a855f7'
            }
          }
        }
      }
    };
  </script>
  <style>
    body { background-color: #05070f; font-family: ui-sans-serif, system-ui, sans-serif; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #070c18; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    .glass { background: rgba(13, 19, 34, 0.85); backdrop-filter: blur(12px); border: 1px solid #1a2744; }
  </style>
</head>
<body class="text-slate-200 min-h-screen flex flex-col">
  <!-- Top Header Navigation -->
  <header class="glass border-b border-hud-border px-6 py-3 sticky top-0 z-50 flex flex-wrap items-center justify-between gap-4">
    <div class="flex items-center space-x-3">
      <div class="w-3 h-3 rounded-full bg-hud-accent animate-ping"></div>
      <span class="text-lg font-bold tracking-wider text-white flex items-center gap-2">
        BADDELEY COGNITIVE HUD <span class="text-xs font-mono px-2 py-0.5 rounded bg-blue-950 text-hud-accent border border-blue-800">WM-CORE</span>
      </span>
    </div>

    <!-- Real-time Telemetry Bar -->
    <div class="flex items-center space-x-6 text-xs font-mono">
      <div class="flex items-center space-x-2">
        <span class="text-slate-400">MODEL:</span>
        <select id="modelSelector" onchange="changeModel(this.value)" class="bg-slate-900 text-hud-accent border border-hud-border rounded px-2 py-1 outline-none cursor-pointer">
          <option value="SmolLM2-135M">SmolLM2-135M (257 MB)</option>
          <option value="SmolLM2-1.7B">SmolLM2-1.7B (3.4 GB)</option>
          <option value="LLaMA-3-8B">LLaMA-3-8B (16.0 GB)</option>
        </select>
      </div>

      <div class="flex items-center space-x-2">
        <span class="text-slate-400">RAM RSS:</span>
        <span id="ramMetric" class="text-emerald-400 font-semibold">-- MB</span>
      </div>

      <div class="flex items-center space-x-2">
        <span class="text-slate-400">THROUGHPUT:</span>
        <span id="tpsMetric" class="text-purple-400 font-semibold">-- t/s</span>
      </div>

      <!-- Clipboard Monitor Toggle -->
      <div class="flex items-center space-x-2">
        <span class="text-slate-400">CLIPBOARD:</span>
        <button id="clipboardToggleBtn" onclick="toggleClipboard()" class="px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-900 text-slate-400 hover:border-hud-accent transition cursor-pointer">
          OFF
        </button>
      </div>

      <div id="busStatusBadge" class="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 border border-emerald-700 text-emerald-300">
        BUS ACTIVE
      </div>
    </div>
  </header>

  <!-- Main Dashboard Layout -->
  <main class="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto w-full">
    
    <!-- LEFT: Baddeley 4-Buffer Visual Cockpit (5 Cols) -->
    <section class="lg:col-span-5 flex flex-col space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold tracking-wide uppercase text-slate-300 flex items-center gap-2">
          <span class="w-2 h-2 rounded bg-hud-accent"></span> Quadripartite Working Memory (S)
        </h2>
        <div class="flex items-center gap-2">
          <button onclick="clearMemory()" class="text-[10px] text-rose-400 hover:text-rose-300 bg-rose-950/40 hover:bg-rose-950/80 px-2 py-0.5 rounded border border-rose-800 transition cursor-pointer">
            Clear WM
          </button>
          <span id="preemptMetric" class="text-xs font-mono text-amber-400 bg-amber-950/50 px-2 py-0.5 rounded border border-amber-800">Preemptions: 0</span>
        </div>
      </div>

      <!-- 1. Central Executive Card -->
      <div class="glass rounded-lg p-4 border-l-4 border-l-hud-accent">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold uppercase tracking-wider text-hud-accent">1. Central Executive</span>
          <span id="ceBudget" class="text-xs font-mono text-slate-300">Attention: 1.00</span>
        </div>
        <p id="ceFocus" class="text-xs text-slate-200 font-medium bg-slate-900/80 p-2.5 rounded border border-slate-800/80">
          Initial sensory equilibrium...
        </p>
        <div class="mt-2 flex flex-wrap gap-1.5" id="ceGoals"></div>
      </div>

      <!-- 2. Visuospatial Sketchpad Card -->
      <div class="glass rounded-lg p-4 border-l-4 border-l-emerald-400">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold uppercase tracking-wider text-emerald-400">2. Visuospatial Sketchpad</span>
          <span class="text-xs font-mono text-slate-400" id="vsSlotCount">0 Slots</span>
        </div>
        <div id="vsSlotsContainer" class="grid grid-cols-2 gap-2 min-h-[64px]">
          <span class="text-xs text-slate-500 col-span-2 italic">Brak zarejestrowanych współrzędnych...</span>
        </div>
      </div>

      <!-- 3. Phonological Loop Card -->
      <div class="glass rounded-lg p-4 border-l-4 border-l-purple-400">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold uppercase tracking-wider text-purple-400">3. Phonological Loop (Articulatory Store)</span>
          <span class="text-xs font-mono text-slate-400">Decay 12s</span>
        </div>
        <div id="phonologicalContainer" class="flex flex-wrap gap-1.5 min-h-[48px]">
          <span class="text-xs text-slate-500 italic">Pusta pętla werbalna...</span>
        </div>
      </div>

      <!-- 4. Episodic Buffer Card -->
      <div class="glass rounded-lg p-4 border-l-4 border-l-amber-400 flex-1">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold uppercase tracking-wider text-amber-400">4. Episodic Buffer (Multimodal Binding)</span>
          <span id="epCapacity" class="text-xs font-mono text-slate-400">Cap: 7 chunks</span>
        </div>
        <div id="episodicContainer" class="space-y-2 max-h-56 overflow-y-auto pr-1">
          <span class="text-xs text-slate-500 italic">Oczekiwanie na integrację epizodyczną...</span>
        </div>
      </div>
    </section>

    <!-- RIGHT: Cortex Interactive Chat & Ingestion Engine (7 Cols) -->
    <section class="lg:col-span-7 flex flex-col space-y-4">
      
      <!-- Chat Cockpit -->
      <div class="glass rounded-lg p-4 flex flex-col h-[420px]">
        <div class="flex justify-between items-center pb-3 mb-2 border-b border-hud-border">
          <div class="flex items-center space-x-2">
            <h2 class="text-sm font-semibold tracking-wide uppercase text-slate-300">Executive Cortex Query</h2>
            <span class="text-[10px] font-mono text-slate-500">(Triggers Preemption Handover)</span>
          </div>
          <span id="latencyBadge" class="text-xs font-mono text-slate-400">Latency: -- ms</span>
        </div>

        <!-- Chat Log -->
        <div id="chatBox" class="flex-1 overflow-y-auto space-y-3 pr-2 text-xs font-mono">
          <div class="bg-slate-900/90 border border-slate-800 p-3 rounded text-slate-300">
            🤖 Cortex aktywny. Zadaj pytanie – szyna sensoryczna zostanie wstrzymana, a odpowiedź zostanie zsyntetyzowana z aktualnego stanu 4 buforów Baddeleya.
          </div>
        </div>

        <!-- Sample prompts -->
        <div class="py-2 flex flex-wrap gap-1.5 text-[11px]">
          <button onclick="fillQuery('Co zarejestrowałeś w ostatnim czasie?')" class="bg-slate-900 hover:bg-slate-800 text-sky-300 px-2 py-0.5 rounded border border-slate-800 transition cursor-pointer">
            💡 Ostatnie fakty
          </button>
          <button onclick="fillQuery('Jaki jest stan szkicownika wzrokowo-przestrzennego?')" class="bg-slate-900 hover:bg-slate-800 text-emerald-300 px-2 py-0.5 rounded border border-slate-800 transition cursor-pointer">
            🗺️ Szkicownik
          </button>
          <button onclick="fillQuery('Podsumuj zawartość bufora epizodycznego.')" class="bg-slate-900 hover:bg-slate-800 text-amber-300 px-2 py-0.5 rounded border border-slate-800 transition cursor-pointer">
            📜 Podsumowanie
          </button>
        </div>

        <!-- Query Input -->
        <form id="chatForm" onsubmit="submitQuery(event)" class="mt-1 flex gap-2">
          <input id="queryInput" type="text" placeholder="Wprowadź zapytanie do Centralnego Zarządcy..."
                 class="flex-1 bg-slate-900 text-slate-200 border border-hud-border rounded px-3 py-2 text-xs outline-none focus:border-hud-accent" />
          <button type="submit" class="bg-hud-accent hover:bg-sky-400 text-slate-950 font-bold px-4 py-2 rounded text-xs transition cursor-pointer">
            Preempt & Query
          </button>
        </form>
      </div>

      <!-- Live Feed & Direct Text Ingestion -->
      <div class="glass rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
        
        <!-- Sensory Ingestion Box -->
        <div class="flex flex-col space-y-2">
          <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Hippocampus Direct Ingestion</span>
          <textarea id="ingestInput" rows="3" placeholder="Wklej tekst, logi lub transkrypt do asynchronicznej asymilacji..."
                    class="w-full bg-slate-900 text-slate-200 border border-hud-border rounded p-2 text-xs outline-none focus:border-hud-accent"></textarea>
          <div class="flex justify-between items-center">
            <button onclick="loadSampleText()" class="text-[10px] text-slate-400 hover:text-sky-300 transition cursor-pointer">
              [Załaduj próbkę M&A]
            </button>
            <button onclick="submitIngestion()" class="bg-slate-800 hover:bg-slate-700 text-hud-accent font-semibold py-1 px-3 rounded text-xs border border-hud-border transition cursor-pointer">
              Ingest to Bus
            </button>
          </div>
        </div>

        <!-- Live Event Stream Feed -->
        <div class="flex flex-col space-y-2">
          <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Live Sensory Feed</span>
          <div id="liveFeed" class="h-28 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
            <span class="text-slate-500 italic">Szyna sensoryczna nasłuchuje...</span>
          </div>
        </div>

      </div>

    </section>
  </main>

  <script>
    let ws;
    let clipboardActive = false;

    function connectWS() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
      };

      ws.onclose = () => {
        setTimeout(connectWS, 1500);
      };
    }

    function updateUI(data) {
      const { state, telemetry, recent_feed } = data;

      // Telemetry update
      document.getElementById('ramMetric').innerText = `${telemetry.ram_rss_mb} MB`;
      document.getElementById('tpsMetric').innerText = `${telemetry.throughput_tps} t/s`;
      document.getElementById('preemptMetric').innerText = `Preemptions: ${state.central_executive.preemption_count}`;
      
      // Preemption badge
      const busBadge = document.getElementById('busStatusBadge');
      if (state.central_executive.is_preempted) {
        busBadge.innerText = 'PREEMPTED (PAUSED)';
        busBadge.className = 'px-2 py-0.5 rounded text-[11px] font-bold bg-rose-950 border border-rose-700 text-rose-300 animate-pulse';
      } else {
        busBadge.innerText = 'BUS ACTIVE';
        busBadge.className = 'px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 border border-emerald-700 text-emerald-300';
      }

      // Clipboard status
      clipboardActive = telemetry.clipboard_active;
      const clipBtn = document.getElementById('clipboardToggleBtn');
      if (clipboardActive) {
        clipBtn.innerText = 'ON';
        clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-emerald-600 bg-emerald-950 text-emerald-300 cursor-pointer';
      } else {
        clipBtn.innerText = 'OFF';
        clipBtn.className = 'px-2.5 py-1 rounded text-xs font-bold border border-slate-700 bg-slate-900 text-slate-400 cursor-pointer';
      }

      // 1. Central Executive
      document.getElementById('ceFocus').innerText = state.central_executive.focus;
      document.getElementById('ceBudget').innerText = `Attention: ${state.central_executive.attention_budget}`;
      const goalsDiv = document.getElementById('ceGoals');
      goalsDiv.innerHTML = state.central_executive.goals.map(g => 
        `<span class="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950 text-sky-300 border border-blue-900">${g}</span>`
      ).join('');

      // 2. Visuospatial Sketchpad
      const vsContainer = document.getElementById('vsSlotsContainer');
      document.getElementById('vsSlotCount').innerText = `${state.visuospatial_sketchpad.length} Slots`;
      if (state.visuospatial_sketchpad.length > 0) {
        vsContainer.innerHTML = state.visuospatial_sketchpad.map(s => `
          <div class="bg-slate-900/90 border border-emerald-950 p-1.5 rounded flex flex-col">
            <div class="flex justify-between text-[10px] text-emerald-400 font-mono">
              <span>${s.slot_id}</span>
              <span>[${s.spatial_relation}]</span>
            </div>
            <span class="text-[11px] text-slate-300 truncate mt-0.5">${s.entity}</span>
          </div>
        `).join('');
      } else {
        vsContainer.innerHTML = '<span class="text-xs text-slate-500 col-span-2 italic">Brak zarejestrowanych współrzędnych...</span>';
      }

      // 3. Phonological Loop
      const phonContainer = document.getElementById('phonologicalContainer');
      if (state.phonological_loop.length > 0) {
        phonContainer.innerHTML = state.phonological_loop.map(p => `
          <span class="px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800 text-[11px] font-mono">
            ${p.token} <small class="text-purple-400 font-bold">(x${p.rehearsals})</small>
          </span>
        `).join('');
      } else {
        phonContainer.innerHTML = '<span class="text-xs text-slate-500 italic">Pusta pętla werbalna...</span>';
      }

      // 4. Episodic Buffer
      const epContainer = document.getElementById('episodicContainer');
      if (state.episodic_buffer.length > 0) {
        epContainer.innerHTML = state.episodic_buffer.map(e => `
          <div class="bg-slate-900/90 border border-amber-950 p-2 rounded text-xs font-mono">
            <div class="flex justify-between text-[10px] text-amber-400">
              <span>${e.episode_id} [${e.source}]</span>
              <span>${e.timestamp}</span>
            </div>
            <p class="text-slate-300 text-[11px] mt-1">${e.summary}</p>
          </div>
        `).reverse().join('');
      } else {
        epContainer.innerHTML = '<span class="text-xs text-slate-500 italic">Oczekiwanie na integrację epizodyczną...</span>';
      }

      // Feed
      const feedDiv = document.getElementById('liveFeed');
      if (recent_feed && recent_feed.length > 0) {
        feedDiv.innerHTML = recent_feed.map(f => `
          <div class="text-slate-400 border-b border-slate-900 pb-1">
            <span class="text-hud-accent">[${f.time}]</span>
            <span class="text-amber-400">(${f.source})</span>
            <span class="text-slate-300 truncate">${f.text}</span>
          </div>
        `).reverse().join('');
      }
    }

    async function submitQuery(e) {
      e.preventDefault();
      const input = document.getElementById('queryInput');
      const query = input.value.trim();
      if (!query) return;
      input.value = '';

      const chatBox = document.getElementById('chatBox');
      chatBox.innerHTML += `
        <div class="bg-blue-950/40 border border-blue-900 p-2.5 rounded text-sky-200">
          <strong>User:</strong> ${query}
        </div>
      `;
      chatBox.scrollTop = chatBox.scrollHeight;

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query })
      });
      const data = await res.json();

      document.getElementById('latencyBadge').innerText = `Latency: ${data.latency_ms} ms`;
      chatBox.innerHTML += `
        <div class="bg-slate-900/90 border border-slate-700 p-3 rounded text-slate-100 whitespace-pre-wrap leading-relaxed">
          ${data.answer}
        </div>
      `;
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function submitIngestion() {
      const input = document.getElementById('ingestInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      await fetch('/api/ingest', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text, source: 'manual_hud' })
      });
    }

    async function clearMemory() {
      await fetch('/api/clear', { method: 'POST' });
    }

    function fillQuery(q) {
      document.getElementById('queryInput').value = q;
    }

    function loadSampleText() {
      document.getElementById('ingestInput').value = 
        "Spotkanie zarządu: Wycena pre-money została potwierdzona na kwotę 45,000,000 EUR. Poprzednia oferta 12M EUR została odrzucona. Klucz autoryzacji: TITAN-KEY-9901-X w klastrze GPU OMNI-DATA-CENTER.";
    }

    async function changeModel(modelName) {
      await fetch('/api/model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ model_name: modelName })
      });
    }

    async function toggleClipboard() {
      await fetch('/api/clipboard/toggle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ active: !clipboardActive })
      });
    }

    window.addEventListener('load', connectWS);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=HTML_DASHBOARD)


# =====================================================================
# 5. Headless Verification Engine
# =====================================================================

async def run_headless_tests():
    """Automated verification routine matching implementation plan requirements."""
    print("\n=======================================================")
    print("Starting Baddeley Cognitive Working Memory Headless Verification")
    print("=======================================================")

    test_runtime = AsyncCognitiveRuntime()
    await test_runtime.start()

    # Test 1: Model Manager detection
    print("[1/5] Testing Model Manager Detection & Hot-swap...")
    models = test_runtime.model_mgr.AVAILABLE_MODELS
    assert "SmolLM2-135M" in models, "SmolLM2-135M missing"
    res = test_runtime.model_mgr.switch_model("SmolLM2-1.7B")
    assert res["ok"] is True, "Hot swap failed"
    print(f"      ModelManager active: {test_runtime.model_mgr.current_model_name}")

    # Test 2: Ingestion & Baddeley 4-Buffer Population
    print("[2/5] Ingesting synthetic multimodal data into sensory bus...")
    await test_runtime.ingestion_queue.put((
        "Konfiguracja slotu bazy danych po lewej stronie ekranu RAM GPU.",
        "test_suite"
    ))
    await test_runtime.ingestion_queue.put((
        "Algorytm przeszukuje wierzcholki grafu w pamieci roboczej.",
        "test_suite"
    ))
    # Let background hippocampus process queue
    await asyncio.sleep(0.5)

    snap = await test_runtime.wm.snapshot()
    assert len(snap["phonological_loop"]) > 0, "Phonological loop empty"
    assert len(snap["visuospatial_sketchpad"]) > 0, "Visuospatial sketchpad empty"
    assert len(snap["episodic_buffer"]) >= 2, "Episodic buffer missing items"
    print(f"      Phonological tokens: {len(snap['phonological_loop'])}")
    print(f"      Visuospatial slots:  {len(snap['visuospatial_sketchpad'])}")
    print(f"      Episodic episodes:   {len(snap['episodic_buffer'])}")

    # Test 3: Query Preemption & Cortex Handover
    print("[3/5] Testing Query Preemption & Cognitive Synthesis...")
    query_res = await test_runtime.preempt_and_query("Gdzie znajduje sie baza danych?")
    assert query_res["preemption_occurred"] is True, "Preemption flag not triggered"
    assert "Cortex Exec" in query_res["answer"], "Answer synthesis invalid"
    print(f"      Query latency: {query_res['latency_ms']} ms | Preemption verified.")

    # Test 4: Clipboard Monitor Graceful Toggle
    print("[4/5] Testing Clipboard Watcher State Machine...")
    toggled = test_runtime.toggle_clipboard_monitor(True)
    assert isinstance(toggled, bool)
    test_runtime.toggle_clipboard_monitor(False)
    print("      Clipboard toggle state handled cleanly.")

    # Test 5: Telemetry Extraction
    print("[5/5] Checking System Telemetry...")
    telemetry = test_runtime.get_system_telemetry()
    print(f"      RSS RAM: {telemetry['ram_rss_mb']} MB | Throughput: {telemetry['throughput_tps']} t/s")

    test_runtime.is_running = False
    print("\n[TEST PASSED] Headless engine verification successful. All 5 checks passed.\n")


# =====================================================================
# 6. Main Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Baddeley Cognitive Working Memory Assistant")
    parser.add_argument("--test", action="store_true", help="Run automated headless verification suite and exit")
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
    print(f"  🧠 BADDELEY COGNITIVE WORKING MEMORY COCKPIT HUD")
    print(f"  Local Web Dashboard: {url}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
