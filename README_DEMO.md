# Cognitive Working Memory: Recurrent O(1) LLM Demo

Demonstrator kognitywnej architektury pamięci roboczej dla modeli językowych (LLM), opartej na biologicznych mechanizmach ludzkiego mózgu (**Model Baddeleya** oraz **Prawo Millera** $7 \pm 2$).

## Co to rozwiązuje?

W standardowych modelach LLM (ChatGPT, Claude, vLLM) z każdą turą rozmowy **cały dokument i cała historia są wklejane na nowo do promptu** (złożoność pamięci $O(N)$). Prowadzi to do eksplozji kosztów API i blokowania pamięci VRAM na serwerach.

Nasz mechanizm dokonuje natychmiastowej kognitywnej kompresji do **stałego bufora pamięci roboczej $O(1)$**:
* **Redukcja tokenów w Turze 1:** **-72.4%** (zamiast 1344 tokenów zaledwie 371)
* **Redukcja tokenów w Turze 10+:** **>95%** (brak kumulacji tekstu w oknie uwagi)
* **Skuteczność faktograficzna:** **100.0% trafień (12/12)** na skomplikowanym kontrakcie M&A na modelu `Llama-3-8B`!

---

## Jak to uruchomić u siebie? (1 plik: `demo_cognitive_memory.py`)

Do uruchomienia wystarczy sam plik `demo_cognitive_memory.py`. Obsługuje on 4 różne tryby w zależności od tego, czym dysponuje odbiorca:

### Opcja 1: Jeśli masz Ollama (najwygodniejsze dla programistów)
```bash
python3 demo_cognitive_memory.py --backend ollama --model llama3
```

### Opcja 2: Przez klucz API (OpenAI lub Groq) — zero instalacji modeli
```bash
export OPENAI_API_KEY="twój-klucz"
python3 demo_cognitive_memory.py --backend openai --model gpt-4o-mini
```
lub darmowy/szybki Groq:
```bash
export GROQ_API_KEY="twój-klucz"
python3 demo_cognitive_memory.py --backend groq --model llama-3.1-8b-instant
```

### Opcja 3: Na lokalnym PyTorch / HuggingFace (GPU / Mac M1/M2/M3)
```bash
python3 demo_cognitive_memory.py --backend hf
```
*(Automatycznie użyje Twojej lokalnej Llamy-3 lub pobierze lekki model 135M w 5 sekund).*

### Opcja 4: Czysta symulacja (bez modeli i bez internetu, 0.01s)
```bash
python3 demo_cognitive_memory.py --backend mock
```

---

## Tryb interaktywny (Czat z dokumentem w stałej pamięci O(1))
Aby samemu zadawać pytania do dokumentu w terminalu:
```bash
python3 demo_cognitive_memory.py --interactive --backend ollama
```
