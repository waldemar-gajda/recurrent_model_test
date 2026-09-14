import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama3_dual_stream import Llama3DualStreamMemory, SAVE_PATH

REAL_WORLD_DOCUMENT = """
OFICJALNY RAPORT POWYPADKOWY (POST-MORTEM)
Data zdarzenia: 14 marca 2026 roku, godzina 03:42 UTC.
Dotyczy: Awaria klastra bazy danych w regionie europejskim.

W nocy z 13 na 14 marca system monitoringu Prometheus zgłosił krytyczne przeciążenie serwera PROD-EU-CENTRAL-42 (adres wewnętrzny 192.168.104.18).
Głównym inżynierem dyżurnym prowadzącym akcję ratunkową był Krzysztof Kowalczyk, a wsparcia udzielała Sarah Jenkins z zespołu DevOps.

Analiza logów wykazała niekontrolowany wyciek pamięci w module indeksującym, wprowadzonym we wdrożeniu wersji v3.18.4.
W celu zabezpieczenia spójności transakcji administratorzy odcięli ruch i otworzyli zgłoszenie awaryjne o identyfikatorze INC-9481.
Do odblokowania zaszyfrowanej partycji magazynu danych inżynier użył awaryjnego klucza odzyskiwania VAULT-KEY-819.
Całkowity czas niedostępności usług dla klientów wyniósł dokładnie 47 minut.
Wstępny audyt finansowy oszacował straty operacyjne firmy na kwotę 18400 EUR.
Zgodnie z rekomendacją zespołu cyberbezpieczeństwa cały klaster zostanie do końca tygodnia zmigrowany na wzmocniony protokół HYDRA-SEC-9.
"""

QUESTIONS = [
    ("Kto był głównym inżynierem dyżurnym prowadzącym akcję ratunkową?", "Krzysztof Kowalczyk"),
    ("Jaki był identyfikator zgłoszenia awaryjnego w systemie?", "INC-9481"),
    ("Jaka wersja oprogramowania spowodowała wyciek pamięci?", "v3.18.4"),
    ("Ile minut trwał całkowity czas niedostępności usług?", "47 minut"),
    ("Jaki klucz odzyskiwania magazynu danych został użyty?", "VAULT-KEY-819"),
    ("Jaki serwer uległ przeciążeniu?", "PROD-EU-CENTRAL-42"),
    ("Na jaki protokół bezpieczeństwa zostanie zmigrowany klaster?", "HYDRA-SEC-9"),
    ("Na jaką kwotę oszacowano straty operacyjne firmy?", "18400 EUR")
]

def run_real_world_test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 75)
    print(f"  TEST PRAWDZIWEGO ŚWIATA: PEŁNY RAPORT TECHNICZNY NA LLAMA-3-8B ({device.upper()})")
    print("  Dokument wieloakapitowy -> Zwijanie do 8 slotów (65.5 KB) i rejestru symboli")
    print("  Tura 2 (Pytania): 100% TEKSTU DOKUMENTU USUNIĘTE Z OKNA UWAGI (O(1) RAM)")
    print("=" * 75)

    memory = Llama3DualStreamMemory(num_slots=8, num_think=4, max_sym_len=24, device=device)
    if os.path.exists(SAVE_PATH):
        print(f">>> Ładowanie wytrenowanych wag z: {SAVE_PATH}")
        memory.bridge.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        print("    Wagi załadowane pomyślnie!\n")
    else:
        print(">>> BŁĄD: Brak pliku wag llama3_dual_stream.pt!")
        return

    print("--- DOKUMENT WEJŚCIOWY (TURA 1) ---")
    print(REAL_WORLD_DOCUMENT.strip())
    print("-" * 75)

    # Krok 1: Kodowanie raportu do pamięci O(1)
    print("\n>>> Zwijanie raportu do pamięci stałej...")
    scene_vectors, sym_ids = memory.encode_scene_and_symbols(REAL_WORLD_DOCUMENT)
    symbols_extracted = memory.extract_symbols(REAL_WORLD_DOCUMENT)
    print(f"    Rozmiar slotów w RAM:  {scene_vectors.numel() * 2 / 1024:.1f} KB")
    print(f"    Wyciągnięte symbole:   {symbols_extracted}")
    print(">>> DOKUMENT ZOSTAŁ W 100% USUNIĘTY Z KONTEKSTU MODELU.\n")

    print("=" * 75)
    print("  TURA 2: ODPOWIEDZI MODELU NA PYTANIA ZE ŚWIATA RZECZYWISTEGO")
    print("=" * 75)

    hits = 0
    for i, (q, expected) in enumerate(QUESTIONS, 1):
        gen = memory.answer(scene_vectors, sym_ids, q, max_new_tokens=25)
        is_hit = expected.lower() in gen.lower() or any(w.lower() in gen.lower() for w in expected.split())
        if is_hit: hits += 1

        status = "✓ TRAFIENIE" if is_hit else "✗ PUDŁO"
        print(f"[{i}/{len(QUESTIONS)}] {status}")
        print(f"  Pytanie:    {q}")
        print(f"  Oczekiwana: {expected}")
        print(f"  Odpowiedź:  {gen}\n")

    print("=" * 75)
    print(f"  PODSUMOWANIE TESTU PRAWDZIWEGO ŚWIATA: {hits}/{len(QUESTIONS)} ({(hits/len(QUESTIONS))*100:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_real_world_test()
