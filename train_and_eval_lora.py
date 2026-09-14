import os
import sys
import time
import torch
import torch.nn as nn

from lora_dual_stream_memory import AttentionLoRADualStreamMemory
from dataset_memory_recall import generate_dataset

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "smollm2-135m-instruct"
NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WEIGHTS_FILE = "smollm_lora_attention.pt" if "smollm" in MODEL_PATH.lower() else "llama3_lora_attention.pt"
SAVE_PATH = os.path.expanduser(f"~/teamwork_projects/recurrent_model_test/{WEIGHTS_FILE}")

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

RW_QUESTIONS = [
    ("Kto był głównym inżynierem dyżurnym prowadzącym akcję ratunkową?", "Krzysztof Kowalczyk"),
    ("Jaki był identyfikator zgłoszenia awaryjnego w systemie?", "INC-9481"),
    ("Jaka wersja oprogramowania spowodowała wyciek pamięci?", "v3.18.4"),
    ("Ile minut trwał całkowity czas niedostępności usług?", "47 minut"),
    ("Jaki klucz odzyskiwania magazynu danych został użyty?", "VAULT-KEY-819"),
    ("Jaki serwer uległ przeciążeniu?", "PROD-EU-CENTRAL-42"),
    ("Na jaki protokół bezpieczeństwa zostanie zmigrowany klaster?", "HYDRA-SEC-9"),
    ("Na jaką kwotę oszacowano straty operacyjne firmy?", "18400 EUR")
]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 72)
    print(f"  TRENING: CHIRURGICZNA LoRA UWAGI (W_q, W_v) + PAMIĘĆ DUAL-STREAM")
    print(f"  Model: {MODEL_PATH} ({device.upper()}) | Epoki: {NUM_EPOCHS}")
    print(f"  Zasada: 100% MLP (wiedza o świecie) ZAMROŻONE! LoRA tylko na uwadze.")
    print("=" * 72)

    memory = AttentionLoRADualStreamMemory(MODEL_PATH, num_slots=8, num_think=4, r=8, device=device)
    train_data = generate_dataset(num_samples=200, base_seed=42)
    test_data = generate_dataset(num_samples=15, base_seed=1234)

    trainable_params = memory.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)

    best_loss = 999.0
    print(f">>> Start treningu ({NUM_EPOCHS} epok)...")

    for epoch in range(NUM_EPOCHS):
        t_start = time.time()
        total_loss = 0.0

        for p, q, a in train_data:
            optimizer.zero_grad()
            scene_v, sym_ids = memory.encode_scene_and_symbols(p)
            loss = memory.forward_qa(scene_v, sym_ids, q, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_start
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoka {epoch + 1:02d}/{NUM_EPOCHS:02d}] Loss: {avg_loss:.4f} | LR: {cur_lr:.5f} | ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "lora_state": [p.data for p in memory.lora_params],
                "empty_slots": memory.empty_slots.data,
                "think_tokens": memory.think_tokens.data
            }, SAVE_PATH)

    print(f"\n>>> Zapisano wagi do {SAVE_PATH}!")

    # 1. Ewaluacja standardowa
    print("\n" + "=" * 72)
    print("  CZĘŚĆ 1: STANDARDOWY BENCHMARK (15 NOWYCH DOKUMENTÓW)")
    print("=" * 72)
    hits = 0
    for i, (p, q, expected) in enumerate(test_data, 1):
        scene_v, sym_ids = memory.encode_scene_and_symbols(p)
        gen = memory.answer(scene_v, sym_ids, q)
        is_hit = expected.lower() in gen.lower() or gen.lower() in expected.lower()
        if is_hit: hits += 1
        status = "✓ TRAF" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/15] {status} | Oczekiwana: {expected:22s} | Odpowiedź: {gen}")

    print(f"\n>>> Wynik standardowy: {hits}/15 ({(hits/15)*100:.1f}%)")

    # 2. Ewaluacja na raporcie ze świata rzeczywistego (Post-Mortem)
    print("\n" + "=" * 72)
    print("  CZĘŚĆ 2: TEST PRAWDZIWEGO ŚWIATA (RAPORT POWYPADKOWY POST-MORTEM)")
    print("=" * 72)
    scene_rw, sym_rw = memory.encode_scene_and_symbols(REAL_WORLD_DOCUMENT)
    print(f">>> Raport zwinięty do pamięci O(1). Wyciągnięte symbole:")
    print(f"    {memory.extract_symbols(REAL_WORLD_DOCUMENT)}\n")

    hits_rw = 0
    for i, (q, expected) in enumerate(RW_QUESTIONS, 1):
        gen = memory.answer(scene_rw, sym_rw, q)
        is_hit = expected.lower() in gen.lower() or any(w.lower() in gen.lower() for w in expected.split())
        if is_hit: hits_rw += 1
        status = "✓ TRAF" if is_hit else "✗ PUDŁO"
        print(f"[{i}/{len(RW_QUESTIONS)}] {status}")
        print(f"  Pytanie:    {q}")
        print(f"  Oczekiwana: {expected}")
        print(f"  Odpowiedź:  {gen}\n")

    print(f">>> Wynik na raporcie rzeczywistym: {hits_rw}/{len(RW_QUESTIONS)} ({(hits_rw/len(RW_QUESTIONS))*100:.1f}%)")
    print("=" * 72)

if __name__ == "__main__":
    main()
