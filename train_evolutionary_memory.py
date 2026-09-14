import os
import sys
import time
import torch
import torch.nn as nn

from evolutionary_memory_model import SmolLMEvolutionaryMemory
from dataset_memory_recall import generate_multiturn_dataset

SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_evolutionary_memory.pt")

def train_evolutionary_memory():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print(f"  TRENING ZINTEGROWANEJ PAMIĘCI EWOLUCYJNEJ (SMOLLM2-135M, {device.upper()})")
    print("  Mikro-ewolucja (wewnątrz tury): 2-krokowa deliberacja (Draft -> Verify)")
    print("  Makro-ewolucja (między turami): Ewolucja stanu pamięci S_0 -> S_1 (9.2 KB)")
    print("=" * 70)

    memory = SmolLMEvolutionaryMemory(num_slots=8, num_draft=4, num_verify=4, device=device)
    trainable_params = sum(p.numel() for p in memory.bridge.parameters() if p.requires_grad)
    print(f">>> Trenowalne parametry: {trainable_params / 1e3:.1f}k ({trainable_params * 2 / 1024:.1f} KB)\n")

    train_data = generate_multiturn_dataset(num_samples=200, base_seed=42)
    val_data = generate_multiturn_dataset(num_samples=10, base_seed=999)
    print(f">>> Zbiór danych: {len(train_data)} próbek 3-turowych")

    # Baseline przed treningiem
    print("\n--- TEST PRZED TRENINGIEM (BASELINE) ---")
    val_sample = val_data[0]
    p_val, q1_val, a1_val, q2_val, a2_val = val_sample
    s0_val = memory.encode_scene(p_val)
    gen1_before = memory.answer_turn(s0_val, q1_val)
    s1_val = memory.consolidate_memory(s0_val, q1_val, a1_val)
    gen2_before = memory.answer_turn(s1_val, q2_val)

    print(f"Scena:            {p_val}")
    print(f"Tura 2 - Pytanie: {q1_val} | Oczekiwana: {a1_val} | Odpowiedź: {gen1_before}")
    print(f"Tura 3 - Pytanie: {q2_val} | Oczekiwana: {a2_val} | Odpowiedź: {gen2_before}\n")

    optimizer = torch.optim.AdamW(memory.bridge.parameters(), lr=1.5e-3, weight_decay=0.01)

    num_epochs = 10
    best_loss = 999.0

    print("=" * 70)
    print("  ROZPOCZĘCIE TRENINGU (10 EPOK)")
    print("=" * 70)

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        total_loss = 0.0

        for i, (passage, q1, a1, q2, a2) in enumerate(train_data, 1):
            optimizer.zero_grad()

            # Tura 1: Kodowanie dokumentu do S_0 (8 slotów)
            s_0 = memory.encode_scene(passage)

            # Tura 2: Odpowiedź na Q1 (Mikro-ewolucja Draft -> Verify)
            loss1 = memory.forward_turn(s_0, q1, a1)

            # Makro-ewolucja: Konsolidacja pamięci S_0 -> S_1
            s_1 = memory.consolidate_memory(s_0, q1, a1)

            # Tura 3: Odpowiedź na Q2 z ewoluującej pamięci S_1
            loss2 = memory.forward_turn(s_1, q2, a2)

            batch_loss = loss1 + loss2
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(memory.bridge.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()

            if i % 50 == 0 or i == len(train_data):
                print(f"  [Epoka {epoch + 1:02d}/{num_epochs:02d}] Krok {i:03d}/{len(train_data)} | Loss T2: {loss1.item():.3f} | Loss T3: {loss2.item():.3f} | Total: {batch_loss.item():.3f}")

        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_epoch_start
        print(f"\n>>> Średnia strata w epoce {epoch + 1}: {avg_loss:.4f} (Czas: {elapsed:.1f}s)")

        # Walidacja po epoce
        s0_v = memory.encode_scene(p_val)
        ans1_v = memory.answer_turn(s0_v, q1_val)
        s1_v = memory.consolidate_memory(s0_v, q1_val, a1_val)
        ans2_v = memory.answer_turn(s1_v, q2_val)

        hit1 = a1_val.lower() in ans1_v.lower()
        hit2 = a2_val.lower() in ans2_v.lower()

        print(f"  [Walidacja Epoka {epoch + 1}]:")
        print(f"  Tura 2: Oczekiwana: {a1_val} | Wygenerowana: {ans1_v} | Trafienie: {'TAK!' if hit1 else 'NIE'}")
        print(f"  Tura 3: Oczekiwana: {a2_val} | Wygenerowana: {ans2_v} | Trafienie: {'TAK!' if hit2 else 'NIE'}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(memory.bridge.state_dict(), SAVE_PATH)

    print(f">>> Zapisano optymalne wagi do: {SAVE_PATH}")

if __name__ == "__main__":
    train_evolutionary_memory()
