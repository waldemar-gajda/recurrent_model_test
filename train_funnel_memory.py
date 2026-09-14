import os
import sys
import time
import torch
import torch.nn as nn

from smollm_scene_slots import SmolLMSceneMemory
from dataset_memory_recall import generate_dataset

SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_funnel_memory.pt")

def train_funnel_memory():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 68)
    print(f"  TRENING LEJKA POZNAWCZEGO (COGNITIVE FUNNEL) NA SMOLLM2-135M ({device.upper()})")
    print("  Tura 1 (Wysoka rozdzielczość): K=8 Slotów Sceny (9.2 KB)")
    print("  Tura 2 (Zwężenie / Lejek):     P=4 Tokeny Myślenia -> Odpowiedź")
    print("=" * 68)

    memory = SmolLMSceneMemory(num_slots=8, num_think_tokens=4, device=device)
    trainable_params = sum(p.numel() for p in memory.bridge.parameters() if p.requires_grad)
    print(f">>> Trenowalne parametry (8 slotów + 4 tokeny myślenia): {trainable_params / 1e3:.1f}k ({trainable_params * 2 / 1024:.1f} KB)\n")

    train_data = generate_dataset(num_samples=250, base_seed=42)
    val_data = generate_dataset(num_samples=10, base_seed=999)
    print(f">>> Zbiór danych: {len(train_data)} próbek treningowych, {len(val_data)} walidacyjnych")

    # Baseline przed treningiem
    print("\n--- TEST PRZED TRENINGIEM (BASELINE) ---")
    val_sample = val_data[0]
    scene_v = memory.encode_scene(val_sample[0])
    gen_before = memory.answer_from_scene(scene_v, val_sample[1], use_think=True)
    print(f"Scena:      {val_sample[0]}")
    print(f"Pytanie:    {val_sample[1]}")
    print(f"Oczekiwana: {val_sample[2]}")
    print(f"Odpowiedź:  {gen_before}")
    is_hit_before = val_sample[2].lower() in gen_before.lower()
    print(f"Trafienie:  {'TAK' if is_hit_before else 'NIE'}\n")

    optimizer = torch.optim.AdamW(memory.bridge.parameters(), lr=1.5e-3, weight_decay=0.01)

    print("=" * 68)
    print("  ROZPOCZĘCIE TRENINGU (10 EPOK Z LEJKIEM 8 -> 4)")
    print("=" * 68)

    num_epochs = 10
    best_loss = 999.0

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        total_loss = 0.0

        for i, (passage, question, answer) in enumerate(train_data, 1):
            optimizer.zero_grad()
            
            # Krok 1: Szeroki zapis do 8 slotów
            scene_vectors = memory.encode_scene(passage)

            # Krok 2: Zwężenie przez 4 tokeny myślenia
            loss = memory.forward_qa(scene_vectors, question, answer, use_think=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.bridge.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0 or i == len(train_data):
                print(f"  [Epoka {epoch + 1:02d}/{num_epochs:02d}] Krok {i:03d}/{len(train_data)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_epoch_start
        print(f"\n>>> Średnia strata w epoce {epoch + 1}: {avg_loss:.4f} (Czas: {elapsed:.1f}s)")

        # Ewaluacja po epoce na próbce walidacyjnej
        scene_v = memory.encode_scene(val_sample[0])
        gen_after = memory.answer_from_scene(scene_v, val_sample[1], use_think=True)
        is_hit = val_sample[2].lower() in gen_after.lower()
        print(f"  Walidacja po epoce {epoch + 1}:")
        print(f"  Oczekiwana: {val_sample[2]}")
        print(f"  Odpowiedź:  {gen_after}")
        print(f"  Trafienie:  {'TAK! (Trafiony kod)' if is_hit else 'NIE'}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(memory.bridge.state_dict(), SAVE_PATH)

    print(f">>> Zapisano optymalne wagi do: {SAVE_PATH}")

if __name__ == "__main__":
    train_funnel_memory()
