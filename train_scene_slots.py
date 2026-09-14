import os
import sys
import time
import torch
import torch.nn as nn

from smollm_scene_slots import SmolLMSceneMemory
from dataset_memory_recall import generate_dataset

SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_scene_slots.pt")

def train_scene_memory():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 60)
    print(f"  TRENING DYNAMICZNYCH UCHWYTÓW SCENY NA SMOLLM2-135M ({device.upper()})")
    print("=" * 60)

    memory = SmolLMSceneMemory(num_slots=4, device=device)
    trainable_params = sum(p.numel() for p in memory.bridge.parameters() if p.requires_grad)
    print(f">>> Trenowalne parametry adaptera slotów: {trainable_params / 1e3:.1f}k ({trainable_params * 2 / 1024:.1f} KB)\n")

    train_data = generate_dataset(num_samples=150, base_seed=42)
    val_data = generate_dataset(num_samples=10, base_seed=999)
    print(f">>> Zbiór danych: {len(train_data)} próbek treningowych, {len(val_data)} walidacyjnych")

    # Test baseline przed treningiem
    print("\n--- TEST PRZED TRENINGIEM (BASELINE) ---")
    val_sample = val_data[0]
    scene_v = memory.encode_scene(val_sample[0])
    gen_before = memory.answer_from_scene(scene_v, val_sample[1])
    print(f"Scena:      {val_sample[0]}")
    print(f"Pytanie:    {val_sample[1]}")
    print(f"Oczekiwana: {val_sample[2]}")
    print(f"Odpowiedź:  {gen_before}")
    is_hit_before = val_sample[2].lower() in gen_before.lower()
    print(f"Trafienie:  {'TAK' if is_hit_before else 'NIE'}\n")

    optimizer = torch.optim.AdamW(memory.bridge.parameters(), lr=1e-3, weight_decay=0.01)

    print("=" * 60)
    print("  ROZPOCZĘCIE TRENINGU (4 EPOKI)")
    print("=" * 60)

    num_epochs = 4
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        total_loss = 0.0

        for i, (passage, question, answer) in enumerate(train_data, 1):
            optimizer.zero_grad()
            
            # Krok 1: Zwijanie sceny do 4 slotów
            scene_vectors = memory.encode_scene(passage)

            # Krok 2: Odpowiedź na pytanie z samych slotów
            loss = memory.forward_qa(scene_vectors, question, answer)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.bridge.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 30 == 0 or i == len(train_data):
                gate_val = torch.sigmoid(memory.bridge.gate).item()
                print(f"  [Epoka {epoch + 1}/{num_epochs}] Krok {i:03d}/{len(train_data)} | Loss: {loss.item():.4f} | Gate: {gate_val:.4f}")

        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_epoch_start
        print(f"\n>>> Średnia strata w epoce {epoch + 1}: {avg_loss:.4f} (Czas epoki: {elapsed:.1f}s)")

        # Ewaluacja po epoce
        scene_v = memory.encode_scene(val_sample[0])
        gen_after = memory.answer_from_scene(scene_v, val_sample[1])
        is_hit = val_sample[2].lower() in gen_after.lower()
        print(f"  Walidacja po epoce {epoch + 1}:")
        print(f"  Oczekiwana: {val_sample[2]}")
        print(f"  Odpowiedź:  {gen_after}")
        print(f"  Trafienie:  {'TAK!' if is_hit else 'NIE'}\n")

    print(f">>> Zapisywanie wag adaptera slotów do: {SAVE_PATH}")
    torch.save(memory.bridge.state_dict(), SAVE_PATH)
    print("Zapisano pomyślnie!")

if __name__ == "__main__":
    train_scene_memory()
