import os
import sys
import time
import torch
import torch.nn as nn

from dual_stream_memory import SmolLMDualStreamMemory
from dataset_memory_recall import generate_dataset

SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_dual_stream.pt")

def run_dual_stream_experiment(num_epochs: int = 14):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 70)
    print(f"  EKSPERYMENT: DUAL-STREAM MEMORY ({num_epochs} EPOK Z SCHEDULEREM)")
    print("  Tor 1: 8 Slotów Sceny (9.2 KB) -> Kontekst, fakty, pojęcia")
    print("  Tor 2: Rejestr Symboli (32 B)  -> Dokładne liczby, kody, literały")
    print(f"  Razem w RAM: 9.23 KB (ściśle O(1)) | Sprzęt: {device.upper()}")
    print("=" * 70)

    memory = SmolLMDualStreamMemory(num_slots=8, num_think=4, max_sym_len=16, device=device)
    train_data = generate_dataset(num_samples=200, base_seed=42)
    test_data = generate_dataset(num_samples=20, base_seed=1234)

    optimizer = torch.optim.AdamW(memory.bridge.parameters(), lr=1.5e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    best_loss = 999.0
    print(f"\n>>> Start treningu ({num_epochs} epok z wygaszaniem LR do 1e-4)...")

    for epoch in range(num_epochs):
        t_start = time.time()
        total_loss = 0.0

        for p, q, a in train_data:
            optimizer.zero_grad()
            scene_v, sym_ids = memory.encode_scene_and_symbols(p)
            loss = memory.forward_qa(scene_v, sym_ids, q, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.bridge.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_start
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Epoka {epoch + 1:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f} | LR: {cur_lr:.5f} | ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(memory.bridge.state_dict(), SAVE_PATH)

    print(f"\n>>> Zapisano optymalne wagi do {SAVE_PATH}. Rozpoczynam ewaluację na 20 nowych scenach...\n")

    # Ewaluacja
    memory.bridge.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    hits = 0
    total = len(test_data)

    hits_pin, tot_pin = 0, 0
    hits_key, tot_key = 0, 0
    hits_text, tot_text = 0, 0

    for i, (p, q, expected) in enumerate(test_data, 1):
        scene_v, sym_ids = memory.encode_scene_and_symbols(p)
        gen = memory.answer(scene_v, sym_ids, q)

        is_hit = expected.lower() in gen.lower() or gen.lower() in expected.lower()
        if is_hit: hits += 1

        is_pure_pin = expected.isdigit() or (expected.startswith('#') and expected[1:].isdigit())
        is_key = '-' in expected and any(c.isdigit() for c in expected)

        if is_pure_pin:
            tot_pin += 1
            if is_hit: hits_pin += 1
            cat_name = "PIN/NUMER"
        elif is_key:
            tot_key += 1
            if is_hit: hits_key += 1
            cat_name = "KLUCZ/ID"
        else:
            tot_text += 1
            if is_hit: hits_text += 1
            cat_name = "TEKST/POMIAR"

        status = "✓ TRAF" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/{total:02d}] {status} | Kat: {cat_name}")
        print(f"  Scena:      {p}")
        print(f"  Pytanie:    {q}")
        print(f"  Oczekiwana: {expected}")
        print(f"  Odpowiedź:  {gen}\n")

    acc = (hits / total) * 100
    print("=" * 70)
    print(f"  WYNIKI KOŃCOWE DUAL-STREAM MEMORY ({num_epochs} EPOK):")
    print(f"  Kody PIN (czyste cyfry):   {hits_pin}/{tot_pin} ({(hits_pin/tot_pin*100) if tot_pin else 0:.1f}%)")
    print(f"  Klucze / ID (np. ALPHA):   {hits_key}/{tot_key} ({(hits_key/tot_key*100) if tot_key else 0:.1f}%)")
    print(f"  Tekst / Pomiary / Osoby:   {hits_text}/{tot_text} ({(hits_text/tot_text*100) if tot_text else 0:.1f}%)")
    print(f"  ŁĄCZNA DOKŁADNOŚĆ:         {hits}/{total} ({acc:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    run_dual_stream_experiment(num_epochs=ep)
