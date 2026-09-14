import os
import sys
import torch

from evolutionary_memory_model import SmolLMEvolutionaryMemory
from dataset_memory_recall import generate_multiturn_dataset

WEIGHTS_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_evolutionary_memory.pt")

def evaluate_evolutionary_memory(num_tests: int = 15, base_seed: int = 888):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 72)
    print(f"  BENCHMARK: ZINTEGROWANA PAMIĘĆ EWOLUCYJNA (SMOLLM2-135M, {device.upper()})")
    print("  Tura 1 (Percepcja):      Scena -> S_0 (8 Slotów, 9.2 KB)")
    print("  Tura 2 (Mikro-ewolucja): S_0 + Q1 + [Draft 4] + [Verify 4] -> Odpowiedź 1")
    print("  Konsolidacja:            S_0 + Q1 + A1 -> Ewolucja pamięci do S_1 (9.2 KB)")
    print("  Tura 3 (Makro-ewolucja): S_1 + Q2 + [Draft 4] + [Verify 4] -> Odpowiedź 2")
    print("=" * 72)

    memory = SmolLMEvolutionaryMemory(num_slots=8, num_draft=4, num_verify=4, device=device)
    if os.path.exists(WEIGHTS_PATH):
        print(f">>> Ładowanie wag adaptera z: {WEIGHTS_PATH}")
        memory.bridge.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("    Załadowano pomyślnie!\n")
    else:
        print(">>> Brak pliku wag — uruchamianie w trybie bazowym\n")

    test_data = generate_multiturn_dataset(num_samples=num_tests, base_seed=base_seed)
    
    hits_t2 = 0
    hits_t3 = 0
    hits_both = 0
    total = len(test_data)

    for i, (passage, q1, a1, q2, a2) in enumerate(test_data, 1):
        # 1. Tura 1: Kodowanie dokumentu do S_0
        s_0 = memory.encode_scene(passage)

        # 2. Tura 2: Odpowiedź na Q1 z mikro-ewolucją
        gen1 = memory.answer_turn(s_0, q1)
        hit1 = a1.lower() in gen1.lower()
        if hit1: hits_t2 += 1

        # 3. Konsolidacja pamięci S_0 -> S_1
        s_1 = memory.consolidate_memory(s_0, q1, gen1)

        # 4. Tura 3: Odpowiedź na Q2 z ewoluującej pamięci S_1 (zero tekstu z Tury 1 ani Tury 2!)
        gen2 = memory.answer_turn(s_1, q2)
        hit2 = a2.lower() in gen2.lower()
        if hit2: hits_t3 += 1

        if hit1 and hit2: hits_both += 1

        print(f"[{i:02d}/{total:02d}] Scena: {passage}")
        print(f"  Tura 2 - Q: {q1}")
        print(f"           Oczekiwana: {a1} | Wygenerowana: {gen1} | {'✓ TRAF' if hit1 else '✗ PUDŁO'}")
        print(f"  Tura 3 - Q: {q2} (ze stanu S_1)")
        print(f"           Oczekiwana: {a2} | Wygenerowana: {gen2} | {'✓ TRAF' if hit2 else '✗ PUDŁO'}")
        print()

    acc_t2 = (hits_t2 / total) * 100
    acc_t3 = (hits_t3 / total) * 100
    acc_both = (hits_both / total) * 100

    print("=" * 72)
    print("  PODSUMOWANIE BENCHMARKU EWOLUCYJNEGO:")
    print(f"  Tura 2 (Mikro-ewolucja Draft->Verify): {hits_t2}/{total} ({acc_t2:.1f}%)")
    print(f"  Tura 3 (Makro-ewolucja ze stanu S_1):   {hits_t3}/{total} ({acc_t3:.1f}%)")
    print(f"  Obie tury trafione bezbłędnie:          {hits_both}/{total} ({acc_both:.1f}%)")
    print("=" * 72)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    evaluate_evolutionary_memory(num_tests=n)
