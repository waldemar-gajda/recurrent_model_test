import os
import sys
import torch

from smollm_scene_slots import SmolLMSceneMemory
from dataset_memory_recall import generate_dataset

WEIGHTS_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_funnel_memory.pt")

def evaluate_funnel_memory(num_tests: int = 15, base_seed: int = 777):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 68)
    print(f"  BENCHMARK: LEJEK POZNAWCZY (K=8 SLOTÓW -> P=4 TOKENY MYŚLENIA) ({device.upper()})")
    print(f"  Tura 1: Scena -> [8 Slotów (9.2 KB)]")
    print(f"  Tura 2: [8 Slotów] + [Pytanie] + [4 Tokeny Myślenia] -> Odpowiedź")
    print("=" * 68)

    memory = SmolLMSceneMemory(num_slots=8, num_think_tokens=4, device=device)
    if os.path.exists(WEIGHTS_PATH):
        print(f">>> Ładowanie wag adaptera z: {WEIGHTS_PATH}")
        memory.bridge.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("    Załadowano pomyślnie!\n")
    else:
        print(">>> Brak pliku wag — uruchamianie w trybie bazowym (przed treningiem)\n")

    test_data = generate_dataset(num_samples=num_tests, base_seed=base_seed)
    hits = 0
    total = len(test_data)

    name_hits, name_tot = 0, 0
    code_hits, code_tot = 0, 0
    other_hits, other_tot = 0, 0

    for i, (passage, question, expected_answer) in enumerate(test_data, 1):
        # 1. Zwijanie sceny do 8 slotów
        scene_vectors = memory.encode_scene(passage)

        # 2. Odpowiedź ze zwężeniem przez 4 tokeny myślenia
        generated_answer = memory.answer_from_scene(scene_vectors, question, use_think=True)

        is_hit = expected_answer.lower() in generated_answer.lower()
        if is_hit:
            hits += 1

        is_code = expected_answer.replace('#', '').strip().isdigit()
        is_name = any(k in expected_answer for k in ['Jan', 'Marcus', 'Elena', 'Adam', 'Sarah', 'Viktor', 'Piotr', 'Operative', 'Agent'])
        if is_name:
            name_tot += 1
            if is_hit: name_hits += 1
        elif is_code:
            code_tot += 1
            if is_hit: code_hits += 1
        else:
            other_tot += 1
            if is_hit: other_hits += 1

        status = "✓ TRAFIENIE" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/{total:02d}] {status}")
        print(f"  Scena:      {passage}")
        print(f"  Pytanie:    {question}")
        print(f"  Oczekiwana: {expected_answer}")
        print(f"  Odpowiedź:  {generated_answer}\n")

    acc = (hits / total) * 100
    print("=" * 68)
    print(f"  WYNIK KOŃCOWY: {hits}/{total} ({acc:.1f}% dokładności)")
    if code_tot > 0:
        print(f"  Kody liczbowe/PIN: {code_hits}/{code_tot}")
    if name_tot > 0:
        print(f"  Nazwiska/Agenci:   {name_hits}/{name_tot}")
    if other_tot > 0:
        print(f"  Inne (komory/itp): {other_hits}/{other_tot}")
    print("=" * 68)
    return hits, total, acc

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    evaluate_funnel_memory(num_tests=n)
