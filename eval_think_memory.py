import os
import sys
import torch

from smollm_scene_slots import SmolLMSceneMemory
from dataset_memory_recall import generate_dataset

WEIGHTS_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_think_memory.pt")

def evaluate_think_memory(num_tests: int = 15, base_seed: int = 777):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 65)
    print(f"  BENCHMARK: SCENE SLOTS + THINK TOKENS (SPOSÓB B) ({device.upper()})")
    print(f"  Scena (Tura 1) -> [4 Sloty (4.6 KB)]")
    print(f"  Tura 2: [4 Sloty] + [Pytanie] + [4 Tokeny Myślenia] -> Odpowiedź")
    print("=" * 65)

    memory = SmolLMSceneMemory(num_slots=4, num_think_tokens=4, device=device)
    if os.path.exists(WEIGHTS_PATH):
        print(f">>> Ładowanie wag adaptera z: {WEIGHTS_PATH}")
        memory.bridge.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("    Załadowano pomyślnie!\n")
    else:
        print(">>> Brak pliku wag — uruchamianie w trybie bazowym (przed treningiem)\n")

    test_data = generate_dataset(num_samples=num_tests, base_seed=base_seed)
    hits = 0
    total = len(test_data)

    for i, (passage, question, expected_answer) in enumerate(test_data, 1):
        # 1. Zwijanie sceny do 4 slotów
        scene_vectors = memory.encode_scene(passage)

        # 2. Odpowiedź z samych slotów + tokeny myślenia (zero tekstu z Tury 1)
        generated_answer = memory.answer_from_scene(scene_vectors, question, use_think=True)

        is_hit = expected_answer.lower() in generated_answer.lower()
        if is_hit:
            hits += 1

        status = "✓ TRAFIENIE" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/{total:02d}] {status}")
        print(f"  Scena:      {passage}")
        print(f"  Pytanie:    {question}")
        print(f"  Oczekiwana: {expected_answer}")
        print(f"  Odpowiedź:  {generated_answer}\n")

    acc = (hits / total) * 100
    print("=" * 65)
    print(f"  WYNIK KOŃCOWY: {hits}/{total} ({acc:.1f}% dokładności)")
    print("=" * 65)
    return hits, total, acc

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    evaluate_think_memory(num_tests=n)
