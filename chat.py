#!/usr/bin/env python3
import os
import glob
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
import transformers
transformers.logging.set_verbosity_error()

from safetensors import safe_open
from transformers import AutoTokenizer
from llama_recurrent_model import LlamaRecurrentConfig, LlamaAllTokenRecurrentModel

SEAM_CACHE = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama3_recurrent_seam.pt")

def load_recurrent_llama():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"\n========================================================")
    print(f"  Ładowanie modelu Llama-3-8B Recurrent na {device.upper()}...")
    print(f"========================================================")

    model_dir = os.path.expanduser("~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots")
    snapshots = glob.glob(os.path.join(model_dir, "*"))
    if not snapshots:
        print("Błąd: Nie znaleziono pobranego modelu Llama-3-8B.")
        sys.exit(1)
    snapshot_path = snapshots[0]

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, clean_up_tokenization_spaces=False)
    config = LlamaRecurrentConfig(
        vocab_size=128256,
        d_model=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        num_encoder_layers=16,
        num_decoder_layers=16,
        max_seq_len=8192,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )

    print(">>> Alokacja bezpośrednio w pamięci MPS (~16 GB)...")
    torch.set_default_dtype(dtype)
    with torch.device(device):
        model = LlamaAllTokenRecurrentModel(config)
    torch.set_default_dtype(torch.float32)

    with torch.no_grad():
        model.h_proj.weight.zero_()
        model.h0.zero_()

    safetensors_files = sorted(glob.glob(os.path.join(snapshot_path, "*.safetensors")))
    print(f">>> Błyskawiczny transfer 8 miliardów parametrów z {len(safetensors_files)} plików...")

    with torch.no_grad():
        for shard_idx, shard_file in enumerate(safetensors_files, 1):
            print(f"    [Shard {shard_idx}/4] {os.path.basename(shard_file)}...", flush=True)
            with safe_open(shard_file, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if key == "model.embed_tokens.weight":
                        model.embed_tokens.weight.copy_(tensor)
                    elif key == "lm_head.weight":
                        model.lm_head.weight.copy_(tensor)
                    elif key == "model.norm.weight":
                        model.decoder_norm.weight.copy_(tensor)
                    elif key.startswith("model.layers."):
                        parts = key.split(".")
                        layer_idx = int(parts[2])
                        sub_key = ".".join(parts[3:])

                        if layer_idx < 16:
                            enc = model.encoder_layers[layer_idx]
                            if sub_key == "input_layernorm.weight": enc.input_layernorm.weight.copy_(tensor)
                            elif sub_key == "post_attention_layernorm.weight": enc.post_attention_layernorm.weight.copy_(tensor)
                            elif sub_key == "self_attn.q_proj.weight": enc.self_attn.q_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.k_proj.weight": enc.self_attn.k_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.v_proj.weight": enc.self_attn.v_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.o_proj.weight": enc.self_attn.o_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.gate_proj.weight": enc.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": enc.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": enc.mlp.down_proj.weight.copy_(tensor)
                        else:
                            dec_idx = layer_idx - 16
                            dec = model.decoder_layers[dec_idx]
                            if sub_key == "input_layernorm.weight": dec.input_layernorm.weight.copy_(tensor)
                            elif sub_key == "post_attention_layernorm.weight": dec.post_attention_layernorm.weight.copy_(tensor)
                            elif sub_key == "self_attn.q_proj.weight":
                                dec.self_attn.q_proj.weight.copy_(tensor)
                                dec.cross_attn.q_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.k_proj.weight": dec.self_attn.k_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.v_proj.weight": dec.self_attn.v_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.o_proj.weight":
                                dec.self_attn.o_proj.weight.copy_(tensor)
                                dec.cross_attn.o_proj.weight.zero_()
                            elif sub_key == "mlp.gate_proj.weight": dec.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": dec.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": dec.mlp.down_proj.weight.copy_(tensor)
                    del tensor

    if os.path.exists(SEAM_CACHE):
        print(f"\n>>> Wczytywanie skalibrowanego szwu adaptacyjnego z {os.path.basename(SEAM_CACHE)}...")
        seam_dict = torch.load(SEAM_CACHE, map_location=device)
        model.h_proj.weight.data.copy_(seam_dict["h_proj"])
        model.h0.data.copy_(seam_dict["h0"])
        for i, w in enumerate(seam_dict["cross_attn_o_proj"]):
            model.decoder_layers[i].cross_attn.o_proj.weight.data.copy_(w)

    print(f"\n>>> Model Llama-3-8B załadowany i gotowy do szybkiej rozmowy! [MPS]\n")
    model.eval()
    return model, tokenizer, device

def generate_stream(
    model,
    tokenizer,
    prompt,
    device,
    current_h=None,
    target_tokens=120,
    min_tokens=25,
    max_tokens=350,
    temperature=0.6,
    top_k=40,
):
    """
    Szybka generacja rekurencyjna O(1) dla Llama-3-8B z ciągłą pamięcią stanu H.
    Stan H (4096 liczb, 8 KB) przechodzi z tury na turę, pamiętając kontekst
    całej rozmowy bez zużywania dodatkowej pamięci RAM.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    B, T = input_ids.shape
    seen_tokens = set(input_ids[0].tolist())

    with torch.no_grad():
        logits, enc_cache, dec_cache, h = model.forward_with_cache(input_ids, initial_h=current_h)
        last_logits = logits[:, -1, :].clone()
        pos = T

        generated_tokens = []
        printed_text = ""

        for step_i in range(max_tokens):
            for t_id in seen_tokens:
                if last_logits[0, t_id] > 0:
                    last_logits[0, t_id] /= 1.15
                else:
                    last_logits[0, t_id] *= 1.15

            if temperature > 0:
                scaled_logits = last_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 128001, 128009]:
                break

            generated_tokens.append(tok_id)
            seen_tokens.add(tok_id)

            # Bezpieczne dekodowanie strumieniowe (zapobiega rozbijaniu znaków UTF-8)
            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Sprawdzamy znaczniki końca tury dialogu przed wydrukowaniem
            stop_turn_markers = ["\nCzłowiek", "\nUżytk", "\nUser", "\nHuman", "\nC:", "\nU:", "\nP:"]
            if any(marker in full_text for marker in stop_turn_markers):
                break

            # Bezpieczne dekodowanie strumieniowe (zapobiega rozbijaniu znaków UTF-8)
            new_text = full_text[len(printed_text):]
            if new_text:
                print(new_text, end="", flush=True)
                printed_text = full_text

            num_gen = step_i + 1

            # 2. Koniec akapitu po wygenerowaniu minimalnej liczby tokenów
            if num_gen >= min_tokens and full_text.endswith("\n\n"):
                break

            # 3. Zamknięcie bloku kodu (```)
            if num_gen >= min_tokens and "```" in full_text and full_text.rstrip().endswith("```"):
                break

            # 4. Zakończenie pełnego zdania po osiągnięciu target_tokens
            if num_gen >= target_tokens:
                stripped = full_text.rstrip()
                if stripped.endswith(("!", "?", '!"', '?"')):
                    break
                if stripped.endswith((".", '."')):
                    words = stripped.split()
                    last_w = words[-1].lower().strip("()[]{}\"'") if words else ""
                    if last_w not in ["e.g", "i.e", "vs", "etc", "mr", "dr", "prof", "e", "al", "fig", "note", "np", "tzw", "tzn"] and len(last_w) > 1:
                        break

            # Błyskawiczny krok O(1) z buforem KV Cache
            last_logits, enc_cache, dec_cache, h = model.step(
                next_tok, pos, enc_cache, dec_cache, h
            )
            pos += 1

    print()
    return h

def main():
    model, tokenizer, device = load_recurrent_llama()

    print("=" * 60)
    print("  LLAMA-3-8B RECURRENT — TRYB ASYSTENTA NA APPLE SILICON (MPS)")
    print("  Architektura All-Token Recurrence z ciągłą pamięcią stanu H.")
    print("  Zadawaj pytania naturalnie po polsku. Pamięć RAM jest stała.")
    print("  Wpisz 'reset' aby zacząć nowy wątek, lub 'q' aby wyjść.")
    print("=" * 60)

    current_h = None
    turn_idx = 1

    while True:
        try:
            user_input = input(f"\n[Tura {turn_idx}] Ty > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\nDo widzenia!")
                break
            if user_input.lower() in ["reset", "/reset", "nowy"]:
                current_h = None
                turn_idx = 1
                print("\n[Pamięć stanu H zresetowana do zera — nowy wątek]")
                continue

            # Formatowanie dialogowe ze wzorcem głębokich, wyczerpujących odpowiedzi
            if turn_idx == 1:
                prompt = f"""Oto rozmowa między dociekliwym Człowiekiem a wybitnym, filozoficznym Asystentem AI. Asystent udziela głębokich, wyczerpujących i wnikliwych odpowiedzi w języku polskim.

Człowiek: Dlaczego ludzie boją się samotności?
Asystent: Samotność konfrontuje człowieka z jego własną skończonością i brakiem zewnętrznych punktów odniesienia. Jako istoty ewolucyjnie społeczne wykształciliśmy lęk przed izolacją, gdyż w naturze odrzucenie przez grupę oznaczało śmierć. Na poziomie egzystencjalnym cisza samotności zmusza nas do spojrzenia w głąb siebie, gdzie często napotykamy pytania o sens, tożsamość i przemijanie, przed którymi na co dzień uciekamy w hałas codzienności.

Człowiek: {user_input}
Asystent:"""
            else:
                prompt = f"\nCzłowiek: {user_input}\nAsystent:"

            print(f"\n[Tura {turn_idx}] Asystent AI > ", end="", flush=True)
            current_h = generate_stream(model, tokenizer, prompt, device, current_h=current_h, target_tokens=140, min_tokens=30)
            turn_idx += 1

        except KeyboardInterrupt:
            print("\nZakończono.")
            break

if __name__ == "__main__":
    main()
