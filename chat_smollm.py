#!/usr/bin/env python3
import os
import glob
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
transformers.logging.set_verbosity_error()

from safetensors import safe_open
from transformers import AutoTokenizer
from llama_recurrent_model import LlamaRecurrentConfig, LlamaAllTokenRecurrentModel

CALIBRATED_CACHE = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm_recurrent_calibrated.pt")

def setup_smollm2_recurrent():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print("\n" + "="*60)
    print(f"  SMOLLM2-1.7B ALL-TOKEN RECURRENT MODEL NA {device.upper()}")
    print("="*60)

    model_dir = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots")
    snapshots = glob.glob(os.path.join(model_dir, "*"))
    if not snapshots:
        print("Błąd: Nie znaleziono pobranego modelu SmolLM2-1.7B.")
        sys.exit(1)
    snapshot_path = snapshots[0]

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, clean_up_tokenization_spaces=False)

    config = LlamaRecurrentConfig(
        vocab_size=49152,
        d_model=2048,
        intermediate_size=8192,
        num_heads=32,
        num_kv_heads=32,
        num_encoder_layers=12,
        num_decoder_layers=12,
        max_seq_len=4096,
        rope_theta=130000.0,
        rms_norm_eps=1e-5,
    )

    model = LlamaAllTokenRecurrentModel(config).to(dtype=dtype, device=device)

    if os.path.exists(CALIBRATED_CACHE):
        print(">>> Wczytuję wcześniej skalibrowany model z dysku...")
        state_dict = torch.load(CALIBRATED_CACHE, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(">>> Model gotowy do rozmowy natychmiast!\n")
        return model, tokenizer, device

    print(">>> 1. Wykonuję przeszczep wag z SmolLM2-1.7B (~3.4 GB)...")
    safetensors_files = sorted(glob.glob(os.path.join(snapshot_path, "*.safetensors")))

    # Inicjalizacja mostka rekurencyjnego Residual-Add
    with torch.no_grad():
        model.h_proj.weight.zero_()
        model.h0.zero_()

    for shard_file in safetensors_files:
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key).to(dtype=dtype, device=device)
                with torch.no_grad():
                    if key == "model.embed_tokens.weight":
                        model.embed_tokens.weight.copy_(tensor)
                        model.lm_head.weight.copy_(tensor)  # Tied embeddings
                    elif key == "lm_head.weight":
                        model.lm_head.weight.copy_(tensor)
                    elif key == "model.norm.weight":
                        model.decoder_norm.weight.copy_(tensor)
                    elif key.startswith("model.layers."):
                        parts = key.split(".")
                        layer_idx = int(parts[2])
                        sub_key = ".".join(parts[3:])

                        # Layers 0..11 -> Encoder
                        if layer_idx < 12:
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

                        # Layers 12..23 -> Decoder
                        else:
                            dec_idx = layer_idx - 12
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
                                dec.cross_attn.o_proj.weight.zero_()  # Zero-init cross attention
                            elif sub_key == "mlp.gate_proj.weight": dec.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": dec.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": dec.mlp.down_proj.weight.copy_(tensor)
                del tensor
        gc.collect()

    print(">>> 2. Przygotowuję szew adaptacyjny Residual-Add...")
    for p in model.parameters():
        p.requires_grad = False

    seam_params = []
    for p in model.h_proj.parameters():
        p.requires_grad = True
        seam_params.append(p)
    model.h0.requires_grad = True
    seam_params.append(model.h0)

    for layer in model.decoder_layers:
        p = layer.cross_attn.o_proj.weight
        p.requires_grad = True
        seam_params.append(p)

    trainable = sum(p.numel() for p in seam_params)
    print(f"    Parametry szwu do kalibracji: {trainable:,} ({trainable/1700000000*100:.2f}% modelu)")

    print(">>> 3. Uruchamiam kalibrację na różnorodnym korpusie zdań...")
    calibration_texts = [
        "In computer science, an algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation.",
        "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities.",
        "The French Revolution was a period of fundamental political and societal change in France that began with the Estates General of 1789 and ended in November 1799 with the formation of the French Consulate.",
        "Operating systems manage computer hardware and software resources and provide common services for computer programs. Examples include Linux, macOS, and Microsoft Windows.",
        "Machine learning focuses on the study of computer algorithms that improve automatically through experience and by the use of training data across multiple domains.",
        "The Atlantic Ocean is the second-largest of the world's five oceans, with an area of about 106,460,000 square kilometers. It covers approximately 20 percent of Earth's surface.",
        "A programming language is a system of notation for writing computer programs. Most programming languages are text-based formal languages, but they may also be graphical.",
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
        "The Renaissance was a fervent period of European cultural, artistic, political, and economic rebirth following the Middle Ages, beginning largely in fourteenth-century Italy.",
        "DNA is a polymer composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development and functioning of all organisms.",
        "Databases are organized collections of data stored and accessed electronically from a computer system. Where databases are more complex they are often developed using formal design methods.",
        "Linguistics is the scientific study of language. It involves the analysis of language form, language meaning, and language in cultural and historical context.",
        "Ecosystems are communities of living organisms in conjunction with the nonliving components of their environment, interacting as a system through nutrient cycles and energy flows.",
        "Philosophy explores fundamental questions about existence, knowledge, values, reason, mind, and language through critical questioning, rational argument, and systematic presentation.",
        "Robotics is an interdisciplinary branch of computer science and engineering involving the design, construction, operation, and use of robots to assist and replace humans in hazardous tasks."
    ]

    optimizer = torch.optim.AdamW(seam_params, lr=1e-4, weight_decay=0.01)
    model.train()
    tokens_batch = [tokenizer.encode(t, return_tensors="pt").to(device) for t in calibration_texts]

    # Kalibracja szwu
    for step in range(1, 11):
        total_loss = 0.0
        optimizer.zero_grad()
        for input_ids in tokens_batch:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, 49152), shift_labels.view(-1))
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(seam_params, 0.5)
        optimizer.step()

        if step % 2 == 0 or step == 1:
            avg_loss = total_loss / len(tokens_batch)
            print(f"    Krok {step:2d}/10 | Błąd szwu (Loss): {avg_loss:.4f}")

    print(">>> 4. Zapisuję zgeneralizowany model SmolLM2-1.7B na dysku...")
    model.eval()
    torch.save(model.state_dict(), CALIBRATED_CACHE)
    print(">>> Gotowe! Baza wag SmolLM2-1.7B Recurrent zoptymalizowana.\n")
    return model, tokenizer, device

def generate_stream(
    model,
    tokenizer,
    prompt,
    device,
    current_h=None,
    target_tokens=100,
    min_tokens=25,
    max_tokens=300,
    temperature=0.6,
    top_k=40,
):
    """
    Szybka generacja rekurencyjna O(1) z ciągłą pamięcią stanu H i inteligentnym zatrzymywaniem.
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
            # Łagodna kara za powtórzenia tokenów z promptu
            for t_id in seen_tokens:
                if last_logits[0, t_id] > 0:
                    last_logits[0, t_id] /= 1.15
                else:
                    last_logits[0, t_id] *= 1.15

            if temperature > 0:
                scaled_logits = last_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 0, 2]:
                break

            generated_tokens.append(tok_id)
            seen_tokens.add(tok_id)

            # Bezpieczne dekodowanie strumieniowe (zapobiega rozbijaniu znaków UTF-8)
            full_text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=False)
            new_text = full_text[len(printed_text):]
            if new_text:
                print(new_text, end="", flush=True)
                printed_text = full_text

            num_gen = step_i + 1

            # INTELIGENTNE ZATRZYMANIE:
            # 1. Koniec akapitu (podwójny enter) po wygenerowaniu minimalnej liczby tokenów
            if num_gen >= min_tokens and full_text.endswith("\n\n"):
                break

            # 2. Zamknięcie bloku kodu (```) po wygenerowaniu kodu
            if num_gen >= min_tokens and "```" in full_text and full_text.rstrip().endswith("```"):
                break

            # 3. Zakończenie pełnego zdania po osiągnięciu docelowej długości (target_tokens)
            if num_gen >= target_tokens:
                stripped = full_text.rstrip()
                if stripped.endswith(("!", "?", '!"', '?"')):
                    break
                if stripped.endswith((".", '."')):
                    words = stripped.split()
                    last_w = words[-1].lower().strip("()[]{}\"'") if words else ""
                    # Nie zatrzymuj się na skrótach (np. e.g., i.e., vs., dr., pojedynczych literach)
                    if last_w not in ["e.g", "i.e", "vs", "etc", "mr", "dr", "prof", "e", "al", "fig", "note"] and len(last_w) > 1:
                        break

            # Szybki krok O(1) z buforowaniem KV
            last_logits, enc_cache, dec_cache, h = model.step(
                next_tok, pos, enc_cache, dec_cache, h
            )
            pos += 1

    print()
    return h

def main():
    model, tokenizer, device = setup_smollm2_recurrent()

    print("=" * 60)
    print("  SMOLLM2-1.7B (ALL-TOKEN RECURRENCE) — LIVE CHAT")
    print("  Szybki model z ciągłą pamięcią stanu H i buforem KV Cache.")
    print("  Wpisz tekst, 'reset' aby zresetować pamięć, lub 'q' aby wyjść.")
    print("=" * 60)

    # Krótki test automatyczny
    print("\n--- Test kontrolny generacji ---")
    test_prompts = [
        "The capital of France is",
        "Artificial intelligence is",
    ]
    for tp in test_prompts:
        print(f"\nPrompt: '{tp}'")
        print("Model:  ", end="", flush=True)
        generate_stream(model, tokenizer, tp, device, target_tokens=50, min_tokens=10, temperature=0.3)
    print("--- Koniec testu kontrolnego ---\n")

    current_h = None
    turn_idx = 1

    while True:
        try:
            prompt = input(f"\n[Tura {turn_idx}] Ty > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["q", "quit", "exit"]:
                print("\nDo widzenia!")
                break
            if prompt.lower() in ["reset", "/reset", "nowy"]:
                current_h = None
                turn_idx = 1
                print("\n[Pamięć stanu H zresetowana do zera — nowy wątek]")
                continue

            print(f"\n[Tura {turn_idx}] SmolLM2 Recurrent > ", end="", flush=True)
            current_h = generate_stream(model, tokenizer, prompt, device, current_h=current_h)
            turn_idx += 1

        except KeyboardInterrupt:
            print("\nZakończono.")
            break

if __name__ == "__main__":
    main()
