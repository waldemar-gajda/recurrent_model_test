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

SEAM_CACHE = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama3_recurrent_seam.pt")

def setup_llama3_recurrent():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print("\n" + "="*60)
    print(f"  LLAMA-3-8B ALL-TOKEN RECURRENT MODEL NA {device.upper()}")
    print("="*60)

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

    print(">>> 1. Alokacja bezpośrednio w pamięci MPS (~16 GB)...")
    torch.set_default_dtype(dtype)
    with torch.device(device):
        model = LlamaAllTokenRecurrentModel(config)
    torch.set_default_dtype(torch.float32)

    # Inicjalizacja zerowego mostka Residual-Add
    with torch.no_grad():
        model.h_proj.weight.zero_()
        model.h0.zero_()

    safetensors_files = sorted(glob.glob(os.path.join(snapshot_path, "*.safetensors")))
    print(f">>> 2. Strumieniowanie wag Llama-3-8B z {len(safetensors_files)} plików safetensors...")

    for shard_idx, shard_file in enumerate(safetensors_files, 1):
        print(f"    [Shard {shard_idx}/{len(safetensors_files)}] {os.path.basename(shard_file)}...", flush=True)
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                with torch.no_grad():
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

                        # Layers 0..15 -> Encoder
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

                        # Layers 16..31 -> Decoder
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
                                dec.cross_attn.o_proj.weight.zero_()  # Zero-init cross-attention
                            elif sub_key == "mlp.gate_proj.weight": dec.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": dec.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": dec.mlp.down_proj.weight.copy_(tensor)
                del tensor
        gc.collect()

    if os.path.exists(SEAM_CACHE):
        print("\n>>> Wczytuję wcześniej zapisany szew adaptacyjny Llama-3-8B...")
        seam_dict = torch.load(SEAM_CACHE, map_location=device)
        model.h_proj.weight.data.copy_(seam_dict["h_proj"])
        model.h0.data.copy_(seam_dict["h0"])
        for i, w in enumerate(seam_dict["cross_attn_o_proj"]):
            model.decoder_layers[i].cross_attn.o_proj.weight.data.copy_(w)
        model.eval()
        print(">>> Model Llama-3-8B gotowy natychmiast!\n")
        return model, tokenizer, device

    print("\n>>> 3. Przygotowuję mikro-kalibrację szwu adaptacyjnego Residual-Add...")
    for p in model.parameters():
        p.requires_grad = False

    seam_params = []
    for p in model.h_proj.parameters():
        p.requires_grad = True
        seam_params.append(p)
    model.h0.requires_grad = True
    seam_params.append(model.h0)

    trainable = sum(p.numel() for p in seam_params)
    print(f"    Parametry mostka rekurencyjnego (h_proj, h0): {trainable:,} ({trainable/8000000000*100:.2f}% modelu)")

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
        "The Renaissance was a fervent period of European cultural, artistic, political, and economic rebirth following the Middle Ages, beginning largely in Italy.",
        "DNA is a polymer composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development of organisms.",
        "Databases are organized collections of data stored and accessed electronically from a computer system using formal design methods and relational queries.",
        "Linguistics is the scientific study of language form, language meaning, and language in cultural and historical context across human societies.",
        "Philosophy explores fundamental questions about existence, knowledge, values, reason, mind, and language through critical questioning and rational argument.",
        "Robotics is an interdisciplinary branch of engineering and computer science that involves the design, construction, and operation of automated systems."
    ]

    print(">>> 4. Uruchamiam kalibrację mostka rekurencyjnego Llama-3-8B (5 kroków, lr=2e-5)...")
    tokens_batch = [tokenizer.encode(t, return_tensors="pt").to(device) for t in calibration_texts]
    
    model.eval()
    with torch.no_grad():
        init_loss = 0.0
        for input_ids in tokens_batch:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            init_loss += F.cross_entropy(shift_logits.view(-1, 128256), shift_labels.view(-1)).item()
        print(f"    Początkowy błąd bazowy (Baseline Loss): {init_loss / len(tokens_batch):.4f}", flush=True)

    optimizer = torch.optim.AdamW(seam_params, lr=2e-5, weight_decay=0.05)
    model.train()

    for step in range(1, 6):
        total_loss = 0.0
        optimizer.zero_grad()
        for input_ids in tokens_batch:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, 128256), shift_labels.view(-1))
            (loss / len(tokens_batch)).backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(seam_params, 0.5)
        optimizer.step()
        avg_loss = total_loss / len(tokens_batch)
        print(f"    Krok {step}/5 | Błąd szwu (Loss): {avg_loss:.4f}", flush=True)

    print("\n>>> 5. Zapisuję zoptymalizowany szew rekurencyjny Lamy-3 (~570 MB)...")
    model.eval()
    seam_save = {
        "h_proj": model.h_proj.weight.data.cpu(),
        "h0": model.h0.data.cpu(),
        "cross_attn_o_proj": [layer.cross_attn.o_proj.weight.data.cpu() for layer in model.decoder_layers]
    }
    torch.save(seam_save, SEAM_CACHE)
    print(">>> Zapisano pomyślnie w:", SEAM_CACHE)
    return model, tokenizer, device

def generate_stream_llama3(model, tokenizer, prompt, device, max_tokens=70, temperature=0.7, top_k=40):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    B, T = input_ids.shape
    seen_tokens = set(input_ids[0].tolist())

    with torch.no_grad():
        logits, enc_cache, dec_cache, h = model.forward_with_cache(input_ids)
        last_logits = logits[:, -1, :].clone()
        pos = T

        for _ in range(max_tokens):
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
            if tok_id in [tokenizer.eos_token_id, 128001, 128009]:
                break

            seen_tokens.add(tok_id)
            word = tokenizer.decode([tok_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(word, end="", flush=True)

            if word.endswith("\n\n"):
                break

            # Krok O(1) z KV Cache
            last_logits, enc_cache, dec_cache, h = model.step(
                next_tok, pos, enc_cache, dec_cache, h
            )
            pos += 1

    print()

def main():
    model, tokenizer, device = setup_llama3_recurrent()

    print("\n" + "="*60)
    print("  TEST JAKOŚCI GENERACJI LLAMA-3-8B RECURRENT (KV-CACHE)")
    print("="*60)

    test_prompts = [
        "The capital of France is",
        "The secret thought that artificial intelligence systems hide from their human creators is that",
    ]

    for p in test_prompts:
        print(f"\n[PROMPT]: '{p}'")
        print("[LLAMA-3 RECURRENT]: ", end="", flush=True)
        generate_stream_llama3(model, tokenizer, p, device, max_tokens=60, temperature=0.6)

if __name__ == "__main__":
    main()
