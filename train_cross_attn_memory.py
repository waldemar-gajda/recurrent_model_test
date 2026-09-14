import os
import sys
import time
import glob
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from safetensors import safe_open

from llama_recurrent_model import LlamaRecurrentConfig, LlamaAllTokenRecurrentModel
from dataset_memory_recall import generate_dataset

INSTRUCT_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama-3-8b-instruct")
SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/recurrent_cross_memory.pt")

def format_instruct_prompt(messages):
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted

def load_llama_for_memory_bank(device="mps"):
    print("=" * 60)
    print(f"  ŁADOWANIE MODELU DLA TRENINGU BANKU PAMIĘCI K-SLOT ({device.upper()})")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_DIR, clean_up_tokenization_spaces=False)
    config = LlamaRecurrentConfig(
        vocab_size=128256,
        d_model=4096,
        intermediate_size=14336,
        num_heads=32,
        num_kv_heads=8,
        num_encoder_layers=16,
        num_decoder_layers=16,
        max_seq_len=2048,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )

    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = LlamaAllTokenRecurrentModel(config)
    torch.set_default_dtype(torch.float32)

    safetensors_files = sorted(glob.glob(os.path.join(INSTRUCT_DIR, "*.safetensors")))
    print(f">>> 1. Transfer wag Llama-3-8B-Instruct...")
    with torch.no_grad():
        for shard_file in safetensors_files:
            with safe_open(shard_file, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if key == "model.embed_tokens.weight": model.embed_tokens.weight.copy_(tensor)
                    elif key == "lm_head.weight": model.lm_head.weight.copy_(tensor)
                    elif key == "model.norm.weight": model.decoder_norm.weight.copy_(tensor)
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
                            elif sub_key == "self_attn.o_proj.weight":
                                dec.self_attn.o_proj.weight.copy_(tensor)
                                # Włączamy wstępną projekcję z wag self-attention!
                                dec.cross_attn.o_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.gate_proj.weight": dec.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": dec.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": dec.mlp.down_proj.weight.copy_(tensor)
                    del tensor
            gc.collect()

    # Zamrażamy wszystkie bazowe parametry LLaMA
    print(">>> 2. Zamrażanie wag bazowych LLaMA-3...")
    for param in model.parameters():
        param.requires_grad = False

    # Inicjalizujemy RecurrentMemoryBank (M=8 slotów pamięci)
    print(">>> 3. Inicjalizacja K-Slot RecurrentMemoryBank (M=8 slotów)...")
    bank = model.enable_memory_bank(num_slots=8, init_gate_bias=-2.0)
    for p in bank.parameters():
        p.requires_grad = True

    trainable_count = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    print(f">>> Gotowe! Trenowalne parametry: {trainable_count / 1e6:.2f}M ({trainable_count * 2 / 1024**2:.1f} MB)\n")
    return model, tokenizer, bank

def evaluate_sample(model, tokenizer, bank, passage, question, answer, device):
    model.eval()
    with torch.no_grad():
        # Tura 1: Kompresja passusu do 8 slotów pamięci
        p_prompt = format_instruct_prompt([
            {"role": "system", "content": "Zapamiętaj poniższy dokument."},
            {"role": "user", "content": passage},
            {"role": "assistant", "content": "Zrozumiałem i zapamiętałem treść."}
        ])
        p_ids = tokenizer.encode(p_prompt, return_tensors="pt").to(device)
        mem_k, mem_v = model.encode_to_memory_bank(p_ids)

        # Tura 2: Odpytywanie WYŁĄCZNIE pytaniem
        q_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = tokenizer.encode(q_prompt, return_tensors="pt").to(device)
        B, T = input_ids.shape

        logits, enc_cache, dec_cache, h = model.forward_with_cache(input_ids, memory_bank_kv=(mem_k, mem_v))
        last_logits = logits[:, -1, :].clone()
        pos = T

        gen_tokens = []
        for _ in range(25):
            next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 128001, 128009]:
                break
            gen_tokens.append(tok_id)
            last_logits, enc_cache, dec_cache, h = model.step(
                next_tok, pos, enc_cache, dec_cache, h, memory_bank_kv=(mem_k, mem_v)
            )
            pos += 1

        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        is_hit = answer.lower() in generated_text.lower()
        return generated_text, is_hit

def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer, bank = load_llama_for_memory_bank(device)

    # Przygotowanie danych
    print(">>> Przygotowanie zbioru danych pamięciowych...")
    train_data = generate_dataset(num_samples=40, base_seed=100)
    val_data = generate_dataset(num_samples=10, base_seed=999)
    print(f"    Train: {len(train_data)} próbek, Val: {len(val_data)} próbek")

    # Test baseline PRZED treningiem
    print("\n--- TEST PRZED TRENINGIEM (BASELINE ZERO-SHOT) ---")
    val_sample = val_data[0]
    gen_text, is_hit = evaluate_sample(model, tokenizer, bank, val_sample[0], val_sample[1], val_sample[2], device)
    print(f"Passus:    {val_sample[0]}")
    print(f"Pytanie:   {val_sample[1]}")
    print(f"Oczekiwana: {val_sample[2]}")
    print(f"Odpowiedź: {gen_text}")
    print(f"Trafienie:  {'TAK' if is_hit else 'NIE (zgodnie z oczekiwaniem)'}\n")

    # Optymalizator
    lr = 5e-4
    optimizer = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("  ROZPOCZĘCIE TRENINGU PAMIĘCI ASOCJACYJNEJ K-SLOT")
    print("=" * 60)

    num_epochs = 4
    model.eval()
    bank.train()

    for epoch in range(num_epochs):
        print(f"\n--- EPOKA {epoch + 1}/{num_epochs} ---")
        epoch_loss = 0.0
        
        for sample_i, (passage, question, answer) in enumerate(train_data, 1):
            t_start = time.time()
            optimizer.zero_grad()

            # 1. Krok A: Enkodowanie passusu do 8 slotów pamięci
            p_prompt = format_instruct_prompt([
                {"role": "system", "content": "Zapamiętaj poniższy dokument."},
                {"role": "user", "content": passage},
                {"role": "assistant", "content": "Zrozumiałem i zapamiętałem treść."}
            ])
            p_ids = tokenizer.encode(p_prompt, return_tensors="pt").to(device)
            mem_k, mem_v = model.encode_to_memory_bank(p_ids)

            # 2. Krok B: Tura pytania + odpowiedzi
            prefix_str = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            target_str = f"{answer}<|eot_id|>"

            prefix_ids = tokenizer.encode(prefix_str, return_tensors="pt").to(device)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False, return_tensors="pt").to(device)
            full_ids = torch.cat([prefix_ids, target_ids], dim=1)

            prefix_len = prefix_ids.shape[1]
            total_len = full_ids.shape[1]

            logits = model(full_ids, memory_bank_kv=(mem_k, mem_v))

            shift_logits = logits[:, (prefix_len - 1) : (total_len - 1), :].contiguous()
            shift_labels = full_ids[:, prefix_len:total_len].contiguous()

            loss = loss_fn(shift_logits.view(-1, 128256).float(), shift_labels.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(bank.parameters(), max_norm=1.0)
            optimizer.step()

            step_time = time.time() - t_start
            epoch_loss += loss.item()

            gate_val = torch.sigmoid(bank.cross_gates).mean().item()
            if sample_i % 5 == 0 or sample_i == len(train_data):
                print(f"  [Krok {sample_i:02d}/{len(train_data):02d}] Loss: {loss.item():.4f} | Gate mean: {gate_val:.4f} | Czas: {step_time:.2f}s")

        avg_loss = epoch_loss / len(train_data)
        print(f"\nŚrednia strata w epoce {epoch + 1}: {avg_loss:.4f}")

        # Ewaluacja po epoce
        gen_text, is_hit = evaluate_sample(model, tokenizer, bank, val_sample[0], val_sample[1], val_sample[2], device)
        print(f"  Ewaluacja po epoce {epoch + 1}:")
        print(f"  Oczekiwana: {val_sample[2]}")
        print(f"  Odpowiedź: {gen_text}")
        print(f"  Trafienie:  {'TAK! (Fakt odzyskany z pamięci asocjacyjnej)' if is_hit else 'NIE'}")

    print(f"\n>>> Zapisywanie wag banku pamięci do: {SAVE_PATH}")
    torch.save(bank.state_dict(), SAVE_PATH)
    print("Zapisano pomyślnie!")

if __name__ == "__main__":
    train()
