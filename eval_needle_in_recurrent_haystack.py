import os
import sys
import time
import glob
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from safetensors import safe_open

from llama_recurrent_model import LlamaRecurrentConfig, LlamaAllTokenRecurrentModel
from recurrent_bridge import GatedRecurrentBridge
from dataset_memory_recall import generate_dataset

INSTRUCT_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama-3-8b-instruct")
ADAPTER_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/recurrent_memory_adapter.pt")

CROSS_ADAPTER_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/recurrent_cross_memory.pt")

def format_instruct_prompt(messages):
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted

def load_eval_model(mode="cross_attn", device="mps"):
    print("=" * 60)
    print(f"  ŁADOWANIE MODELU DO EWALUACJI (TRYB: {mode.upper()})")
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
                                dec.cross_attn.o_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.gate_proj.weight": dec.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight": dec.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight": dec.mlp.down_proj.weight.copy_(tensor)
                    del tensor
            gc.collect()

    if mode == "cross_attn":
        bank = model.enable_memory_bank(num_slots=8)
        if os.path.exists(CROSS_ADAPTER_PATH):
            print(f">>> 2. Ładowanie wag banku pamięci z: {CROSS_ADAPTER_PATH}")
            state_dict = torch.load(CROSS_ADAPTER_PATH, map_location=device)
            bank.load_state_dict(state_dict)
            print("    Bank pamięci załadowany pomyślnie!")
        else:
            print(">>> 2. Uwaga: Plik wag banku pamięci jeszcze nie istnieje (tryb niezainicjalizowany)")
    else:
        bridge = model.enable_gated_bridge()
        if os.path.exists(ADAPTER_PATH):
            print(f">>> 2. Ładowanie wag adaptera pojedynczego wektora z: {ADAPTER_PATH}")
            state_dict = torch.load(ADAPTER_PATH, map_location=device)
            bridge.load_state_dict(state_dict)

    model.eval()
    return model, tokenizer, device

def evaluate_needle_recall(num_tests=10, mode="cross_attn"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer, device = load_eval_model(mode=mode, device=device)
    test_data = generate_dataset(num_samples=num_tests, base_seed=777)
    
    hits = 0
    total = len(test_data)

    print("\n" + "=" * 65)
    print(f"  BENCHMARK: NEEDLE RECALL (TRYB: {mode.upper()})")
    if mode == "cross_attn":
        print("  Tura 1 (Passus) -> [Bank Pamięci K-Slot (16 KB)] -> Tura 2 (Pytanie)")
    else:
        print("  Tura 1 (Passus) -> [Pojedynczy Wektor H (8 KB)] -> Tura 2 (Pytanie)")
    print("=" * 65)

    for i, (passage, question, expected_answer) in enumerate(test_data, 1):
        with torch.no_grad():
            p_prompt = format_instruct_prompt([
                {"role": "system", "content": "Zapamiętaj poniższy dokument."},
                {"role": "user", "content": passage},
                {"role": "assistant", "content": "Zrozumiałem i zapamiętałem treść."}
            ])
            p_ids = tokenizer.encode(p_prompt, return_tensors="pt").to(device)

            q_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            q_ids = tokenizer.encode(q_prompt, return_tensors="pt").to(device)
            B, T = q_ids.shape

            if mode == "cross_attn":
                mem_k, mem_v = model.encode_to_memory_bank(p_ids)
                logits, enc_cache, dec_cache, h = model.forward_with_cache(q_ids, memory_bank_kv=(mem_k, mem_v))
                mem_kv_arg = (mem_k, mem_v)
            else:
                _, _, _, h_passage = model.forward_with_cache(p_ids)
                logits, enc_cache, dec_cache, h = model.forward_with_cache(q_ids, initial_h=h_passage)
                mem_kv_arg = None

            last_logits = logits[:, -1, :].clone()
            pos = T

            gen_tokens = []
            for _ in range(30):
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
                tok_id = next_tok.item()
                if tok_id in [tokenizer.eos_token_id, 128001, 128009]:
                    break
                gen_tokens.append(tok_id)
                if mode == "cross_attn":
                    last_logits, enc_cache, dec_cache, h = model.step(
                        next_tok, pos, enc_cache, dec_cache, h, memory_bank_kv=mem_kv_arg
                    )
                else:
                    last_logits, enc_cache, dec_cache, h = model.step(next_tok, pos, enc_cache, dec_cache, h)
                pos += 1

            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            is_hit = expected_answer.lower() in gen_text.lower()
            if is_hit:
                hits += 1

            status = "✓ TRAFIENIE" if is_hit else "✗ PUDŁO"
            print(f"\n[Test {i:02d}/{total:02d}] {status}")
            print(f"  Pytanie:    {question}")
            print(f"  Oczekiwana: {expected_answer}")
            print(f"  Odpowiedź:  {gen_text}")

    acc = (hits / total) * 100
    print("\n" + "=" * 65)
    print(f"  WYNIK KOŃCOWY ({mode}): {hits}/{total} ({acc:.1f}% dokładności)")
    print("=" * 65)
    return acc

if __name__ == "__main__":
    mode = "single_vector" if "--single" in sys.argv else "cross_attn"
    evaluate_needle_recall(num_tests=10, mode=mode)
