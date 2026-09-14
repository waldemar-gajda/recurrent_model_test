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

INSTRUCT_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama-3-8b-instruct")

def load_recurrent_llama_instruct():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print("\n" + "="*60)
    print(f"  ŁADOWANIE META-LLAMA-3-8B-INSTRUCT NA {device.upper()}...")
    print("="*60)

    if not os.path.exists(INSTRUCT_DIR) or not glob.glob(os.path.join(INSTRUCT_DIR, "*.safetensors")):
        print(f"Błąd: Nie znaleziono plików safetensors w: {INSTRUCT_DIR}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_DIR, clean_up_tokenization_spaces=False)
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

    with torch.no_grad():
        model.h_proj.weight.zero_()
        model.h0.zero_()

    safetensors_files = sorted(glob.glob(os.path.join(INSTRUCT_DIR, "*.safetensors")))
    print(f">>> 2. Błyskawiczny transfer wag Llama-3-8B-Instruct z {len(safetensors_files)} plików...")

    with torch.no_grad():
        for shard_idx, shard_file in enumerate(safetensors_files, 1):
            print(f"    [Shard {shard_idx}/{len(safetensors_files)}] {os.path.basename(shard_file)}...", flush=True)
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
            gc.collect()

    print(f"\n>>> Model Llama-3-8B-Instruct gotowy do rozmowy na MPS!\n")
    model.eval()
    return model, tokenizer, device

def generate_stream_instruct(
    model,
    tokenizer,
    prompt,
    device,
    current_h=None,
    max_tokens=500,
    temperature=0.6,
    top_p=0.9,
):
    """
    Natywna generacja konwersacyjna Llama-3-Instruct z tokenami sterującymi.
    Automatycznie zatrzymuje się na tokenie <|eot_id|> (128009).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    B, T = input_ids.shape

    with torch.no_grad():
        logits, enc_cache, dec_cache, h = model.forward_with_cache(input_ids, initial_h=current_h)
        last_logits = logits[:, -1, :].clone()
        pos = T

        generated_tokens = []
        printed_text = ""

        # Stop tokens oficjalne dla Llama-3 Instruct
        stop_token_ids = {
            tokenizer.eos_token_id,
            128001, # <|end_of_text|>
            128009, # <|eot_id|>
        }

        for step_i in range(max_tokens):
            if temperature > 0:
                scaled_logits = last_logits / temperature
                # Top-p (nucleus sampling)
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = -float('Inf')

                probs = torch.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id in stop_token_ids:
                break

            generated_tokens.append(tok_id)

            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not full_text.endswith("\ufffd"):
                new_text = full_text[len(printed_text):]
                if new_text:
                    print(new_text, end="", flush=True)
                    printed_text = full_text

            last_logits, enc_cache, dec_cache, h = model.step(
                next_tok, pos, enc_cache, dec_cache, h
            )
            pos += 1

        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(final_text) > len(printed_text):
            print(final_text[len(printed_text):], end="", flush=True)
            printed_text = final_text

    print()
    return h, printed_text

def format_instruct_prompt(messages):
    """
    Formatowanie oficjalnego szablonu czatu Meta Llama-3:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted

def main():
    model, tokenizer, device = load_recurrent_llama_instruct()

    print("=" * 60)
    print("  META-LLAMA-3-8B-INSTRUCT — LIVE CHAT Z PAMIĘCIĄ REKURENCYJNĄ")
    print("  Natywny model czatowy z oficjalnymi znacznikami dialogowymi Mety.")
    print("  Rozmawiaj swobodnie po polsku i po angielsku. Pamięta cały wątek.")
    print("  Wpisz 'reset' aby zacząć nowy wątek, lub 'q' aby wyjść.")
    print("=" * 60)

    system_prompt = "Jesteś wybitnym, elokwentnym asystentem AI. Odpowiadasz mądrze, bezpośrednio, z głębokim zrozumieniem kontekstu i zawsze w języku, w którym pisze użytkownik."
    messages = [{"role": "system", "content": system_prompt}]
    current_h = None
    turn_idx = 1

    while True:
        try:
            user_input = input(f"\nTy > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\nDo widzenia!")
                break
            if user_input.lower() in ["reset", "/reset", "nowy"]:
                current_h = None
                messages = [{"role": "system", "content": system_prompt}]
                turn_idx = 1
                print("\n[Pamięć wątku zresetowana — nowy wątek rozmowy]")
                continue

            messages.append({"role": "user", "content": user_input})
            prompt = format_instruct_prompt(messages)

            print("\nLlama-3 Instruct > ", end="", flush=True)
            current_h, response_text = generate_stream_instruct(model, tokenizer, prompt, device, current_h=current_h)
            messages.append({"role": "assistant", "content": response_text})
            turn_idx += 1

        except KeyboardInterrupt:
            print("\nZakończono.")
            break

if __name__ == "__main__":
    main()
