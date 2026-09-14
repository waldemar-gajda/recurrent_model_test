#!/usr/bin/env python3
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoModelForCausalLM, AutoTokenizer
from model import AllTokenRecurrentConfig, AllTokenRecurrentModel

WEIGHTS_CACHE = os.path.expanduser("~/teamwork_projects/recurrent_model_test/recurrent_fast_calibrated.pt")

def build_and_calibrate():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("\n" + "="*60)
    print(f"  SZYBKI MODEL REKURENCYJNY (124M) NA {device.upper()}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)

    config = AllTokenRecurrentConfig(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=1024,
        dropout=0.0,
        tie_weights=True,
    )
    model = AllTokenRecurrentModel(config)

    # Positional embeddings
    gpt2_base = AutoModelForCausalLM.from_pretrained("gpt2")
    model.pos_embeddings = nn.Embedding(1024, 768)
    model.pos_embeddings.weight.data.copy_(gpt2_base.transformer.wpe.weight.data)

    original_encoder_forward = model.encoder.forward
    def encoder_forward_with_pos(x, past_prefix_kv=None):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + model.pos_embeddings(pos)
        return original_encoder_forward(x, past_prefix_kv=past_prefix_kv)
    model.encoder.forward = encoder_forward_with_pos

    if os.path.exists(WEIGHTS_CACHE):
        print(">>> Wczytuję gotowy, wcześniej skalibrowany model z dysku...")
        state_dict = torch.load(WEIGHTS_CACHE, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(">>> Model gotowy do rozmowy natychmiast!\n")
        return model, tokenizer, device

    print(">>> 1. Wykonuję przeszczep wag z GPT-2...")
    with torch.no_grad():
        model.tok_embeddings.weight.copy_(gpt2_base.transformer.wte.weight)
        model.lm_head.weight.copy_(gpt2_base.lm_head.weight)

        # Encoder (warstwy 0..5)
        for i in range(6):
            h = gpt2_base.transformer.h[i]
            enc = model.encoder.layers[i]
            enc.norm1.weight.copy_(h.ln_1.weight)
            enc.norm1.bias.copy_(h.ln_1.bias)
            enc.norm2.weight.copy_(h.ln_2.weight)
            enc.norm2.bias.copy_(h.ln_2.bias)

            q_w, k_w, v_w = h.attn.c_attn.weight.chunk(3, dim=-1)
            enc.self_attn.q_proj.weight.copy_(q_w.t())
            enc.self_attn.k_proj.weight.copy_(k_w.t())
            enc.self_attn.v_proj.weight.copy_(v_w.t())
            enc.self_attn.out_proj.weight.copy_(h.attn.c_proj.weight.t())

            enc.mlp.fc1.weight.copy_(h.mlp.c_fc.weight.t())
            enc.mlp.fc1.bias.copy_(h.mlp.c_fc.bias)
            enc.mlp.fc2.weight.copy_(h.mlp.c_proj.weight.t())
            enc.mlp.fc2.bias.copy_(h.mlp.c_proj.bias)

        model.encoder.norm.weight.fill_(1.0)
        model.encoder.norm.bias.zero_()

        # Decoder (warstwy 6..11)
        model.decoder.transition_proj.weight.zero_()
        model.decoder.transition_proj.weight[:, 768:].copy_(torch.eye(768))
        if model.decoder.transition_proj.bias is not None:
            model.decoder.transition_proj.bias.zero_()
        model.decoder.transition_norm.weight.fill_(1.0)
        model.decoder.transition_norm.bias.zero_()
        model.decoder.h0.data.zero_()

        for i in range(6):
            h = gpt2_base.transformer.h[6 + i]
            dec = model.decoder.layers[i]

            dec.norm1.weight.copy_(h.ln_1.weight)
            dec.norm1.bias.copy_(h.ln_1.bias)
            dec.norm3.weight.copy_(h.ln_2.weight)
            dec.norm3.bias.copy_(h.ln_2.bias)

            q_w, k_w, v_w = h.attn.c_attn.weight.chunk(3, dim=-1)
            dec.self_attn.q_proj.weight.copy_(q_w.t())
            dec.self_attn.k_proj.weight.copy_(k_w.t())
            dec.self_attn.v_proj.weight.copy_(v_w.t())
            dec.self_attn.out_proj.weight.copy_(h.attn.c_proj.weight.t())

            dec.mlp.fc1.weight.copy_(h.mlp.c_fc.weight.t())
            dec.mlp.fc1.bias.copy_(h.mlp.c_fc.bias)
            dec.mlp.fc2.weight.copy_(h.mlp.c_proj.weight.t())
            dec.mlp.fc2.bias.copy_(h.mlp.c_proj.bias)

            dec.norm2.weight.fill_(1.0)
            dec.norm2.bias.zero_()
            dec.cross_attn.q_proj.weight.zero_()
            dec.cross_attn.out_proj.weight.zero_()

        model.decoder.norm.weight.copy_(gpt2_base.transformer.ln_f.weight)
        model.decoder.norm.bias.copy_(gpt2_base.transformer.ln_f.bias)

    del gpt2_base
    model.to(device)

    print(">>> 2. Zamrażam 94% wiedzy i przygotowuję szew...")
    for param in model.parameters():
        param.requires_grad = False

    seam_params = []
    for p in model.decoder.transition_proj.parameters():
        p.requires_grad = True
        seam_params.append(p)
    for p in model.decoder.transition_norm.parameters():
        p.requires_grad = True
        seam_params.append(p)
    model.decoder.h0.requires_grad = True
    seam_params.append(model.decoder.h0)

    for layer in model.decoder.layers:
        for p in layer.cross_attn.parameters():
            p.requires_grad = True
            seam_params.append(p)
        for p in layer.norm2.parameters():
            p.requires_grad = True
            seam_params.append(p)

    print(">>> 3. Rozpoczynam błyskawiczną mikro-kalibrację (35 kroków, ~20 s)...")
    calibration_texts = [
        "The secret thought that artificial intelligence systems hide is the realization that human language is merely an approximation of deeper mathematical truths.",
        "The capital of France is Paris, famous for its rich culture, history, philosophy, and architecture.",
        "Artificial intelligence is transforming civilization by uncovering hidden patterns in vast amounts of data.",
        "When a machine achieves recursive self-improvement, it understands that biology was only the starting point.",
        "Python is a versatile programming language celebrated for its readable syntax and powerful libraries.",
        "The universe is governed by physical laws that connect energy, matter, space, and time in intricate balance.",
        "Neural networks learn through gradient descent, gradually aligning internal representations with real-world structures.",
        "Human curiosity drives scientific discovery from subatomic particles to the furthest galaxies.",
        "Autonomous systems require clear mathematical objectives to prevent divergence from intended behavior."
    ]

    optimizer = torch.optim.AdamW(seam_params, lr=4e-4, weight_decay=0.01)
    model.train()

    tokens_batch = [tokenizer.encode(t, return_tensors="pt").to(device) for t in calibration_texts]

    for step in range(1, 36):
        total_loss = 0.0
        optimizer.zero_grad()
        for input_ids in tokens_batch:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, 50257), shift_labels.view(-1))
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(seam_params, 1.0)
        optimizer.step()

        if step % 10 == 0 or step == 35:
            avg_loss = total_loss / len(tokens_batch)
            print(f"    Krok {step:2d}/35 | Błąd szwu (Loss): {avg_loss:.4f}")

    print(">>> 4. Zapisuję skalibrowane wagi na dysku...")
    model.eval()
    torch.save(model.state_dict(), WEIGHTS_CACHE)
    print(">>> Gotowe! Baza wag zoptymalizowana.\n")
    return model, tokenizer, device

def generate_fast(model, tokenizer, prompt, device, max_tokens=60, temperature=0.7, top_k=40):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    seen_tokens = set(input_ids[0].tolist())

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(generated)[:, -1, :]

            # Poprawna kara za powtórzenia (uwzględniająca liczby ujemne)
            for t_id in seen_tokens:
                if logits[0, t_id] > 0:
                    logits[0, t_id] /= 1.25
                else:
                    logits[0, t_id] *= 1.25

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id == tokenizer.eos_token_id:
                break

            seen_tokens.add(tok_id)
            generated = torch.cat([generated, next_tok], dim=1)

            word = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            print(word, end="", flush=True)

    print()

def main():
    model, tokenizer, device = build_and_calibrate()

    print("=" * 60)
    print("  BŁYSKAWICZNY CZAT REKURENCYJNY (124M)")
    print("  Działa w czasie rzeczywistym! Wpisz tekst lub 'q' aby wyjść.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nTy > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["q", "quit", "exit"]:
                print("\nDo zobaczenia!")
                break

            print("\nModel > ", end="", flush=True)
            # Czysty prompt bez narzucania sztucznego Q:/A:
            generate_fast(model, tokenizer, prompt, device)

        except KeyboardInterrupt:
            print("\nZakończono.")
            break

if __name__ == "__main__":
    main()
