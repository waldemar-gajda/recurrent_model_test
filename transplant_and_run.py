import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import AllTokenRecurrentConfig, AllTokenRecurrentModel

def run_transplant_with_positional_embeddings():
    print(">>> 1. Loading Pretrained GPT-2 & Tokenizer...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2.eval()

    print(">>> 2. Initializing All-Token Recurrence Model with GPT-2 specs...")
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
    model.eval()

    # Add positional embeddings from GPT-2 to preserve word order
    model.pos_embeddings = nn.Embedding(1024, 768)
    model.pos_embeddings.weight.data.copy_(gpt2.transformer.wpe.weight.data)

    # Wrap forward to add positional encoding: x_embed = tok_embed + pos_embed
    original_encoder_forward = model.encoder.forward
    def encoder_forward_with_pos(x, past_prefix_kv=None):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + model.pos_embeddings(pos)
        return original_encoder_forward(x, past_prefix_kv=past_prefix_kv)
    model.encoder.forward = encoder_forward_with_pos

    print(">>> 3. Performing Calibrated Weight Surgery...")
    with torch.no_grad():
        # A. Embeddings and LM Head
        model.tok_embeddings.weight.copy_(gpt2.transformer.wte.weight)
        model.lm_head.weight.copy_(gpt2.lm_head.weight)

        # B. Phase 01: Causal Encoder (GPT-2 layers 0..5)
        for i in range(6):
            h = gpt2.transformer.h[i]
            enc = model.encoder.layers[i]
            
            enc.norm1.weight.copy_(h.ln_1.weight)
            enc.norm1.bias.copy_(h.ln_1.bias)
            enc.norm2.weight.copy_(h.ln_2.weight)
            enc.norm2.bias.copy_(h.ln_2.bias)

            qkv_w = h.attn.c_attn.weight
            q_w, k_w, v_w = qkv_w.chunk(3, dim=-1)
            enc.self_attn.q_proj.weight.copy_(q_w.t())
            enc.self_attn.k_proj.weight.copy_(k_w.t())
            enc.self_attn.v_proj.weight.copy_(v_w.t())
            enc.self_attn.out_proj.weight.copy_(h.attn.c_proj.weight.t())

            enc.mlp.fc1.weight.copy_(h.mlp.c_fc.weight.t())
            enc.mlp.fc1.bias.copy_(h.mlp.c_fc.bias)
            enc.mlp.fc2.weight.copy_(h.mlp.c_proj.weight.t())
            enc.mlp.fc2.bias.copy_(h.mlp.c_proj.bias)

        # Neutral norm on encoder boundary so activation scale doesn't drift
        model.encoder.norm.weight.fill_(1.0)
        model.encoder.norm.bias.zero_()

        # C. Phase 02: Recurrent Transition Decoder (GPT-2 layers 6..11)
        model.decoder.transition_proj.weight.zero_()
        model.decoder.transition_proj.weight[:, 768:].copy_(torch.eye(768))
        if model.decoder.transition_proj.bias is not None:
            model.decoder.transition_proj.bias.zero_()
        model.decoder.transition_norm.weight.fill_(1.0)
        model.decoder.transition_norm.bias.zero_()
        model.decoder.h0.data.zero_()

        for i in range(6):
            h = gpt2.transformer.h[6 + i]
            dec = model.decoder.layers[i]

            dec.norm1.weight.copy_(h.ln_1.weight)
            dec.norm1.bias.copy_(h.ln_1.bias)
            dec.norm3.weight.copy_(h.ln_2.weight)
            dec.norm3.bias.copy_(h.ln_2.bias)

            qkv_w = h.attn.c_attn.weight
            q_w, k_w, v_w = qkv_w.chunk(3, dim=-1)
            dec.self_attn.q_proj.weight.copy_(q_w.t())
            dec.self_attn.k_proj.weight.copy_(k_w.t())
            dec.self_attn.v_proj.weight.copy_(v_w.t())
            dec.self_attn.out_proj.weight.copy_(h.attn.c_proj.weight.t())

            dec.mlp.fc1.weight.copy_(h.mlp.c_fc.weight.t())
            dec.mlp.fc1.bias.copy_(h.mlp.c_fc.bias)
            dec.mlp.fc2.weight.copy_(h.mlp.c_proj.weight.t())
            dec.mlp.fc2.bias.copy_(h.mlp.c_proj.bias)

            # Prefix cross-attention: Identity passthrough
            dec.norm2.weight.fill_(1.0)
            dec.norm2.bias.zero_()
            dec.cross_attn.q_proj.weight.zero_()
            dec.cross_attn.out_proj.weight.zero_()

        model.decoder.norm.weight.copy_(gpt2.transformer.ln_f.weight)
        model.decoder.norm.bias.copy_(gpt2.transformer.ln_f.bias)

    print(">>> 4. Surgery Completed with Positional Encodings!")

    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "The fastest animal on Earth is the",
        "Python is a programming language that",
    ]

    print("\n" + "="*60)
    print(">>> 5. COMPARING ORIGINAL GPT-2 vs TRANSPLANTED RECURRENT MODEL")
    print("="*60)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Original GPT-2 generation
        orig_ids = gpt2.generate(input_ids, max_new_tokens=8, do_sample=False)
        orig_text = tokenizer.decode(orig_ids[0])

        # Recurrent Model generation
        rec_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(8):
                logits = model(rec_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                rec_ids = torch.cat([rec_ids, next_tok], dim=1)
        rec_text = tokenizer.decode(rec_ids[0])

        print(f"\n[PROMPT]: '{prompt}'")
        print(f"  -> Original GPT-2:     '{orig_text}'")
        print(f"  -> Recurrent Model:    '{rec_text}'")

if __name__ == "__main__":
    run_transplant_with_positional_embeddings()
