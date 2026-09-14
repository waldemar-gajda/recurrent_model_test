import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import AllTokenRecurrentConfig, AllTokenRecurrentModel

def build_transplanted_model(device="mps"):
    print(">>> 1. Loading Pretrained GPT-2...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2.eval()

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

    # Positional embeddings from GPT-2
    model.pos_embeddings = nn.Embedding(1024, 768)
    model.pos_embeddings.weight.data.copy_(gpt2.transformer.wpe.weight.data)

    original_encoder_forward = model.encoder.forward
    def encoder_forward_with_pos(x, past_prefix_kv=None):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + model.pos_embeddings(pos)
        return original_encoder_forward(x, past_prefix_kv=past_prefix_kv)
    model.encoder.forward = encoder_forward_with_pos

    print(">>> 2. Performing Organ Transplant from GPT-2...")
    with torch.no_grad():
        model.tok_embeddings.weight.copy_(gpt2.transformer.wte.weight)
        model.lm_head.weight.copy_(gpt2.lm_head.weight)

        # Encoder (layers 0..5)
        for i in range(6):
            h = gpt2.transformer.h[i]
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

        # Decoder (layers 6..11)
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

        model.decoder.norm.weight.copy_(gpt2.transformer.ln_f.weight)
        model.decoder.norm.bias.copy_(gpt2.transformer.ln_f.bias)

    model.to(device)
    return model, tokenizer

def run_adaptive_stitch():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute acceleration: {device.upper()}")

    model, tokenizer = build_transplanted_model(device=device)

    # 1. Freeze 99% of the model (preserving pre-trained intelligence)
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze ONLY the "adaptive seam" (recurrent transition bridge)
    seam_params = []
    for p in model.decoder.transition_proj.parameters():
        p.requires_grad = True
        seam_params.append(p)
    for p in model.decoder.transition_norm.parameters():
        p.requires_grad = True
        seam_params.append(p)
    model.decoder.h0.requires_grad = True
    seam_params.append(model.decoder.h0)

    # Also unfreeze cross-attention output projection and norms in decoder
    for layer in model.decoder.layers:
        for p in layer.cross_attn.parameters():
            p.requires_grad = True
            seam_params.append(p)
        for p in layer.norm2.parameters():
            p.requires_grad = True
            seam_params.append(p)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_seam = sum(p.numel() for p in seam_params)

    print("\n" + "="*60)
    print(">>> 3. ADAPTIVE SEAM CONFIGURATION")
    print("="*60)
    print(f"Total Model Parameters:      {total_params:,}")
    print(f"Trainable Seam Parameters:   {trainable_seam:,} ({trainable_seam/total_params*100:.2f}% of model)")
    print(f"Frozen Pretrained Knowledge: {total_params - trainable_seam:,} ({(total_params-trainable_seam)/total_params*100:.2f}%)")

    # 3. Calibration dataset: High-quality natural language calibration sentences
    calibration_texts = [
        "The capital of France is Paris, one of the most famous cities in the world.",
        "Artificial intelligence is transforming science, medicine, and engineering across the globe.",
        "Python is a popular programming language known for its simplicity and versatility in data science.",
        "The Earth revolves around the Sun in an elliptical orbit taking approximately 365 days.",
        "Computer science involves the study of computation, algorithms, information, and automation.",
        "Neural networks learn representations of data through optimization algorithms like gradient descent.",
        "Deep learning models can recognize complex patterns in images, audio recordings, and text sequences.",
        "Open source software empowers developers worldwide to collaborate and build transparent tools.",
        "Quantum computing explores principles of quantum mechanics to solve specialized computational problems.",
        "Solar energy is a renewable source of power captured using modern photovoltaic cells and panels.",
        "The human brain contains billions of neurons communicating through synaptic electrical impulses.",
        "Mathematics provides the foundational framework for physics, statistics, and machine learning models."
    ]

    print("\n" + "="*60)
    print(">>> 4. RUNNING MICRO-CALIBRATION (ADAPTIVE STITCH)...")
    print("="*60)

    optimizer = torch.optim.AdamW(seam_params, lr=3e-4, weight_decay=0.01)
    model.train()

    # Fast micro-calibration (35 steps)
    tokens_batch = [tokenizer.encode(t, return_tensors="pt").to(device) for t in calibration_texts]

    for step in range(1, 36):
        total_loss = 0.0
        optimizer.zero_grad()
        for input_ids in tokens_batch:
            if input_ids.size(1) < 4:
                continue
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, config_vocab := 50257), shift_labels.view(-1))
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(seam_params, 1.0)
        optimizer.step()

        avg_loss = total_loss / len(tokens_batch)
        if step % 5 == 0 or step == 1:
            print(f"Step {step:2d}/35 | Seam Calibration Loss: {avg_loss:.4f}")

    print("\n>>> 5. Micro-calibration finished! Testing text generation...")
    model.eval()

    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "Python is a programming language that",
    ]

    print("\n" + "="*60)
    print(">>> 6. RESULTS AFTER ADAPTIVE SEAM CALIBRATION")
    print("="*60)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(12):
                logits = model(generated_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_tok], dim=1)

        result_text = tokenizer.decode(generated_ids[0].cpu())
        print(f"\n[PROMPT]: '{prompt}'")
        print(f"[RECURRENT STITCHED]: '{result_text}'")

if __name__ == "__main__":
    run_adaptive_stitch()
