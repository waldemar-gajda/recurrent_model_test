import os
import glob
import gc
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from llama_recurrent_model import LlamaRecurrentConfig, LlamaAllTokenRecurrentModel

def run_llama3_transplant():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f">>> Target Device: {device.upper()} | Precision: {dtype}")

    model_dir = os.path.expanduser("~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots")
    snapshots = glob.glob(os.path.join(model_dir, "*"))
    if not snapshots:
        print("Error: Llama-3-8B snapshot directory not found.")
        return
    snapshot_path = snapshots[0]
    print(f">>> Model Directory: {snapshot_path}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)

    config = LlamaRecurrentConfig()
    print("\n>>> Allocating Llama-3 All-Token Recurrent Model directly on", device.upper(), "(~16 GB)...")
    model = LlamaAllTokenRecurrentModel(config).to(dtype=dtype, device=device)

    # Initialize recurrent transition bridge
    with torch.no_grad():
        model.transition_proj.weight.zero_()
        model.transition_proj.weight[:, config.d_model:].copy_(torch.eye(config.d_model, dtype=dtype, device=device))
        model.transition_norm.weight.fill_(1.0)
        model.encoder_norm.weight.fill_(1.0)
        model.h0.zero_()

    safetensors_files = sorted(glob.glob(os.path.join(snapshot_path, "*.safetensors")))
    print(f"\n>>> Streaming 8 Billion parameters tensor-by-tensor from {len(safetensors_files)} shards...")

    total_streamed = 0
    for shard_idx, shard_file in enumerate(safetensors_files, 1):
        print(f"  -> Streaming Shard {shard_idx}/{len(safetensors_files)}: {os.path.basename(shard_file)}...")
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key).to(dtype=dtype, device=device)
                total_streamed += tensor.numel()

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

                        # Layers 0..15 -> Causal Encoder
                        if layer_idx < 16:
                            enc_layer = model.encoder_layers[layer_idx]
                            if sub_key == "input_layernorm.weight":
                                enc_layer.input_layernorm.weight.copy_(tensor)
                            elif sub_key == "post_attention_layernorm.weight":
                                enc_layer.post_attention_layernorm.weight.copy_(tensor)
                            elif sub_key == "self_attn.q_proj.weight":
                                enc_layer.self_attn.q_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.k_proj.weight":
                                enc_layer.self_attn.k_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.v_proj.weight":
                                enc_layer.self_attn.v_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.o_proj.weight":
                                enc_layer.self_attn.o_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.gate_proj.weight":
                                enc_layer.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight":
                                enc_layer.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight":
                                enc_layer.mlp.down_proj.weight.copy_(tensor)

                        # Layers 16..31 -> Recurrent Transition Decoder
                        else:
                            dec_idx = layer_idx - 16
                            dec_layer = model.decoder_layers[dec_idx]
                            if sub_key == "input_layernorm.weight":
                                dec_layer.input_layernorm.weight.copy_(tensor)
                            elif sub_key == "post_attention_layernorm.weight":
                                dec_layer.post_attention_layernorm.weight.copy_(tensor)
                            elif sub_key == "self_attn.q_proj.weight":
                                dec_layer.self_attn.q_proj.weight.copy_(tensor)
                                dec_layer.cross_attn.q_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.k_proj.weight":
                                dec_layer.self_attn.k_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.v_proj.weight":
                                dec_layer.self_attn.v_proj.weight.copy_(tensor)
                            elif sub_key == "self_attn.o_proj.weight":
                                dec_layer.self_attn.o_proj.weight.copy_(tensor)
                                dec_layer.cross_attn.o_proj.weight.zero_()  # Zero init for clean residual bypass
                            elif sub_key == "mlp.gate_proj.weight":
                                dec_layer.mlp.gate_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.up_proj.weight":
                                dec_layer.mlp.up_proj.weight.copy_(tensor)
                            elif sub_key == "mlp.down_proj.weight":
                                dec_layer.mlp.down_proj.weight.copy_(tensor)

                del tensor
        gc.collect()

    print(f"\n>>> Organ Transplant Complete! Total parameters streamed: {total_streamed:,}")
    model.eval()

    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "Python is a popular programming language that",
    ]

    print("\n" + "="*60)
    print(">>> 8-BILLION PARAMETER RECURRENT TEXT GENERATION")
    print("="*60)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(8):
                logits = model(generated_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_tok], dim=1)

        result = tokenizer.decode(generated_ids[0].cpu())
        print(f"\n[PROMPT]: '{prompt}'")
        print(f"[RECURRENT 8B RESULT]: '{result}'")

if __name__ == "__main__":
    run_llama3_transplant()
