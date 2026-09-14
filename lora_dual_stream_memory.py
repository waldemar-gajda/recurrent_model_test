import os
import re
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

SYMBOL_REGEX = re.compile(r'([A-Za-z0-9]+-[A-Za-z0-9]+|\b-?\d+\.?\d*\b)')

class LoRALinear(nn.Module):
    """
    Surgical Low-Rank Adapter for Linear layers:
    y = W_frozen(x) + (x @ A.T @ B.T) * (alpha / r)
    Leaves base weights 100% frozen. B is initialized to 0.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(r, in_features, device=device, dtype=dtype) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

class AttentionLoRADualStreamMemory(nn.Module):
    """
    Dual-Stream Cognitive Architecture with Surgical Attention LoRA:
    - 100% of MLP layers (knowledge encyclopedia) are FROZEN.
    - LoRA (r=8) is attached ONLY to W_q and W_v in attention heads.
    - Tor 1: 8 Scene Slots (Semantic Context).
    - Tor 2: Phonological Loop / Symbol Register (Raw Numbers & Codes).
    """
    def __init__(self, model_dir: str, num_slots: int = 8, num_think: int = 4, r: int = 8, device: str = "mps"):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.num_think = num_think
        self.r = r
        self.model_dir = os.path.expanduser(model_dir)

        print(f">>> Ładowanie modelu z: {self.model_dir} na {device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)

        # 1. Zamrożenie 100% wag bazowych
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. Chirurgiczne wpięcie LoRA TYLKO w W_q i W_v warstw uwagi (MLP nienaruszone!)
        self.lora_params = []
        for layer in self.model.model.layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=r)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=r)
            self.lora_params.extend([
                layer.self_attn.q_proj.lora_A, layer.self_attn.q_proj.lora_B,
                layer.self_attn.v_proj.lora_A, layer.self_attn.v_proj.lora_B
            ])

        d_model = self.model.config.hidden_size

        # 3. Sloty sceny i tokeny myślenia
        self.empty_slots = nn.Parameter(torch.randn(1, num_slots, d_model, device=device, dtype=torch.bfloat16) * 0.02)
        self.think_tokens = nn.Parameter(torch.randn(1, num_think, d_model, device=device, dtype=torch.bfloat16) * 0.02)

        self.slot_params = [self.empty_slots, self.think_tokens]

        lora_count = sum(p.numel() for p in self.lora_params)
        slot_count = sum(p.numel() for p in self.slot_params)
        print(f">>> Zainstalowano chirurgiczną LoRA!")
        print(f"    Parametry LoRA uwagi (W_q, W_v): {lora_count / 1e6:.2f}M ({lora_count * 2 / 1024:.1f} KB)")
        print(f"    Parametry Slotów i Myślenia:     {slot_count / 1e3:.1f}k ({slot_count * 2 / 1024:.1f} KB)")
        print(f"    Warstwy wiedzy (MLP):            W 100% ZAMROŻONE I NIENARUSZONE.\n")

    def get_trainable_parameters(self):
        return self.lora_params + self.slot_params

    def extract_symbols(self, passage: str) -> List[str]:
        return list(dict.fromkeys(SYMBOL_REGEX.findall(passage)))

    def encode_scene_and_symbols(self, passage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        is_llama = "llama" in self.model_dir.lower()
        if is_llama:
            p_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nZapamiętaj poniższy dokument:\n{passage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            p_text = f"<|im_start|>user\nZapamiętaj poniższy dokument:\n{passage}<|im_end|>\n<|im_start|>assistant\n"

        p_ids = self.tokenizer.encode(p_text, return_tensors="pt").to(self.device)
        p_embeds = self.model.model.embed_tokens(p_ids)

        slots = self.empty_slots.expand(p_embeds.shape[0], -1, -1)
        out = self.model.model(inputs_embeds=torch.cat([p_embeds, slots], dim=1))
        scene_vectors = out.last_hidden_state[:, -self.num_slots:, :]

        symbols = self.extract_symbols(passage)
        sym_str = " ".join(symbols) if symbols else ""
        sym_ids = self.tokenizer.encode(sym_str, add_special_tokens=False, return_tensors="pt").to(self.device)
        if sym_ids.shape[1] > 24:
            sym_ids = sym_ids[:, :24]
        elif sym_ids.shape[1] == 0:
            sym_ids = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)

        return scene_vectors, sym_ids

    def forward_qa(
        self,
        scene_vectors: torch.Tensor,
        sym_ids: torch.Tensor,
        question: str,
        target_answer: str
    ) -> torch.Tensor:
        is_llama = "llama" in self.model_dir.lower()
        if is_llama:
            q_prefix = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            target_str = f"{target_answer}<|eot_id|>"
        else:
            q_prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            target_str = f"{target_answer}<|im_end|>"

        q_ids = self.tokenizer.encode(q_prefix, return_tensors="pt").to(self.device)
        t_ids = self.tokenizer.encode(target_str, add_special_tokens=False, return_tensors="pt").to(self.device)

        sym_embeds = self.model.model.embed_tokens(sym_ids)
        q_embeds = self.model.model.embed_tokens(q_ids)
        t_embeds = self.model.model.embed_tokens(t_ids)
        th_embeds = self.think_tokens.expand(q_embeds.shape[0], -1, -1)

        combined = torch.cat([scene_vectors, sym_embeds, q_embeds, th_embeds, t_embeds], dim=1)
        logits = self.model(inputs_embeds=combined).logits

        prefix_len = self.num_slots + sym_ids.shape[1] + q_ids.shape[1] + self.num_think
        t_len = t_ids.shape[1]
        start_idx = prefix_len - 1
        end_idx = prefix_len + t_len - 1

        shift_logits = logits[:, start_idx:end_idx, :].contiguous()
        shift_labels = t_ids.contiguous()

        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, self.model.config.vocab_size).float(),
            shift_labels.view(-1)
        )
        return loss

    def answer(
        self,
        scene_vectors: torch.Tensor,
        sym_ids: torch.Tensor,
        question: str,
        max_new_tokens: int = 25
    ) -> str:
        self.model.eval()
        with torch.no_grad():
            is_llama = "llama" in self.model_dir.lower()
            if is_llama:
                q_text = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                stop_ids = [self.tokenizer.eos_token_id, 128001, 128009]
            else:
                q_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                stop_ids = [self.tokenizer.eos_token_id, 0, 1, 2]

            q_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)
            sym_embeds = self.model.model.embed_tokens(sym_ids)
            q_embeds = self.model.model.embed_tokens(q_ids)
            th_embeds = self.think_tokens.expand(q_embeds.shape[0], -1, -1)

            cur_embeds = torch.cat([scene_vectors, sym_embeds, q_embeds, th_embeds], dim=1)

            gen_tokens = []
            for _ in range(max_new_tokens):
                out = self.model(inputs_embeds=cur_embeds)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = next_tok.item()
                if tok_id in stop_ids:
                    break
                gen_tokens.append(tok_id)
                next_embed = self.model.model.embed_tokens(next_tok)
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

            return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
