import os
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

MODEL_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm2-135m-instruct")
SYMBOL_REGEX = re.compile(r'([A-Za-z0-9]+-[A-Za-z0-9]+|\b-?\d+\.?\d*\b)')

class DualStreamBridge(nn.Module):
    """
    Bridge combining:
    - K=8 Scene Slots (Continuous semantic memory, 9.2 KB)
    - P=4 Think/Routing Tokens (Induction workspace)
    - Adapter MLP
    """
    def __init__(self, d_model: int = 576, num_slots: int = 8, num_think: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_think = num_think

        self.empty_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.think_tokens = nn.Parameter(torch.randn(1, num_think, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.gate = nn.Parameter(torch.tensor([-1.0]))

    def get_empty_slots(self, batch_size: int = 1) -> torch.Tensor:
        return self.empty_slots.expand(batch_size, -1, -1)

    def get_think_tokens(self, batch_size: int = 1) -> torch.Tensor:
        return self.think_tokens.expand(batch_size, -1, -1)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return x + g * self.mlp(self.norm(x))

class SmolLMDualStreamMemory(nn.Module):
    """
    Dual-Stream Cognitive Memory:
    - Stream 1: Continuous Scene Slots (K=8, 9.2 KB) -> Captures semantics, agents, relations
    - Stream 2: Phonological Loop / Discrete Symbol Register (<= 16 token IDs, 32 bytes) -> Exact numbers & codes
    Total memory footprint: ~9.23 KB strictly O(1) RAM.
    """
    def __init__(self, num_slots: int = 8, num_think: int = 4, max_sym_len: int = 16, device: str = "mps"):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.num_think = num_think
        self.max_sym_len = max_sym_len

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.bridge = DualStreamBridge(
            d_model=self.model.config.hidden_size,
            num_slots=num_slots,
            num_think=num_think
        ).to(device).to(torch.bfloat16)

    def extract_symbols(self, passage: str) -> List[str]:
        return list(dict.fromkeys(SYMBOL_REGEX.findall(passage)))

    def encode_scene_and_symbols(self, passage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Turn 1:
        1. Encodes passage into 8 scene slots.
        2. Extracts literal symbols into phonological token buffer (<= 16 tokens).
        """
        p_text = f"<|im_start|>user\nZapamiętaj poniższy dokument:\n{passage}<|im_end|>\n<|im_start|>assistant\n"
        p_ids = self.tokenizer.encode(p_text, return_tensors="pt").to(self.device)
        p_embeds = self.model.model.embed_tokens(p_ids)

        slots = self.bridge.get_empty_slots(batch_size=p_embeds.shape[0])
        out = self.model.model(inputs_embeds=torch.cat([p_embeds, slots], dim=1))
        scene_vectors = self.bridge.adapt(out.last_hidden_state[:, -self.num_slots:, :])

        # Ekstrakcja symboli do pętli fonologicznej
        symbols = self.extract_symbols(passage)
        sym_str = " ".join(symbols) if symbols else ""
        sym_ids = self.tokenizer.encode(sym_str, add_special_tokens=False, return_tensors="pt").to(self.device)
        if sym_ids.shape[1] > self.max_sym_len:
            sym_ids = sym_ids[:, :self.max_sym_len]
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
        q_prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        target_str = f"{target_answer}<|im_end|>"

        q_ids = self.tokenizer.encode(q_prefix, return_tensors="pt").to(self.device)
        t_ids = self.tokenizer.encode(target_str, add_special_tokens=False, return_tensors="pt").to(self.device)

        sym_embeds = self.model.model.embed_tokens(sym_ids)
        q_embeds = self.model.model.embed_tokens(q_ids)
        t_embeds = self.model.model.embed_tokens(t_ids)
        th_embeds = self.bridge.get_think_tokens(batch_size=q_embeds.shape[0])

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
            q_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            q_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)

            sym_embeds = self.model.model.embed_tokens(sym_ids)
            q_embeds = self.model.model.embed_tokens(q_ids)
            th_embeds = self.bridge.get_think_tokens(batch_size=q_embeds.shape[0])

            cur_embeds = torch.cat([scene_vectors, sym_embeds, q_embeds, th_embeds], dim=1)

            gen_tokens = []
            for _ in range(max_new_tokens):
                out = self.model(inputs_embeds=cur_embeds)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                tok_id = next_tok.item()
                if tok_id in [self.tokenizer.eos_token_id, 0, 1, 2]:
                    break
                gen_tokens.append(tok_id)
                next_embed = self.model.model.embed_tokens(next_tok)
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)

            return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
