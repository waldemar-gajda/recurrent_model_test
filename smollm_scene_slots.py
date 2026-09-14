import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List

MODEL_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm2-135m-instruct")

class SlotBridge(nn.Module):
    """
    Dynamic Scene Slots Adapter with Think Tokens (Pause/Deliberation workspace):
    Holds M empty slot tokens, P think tokens, and a lightweight non-linear adapter.
    """
    def __init__(self, d_model: int = 576, num_slots: int = 4, num_think_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_think_tokens = num_think_tokens
        self.empty_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        # Learnable think tokens (intermediate deliberation workspace)
        self.think_tokens = nn.Parameter(torch.randn(1, num_think_tokens, d_model) * 0.02)
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

    def adapt(self, scene_vectors: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return scene_vectors + g * self.mlp(self.norm(scene_vectors))

class SmolLMSceneMemory(nn.Module):
    """
    SmolLM2-135M with Dynamic Scene Slots & Think Tokens:
    Compresses arbitrary scenes/documents into M=4 natural scene vectors (4.6 KB)
    and uses P=4 deliberation think tokens to resolve questions accurately.
    """
    def __init__(self, num_slots: int = 4, num_think_tokens: int = 4, device: str = "mps"):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.num_think_tokens = num_think_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device)

        # Freeze 100% of SmolLM2 base weights
        for p in self.model.parameters():
            p.requires_grad = False

        self.bridge = SlotBridge(
            d_model=self.model.config.hidden_size,
            num_slots=num_slots,
            num_think_tokens=num_think_tokens
        ).to(device).to(torch.bfloat16)

    def encode_scene(self, passage: str) -> torch.Tensor:
        """
        Processes passage through SmolLM2 and returns M=4 scene vectors.
        """
        p_text = f"<|im_start|>user\nZapamiętaj poniższy dokument:\n{passage}<|im_end|>\n<|im_start|>assistant\n"
        p_ids = self.tokenizer.encode(p_text, return_tensors="pt").to(self.device)
        p_embeds = self.model.model.embed_tokens(p_ids)

        slots = self.bridge.get_empty_slots(batch_size=p_embeds.shape[0])
        combined = torch.cat([p_embeds, slots], dim=1)

        out = self.model.model(inputs_embeds=combined)
        raw_scene = out.last_hidden_state[:, -self.num_slots:, :]
        adapted_scene = self.bridge.adapt(raw_scene)
        return adapted_scene

    def forward_qa(
        self,
        scene_vectors: torch.Tensor,
        question: str,
        target_answer: str,
        use_think: bool = True
    ) -> torch.Tensor:
        """
        Computes training loss on target answer given scene vectors + question (+ optional think tokens).
        """
        q_prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        target_str = f"{target_answer}<|im_end|>"

        q_ids = self.tokenizer.encode(q_prefix, return_tensors="pt").to(self.device)
        t_ids = self.tokenizer.encode(target_str, add_special_tokens=False, return_tensors="pt").to(self.device)
        
        q_embeds = self.model.model.embed_tokens(q_ids)
        t_embeds = self.model.model.embed_tokens(t_ids)

        if use_think:
            th_embeds = self.bridge.get_think_tokens(batch_size=q_embeds.shape[0])
            combined_turn2 = torch.cat([scene_vectors, q_embeds, th_embeds, t_embeds], dim=1)
            prefix_len = self.num_slots + q_ids.shape[1] + self.num_think_tokens
        else:
            combined_turn2 = torch.cat([scene_vectors, q_embeds, t_embeds], dim=1)
            prefix_len = self.num_slots + q_ids.shape[1]

        logits = self.model(inputs_embeds=combined_turn2).logits

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

    def answer_from_scene(
        self,
        scene_vectors: torch.Tensor,
        question: str,
        max_new_tokens: int = 25,
        use_think: bool = True
    ) -> str:
        """
        Generates text answer using ONLY scene_vectors + question (+ optional think tokens).
        Zero document text in context.
        """
        self.model.eval()
        with torch.no_grad():
            q_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            q_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)
            q_embeds = self.model.model.embed_tokens(q_ids)

            if use_think:
                th_embeds = self.bridge.get_think_tokens(batch_size=q_embeds.shape[0])
                cur_embeds = torch.cat([scene_vectors, q_embeds, th_embeds], dim=1)
            else:
                cur_embeds = torch.cat([scene_vectors, q_embeds], dim=1)

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
