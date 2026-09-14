import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Optional

MODEL_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/smollm2-135m-instruct")

class EvolutionaryBridge(nn.Module):
    """
    Bridge supporting:
    - K=8 Scene Slots (High-resolution sensory memory)
    - P=4 Draft Tokens (Hypothesis phase)
    - P=4 Verify Tokens (Evolutionary refinement phase)
    - Lightweight non-linear MLP adapter
    """
    def __init__(self, d_model: int = 576, num_slots: int = 8, num_draft: int = 4, num_verify: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_draft = num_draft
        self.num_verify = num_verify

        self.empty_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.draft_tokens = nn.Parameter(torch.randn(1, num_draft, d_model) * 0.02)
        self.verify_tokens = nn.Parameter(torch.randn(1, num_verify, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.gate = nn.Parameter(torch.tensor([-1.0]))

    def get_empty_slots(self, batch_size: int = 1) -> torch.Tensor:
        return self.empty_slots.expand(batch_size, -1, -1)

    def get_draft_tokens(self, batch_size: int = 1) -> torch.Tensor:
        return self.draft_tokens.expand(batch_size, -1, -1)

    def get_verify_tokens(self, batch_size: int = 1) -> torch.Tensor:
        return self.verify_tokens.expand(batch_size, -1, -1)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return x + g * self.mlp(self.norm(x))

class SmolLMEvolutionaryMemory(nn.Module):
    """
    Unified Evolutionary Memory on SmolLM2-135M:
    - Micro-Evolution: 2-Pass Deliberation (Draft -> Verify) within each turn.
    - Macro-Evolution: State Consolidation (S_t -> S_{t+1}) across conversational turns.
    """
    def __init__(self, num_slots: int = 8, num_draft: int = 4, num_verify: int = 4, device: str = "mps"):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.num_draft = num_draft
        self.num_verify = num_verify

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device)

        # Freeze 100% of base model weights
        for p in self.model.parameters():
            p.requires_grad = False

        self.bridge = EvolutionaryBridge(
            d_model=self.model.config.hidden_size,
            num_slots=num_slots,
            num_draft=num_draft,
            num_verify=num_verify
        ).to(device).to(torch.bfloat16)

    def encode_scene(self, passage: str) -> torch.Tensor:
        """
        Turn 1: Compresses passage into S_0 (8 scene slots).
        """
        p_text = f"<|im_start|>user\nZapamiętaj poniższy dokument:\n{passage}<|im_end|>\n<|im_start|>assistant\n"
        p_ids = self.tokenizer.encode(p_text, return_tensors="pt").to(self.device)
        p_embeds = self.model.model.embed_tokens(p_ids)

        slots = self.bridge.get_empty_slots(batch_size=p_embeds.shape[0])
        combined = torch.cat([p_embeds, slots], dim=1)

        out = self.model.model(inputs_embeds=combined)
        raw_scene = out.last_hidden_state[:, -self.num_slots:, :]
        return self.bridge.adapt(raw_scene)

    def forward_turn(
        self,
        scene_vectors: torch.Tensor,
        question: str,
        target_answer: str
    ) -> torch.Tensor:
        """
        Executes 2-Pass Micro-Evolution:
        Pass 1: [Scene] + [Question] + [Draft Tokens] -> Hypothesis vectors
        Pass 2: [Scene] + [Question] + [Hypothesis] + [Verify Tokens] + [Answer] -> CrossEntropy Loss
        """
        q_prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        target_str = f"{target_answer}<|im_end|>"

        q_ids = self.tokenizer.encode(q_prefix, return_tensors="pt").to(self.device)
        t_ids = self.tokenizer.encode(target_str, add_special_tokens=False, return_tensors="pt").to(self.device)

        q_embeds = self.model.model.embed_tokens(q_ids)
        t_embeds = self.model.model.embed_tokens(t_ids)

        # Pass 1: Hipoteza (Draft)
        draft_toks = self.bridge.get_draft_tokens(batch_size=q_embeds.shape[0])
        pass1_in = torch.cat([scene_vectors, q_embeds, draft_toks], dim=1)
        out1 = self.model.model(inputs_embeds=pass1_in)
        draft_vectors = out1.last_hidden_state[:, -self.num_draft:, :]

        # Pass 2: Ewolucyjna weryfikacja (Verify)
        verify_toks = self.bridge.get_verify_tokens(batch_size=q_embeds.shape[0])
        pass2_in = torch.cat([scene_vectors, q_embeds, draft_vectors, verify_toks, t_embeds], dim=1)
        logits = self.model(inputs_embeds=pass2_in).logits

        prefix_len = self.num_slots + q_ids.shape[1] + self.num_draft + self.num_verify
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

    def consolidate_memory(
        self,
        scene_vectors: torch.Tensor,
        question: str,
        answer: str
    ) -> torch.Tensor:
        """
        Macro-Evolution: Consolidates previous memory state with the turn's Q&A experience:
        S_{t+1} = Transformer([S_t, Q, A, empty_slots])
        """
        q_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
        qa_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)
        qa_embeds = self.model.model.embed_tokens(qa_ids)

        slots = self.bridge.get_empty_slots(batch_size=qa_embeds.shape[0])
        # Detach previous state to prevent gradient explosion across multiple turns
        combined = torch.cat([scene_vectors.detach(), qa_embeds, slots], dim=1)

        out = self.model.model(inputs_embeds=combined)
        raw_next_scene = out.last_hidden_state[:, -self.num_slots:, :]
        return self.bridge.adapt(raw_next_scene)

    def answer_turn(
        self,
        scene_vectors: torch.Tensor,
        question: str,
        max_new_tokens: int = 25
    ) -> str:
        """
        Inference with 2-Pass Micro-Evolution:
        Pass 1 generates hypothesis vectors.
        Pass 2 autoregressively generates verified answer.
        """
        self.model.eval()
        with torch.no_grad():
            q_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            q_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)
            q_embeds = self.model.model.embed_tokens(q_ids)

            # Pass 1: Hipoteza
            draft_toks = self.bridge.get_draft_tokens(batch_size=q_embeds.shape[0])
            pass1_in = torch.cat([scene_vectors, q_embeds, draft_toks], dim=1)
            out1 = self.model.model(inputs_embeds=pass1_in)
            draft_vectors = out1.last_hidden_state[:, -self.num_draft:, :]

            # Pass 2: Weryfikacja i generacja
            verify_toks = self.bridge.get_verify_tokens(batch_size=q_embeds.shape[0])
            cur_embeds = torch.cat([scene_vectors, q_embeds, draft_vectors, verify_toks], dim=1)

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
