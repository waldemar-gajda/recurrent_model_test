import os
import re
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

from dataset_memory_recall import generate_dataset

MODEL_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama-3-8b-instruct")
SAVE_PATH = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama3_dual_stream.pt")
SYMBOL_REGEX = re.compile(r'([A-Za-z0-9]+-[A-Za-z0-9]+|\b-?\d+\.?\d*\b)')

class Llama3DualStreamBridge(nn.Module):
    """
    Bridge for Llama 3 8B (d_model = 4096):
    - K=8 Scene Slots (High-capacity semantic memory, 65.5 KB)
    - P=4 Think Tokens (Deliberation workspace)
    - Adapter MLP (4096 -> 8192 -> 4096)
    """
    def __init__(self, d_model: int = 4096, num_slots: int = 8, num_think: int = 4):
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

class Llama3DualStreamMemory(nn.Module):
    """
    Dual-Stream Cognitive Architecture on Meta Llama-3-8B-Instruct:
    - Tor 1: Continuous Scene Slots (K=8, 65.5 KB RAM) -> Semantics, relations, scene facts
    - Tor 2: Phonological Loop / Symbol Register (<= 16 tokens, 32 B) -> Raw numbers, keys, codes
    Total memory footprint: 65.5 KB strictly O(1) RAM.
    """
    def __init__(self, num_slots: int = 8, num_think: int = 4, max_sym_len: int = 16, device: str = "mps"):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.num_think = num_think
        self.max_sym_len = max_sym_len

        print(f">>> Ładowanie Llama-3-8B-Instruct na {device.upper()} (bfloat16)...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)

        # 100% wag Llama 3 zamrożone
        for p in self.model.parameters():
            p.requires_grad = False

        self.bridge = Llama3DualStreamBridge(
            d_model=self.model.config.hidden_size, # 4096
            num_slots=num_slots,
            num_think=num_think
        ).to(device).to(torch.bfloat16)

        trainable = sum(p.numel() for p in self.bridge.parameters() if p.requires_grad)
        print(f">>> Llama 3 załadowana! Trenowalne parametry adaptera: {trainable / 1e6:.2f}M ({trainable * 2 / (1024**2):.1f} MB)\n")

    def extract_symbols(self, passage: str) -> List[str]:
        return list(dict.fromkeys(SYMBOL_REGEX.findall(passage)))

    def encode_scene_and_symbols(self, passage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        p_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nZapamiętaj poniższy dokument:\n{passage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        p_ids = self.tokenizer.encode(p_text, return_tensors="pt").to(self.device)
        p_embeds = self.model.model.embed_tokens(p_ids)

        slots = self.bridge.get_empty_slots(batch_size=p_embeds.shape[0])
        out = self.model.model(inputs_embeds=torch.cat([p_embeds, slots], dim=1))
        scene_vectors = self.bridge.adapt(out.last_hidden_state[:, -self.num_slots:, :])

        # Pętla fonologiczna
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
        q_prefix = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        target_str = f"{target_answer}<|eot_id|>"

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
            q_text = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            q_ids = self.tokenizer.encode(q_text, return_tensors="pt").to(self.device)

            sym_embeds = self.model.model.embed_tokens(sym_ids)
            q_embeds = self.model.model.embed_tokens(q_ids)
            th_embeds = self.bridge.get_think_tokens(batch_size=q_embeds.shape[0])

            cur_embeds = torch.cat([scene_vectors, sym_embeds, q_embeds, th_embeds], dim=1)

            stop_ids = [self.tokenizer.eos_token_id, 128001, 128009]
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

def train_and_eval_llama3(num_epochs: int = 10):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 72)
    print(f"  DUAL-STREAM COGNITIVE MEMORY NA META LLAMA-3-8B-INSTRUCT ({device.upper()})")
    print("  Tor 1: 8 Slotów Sceny (65.5 KB) -> Pełny kontekst semantyczny")
    print("  Tor 2: Rejestr Symboli (32 B)   -> Pętla fonologiczna cyfr i kodów")
    print("  Razem: Stałe 65.5 KB w RAM (O(1)) | 100% wag bazowych zamrożone")
    print("=" * 72)

    memory = Llama3DualStreamMemory(num_slots=8, num_think=4, max_sym_len=16, device=device)
    train_data = generate_dataset(num_samples=200, base_seed=42)
    test_data = generate_dataset(num_samples=15, base_seed=1234)

    optimizer = torch.optim.AdamW(memory.bridge.parameters(), lr=1.5e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    best_loss = 999.0
    print(f"\n>>> Start treningu ({num_epochs} epok na 200 próbkach z Cosine Annealing)...")

    for epoch in range(num_epochs):
        t_start = time.time()
        total_loss = 0.0

        for p, q, a in train_data:
            optimizer.zero_grad()
            scene_v, sym_ids = memory.encode_scene_and_symbols(p)
            loss = memory.forward_qa(scene_v, sym_ids, q, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.bridge.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - t_start
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  [Epoka {epoch + 1:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} | LR: {cur_lr:.5f} | Czas: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(memory.bridge.state_dict(), SAVE_PATH)

    print(f"\n>>> Zapisano optymalne wagi do {SAVE_PATH}. Rozpoczynam ewaluację na 15 nowych scenach...\n")

    # Ewaluacja na niewidzianych danych testowych
    memory.bridge.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    hits = 0
    total = len(test_data)

    for i, (p, q, expected) in enumerate(test_data, 1):
        scene_v, sym_ids = memory.encode_scene_and_symbols(p)
        gen = memory.answer(scene_v, sym_ids, q)

        is_hit = expected.lower() in gen.lower() or gen.lower() in expected.lower()
        if is_hit: hits += 1

        status = "✓ TRAF" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/{total:02d}] {status}")
        print(f"  Scena:      {p}")
        print(f"  Pytanie:    {q}")
        print(f"  Oczekiwana: {expected}")
        print(f"  Odpowiedź:  {gen}\n")

    acc = (hits / total) * 100
    print("=" * 72)
    print(f"  WYNIK KOŃCOWY LLAMA-3-8B DUAL-STREAM: {hits}/{total} ({acc:.1f}% dokładności)")
    print("=" * 72)

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    train_and_eval_llama3(num_epochs=ep)
