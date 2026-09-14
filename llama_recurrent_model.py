import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from recurrent_bridge import GatedRecurrentBridge
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaRecurrentConfig:
    vocab_size: int = 128256
    d_model: int = 4096
    intermediate_size: int = 14336
    num_heads: int = 32
    num_kv_heads: int = 8
    num_encoder_layers: int = 16
    num_decoder_layers: int = 16
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_f32.to(input_dtype)


def precompute_rope_cache(dim: int, end: int, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device="cpu")[: (dim // 2)].float() / dim))
    t = torch.arange(end, device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # expand to (1, 1, end, dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq: (B, H, T, D), cos: (1, 1, T, D)
    T = xq.size(2)
    cos_t = cos[:, :, :T, :].to(dtype=xq.dtype, device=xq.device)
    sin_t = sin[:, :, :T, :].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos_t) + (rotate_half(xq) * sin_t)
    xk_out = (xk * cos_t) + (rotate_half(xk) * sin_t)
    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key/Value heads for Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaCausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaRecurrentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        # GQA repeat
        k_rep = repeat_kv(k, self.num_rep)
        v_rep = repeat_kv(v, self.num_rep)

        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=(causal and T > 1 and past_kv is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), new_kv


class LlamaPrefixCrossAttention(nn.Module):
    def __init__(self, config: LlamaRecurrentConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
    ) -> torch.Tensor:
        B, T_q, _ = x.shape
        q = self.q_proj(x).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_rep = repeat_kv(prefix_k, self.num_rep)
        v_rep = repeat_kv(prefix_v, self.num_rep)

        out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.o_proj(out)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, config: LlamaRecurrentConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = LlamaCausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config.d_model, config.intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_kv=None):
        h, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv, causal=True)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class LlamaRecurrentDecoderLayer(nn.Module):
    def __init__(self, config: LlamaRecurrentConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = LlamaCausalSelfAttention(config)
        self.cross_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.cross_attn = LlamaPrefixCrossAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config.d_model, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_dec_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        step_idx: int,
        cross_gate: Optional[torch.Tensor] = None,
        is_memory_bank: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Decoder Self Attention with past KV
        h, new_dec_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_dec_kv, causal=False)
        x = x + h

        # 2. Prefix or Memory-Bank cross-attention
        if prefix_kv is not None:
            pref_k, pref_v = prefix_kv
            if is_memory_bank:
                h_cross = self.cross_attn(self.cross_layernorm(x), pref_k, pref_v)
            else:
                curr_len = min(step_idx + 1, pref_k.size(2))
                h_cross = self.cross_attn(self.cross_layernorm(x), pref_k[:, :, :curr_len, :], pref_v[:, :, :curr_len, :])

            if cross_gate is not None:
                h_cross = cross_gate * h_cross
            x = x + h_cross

        # 3. SwiGLU MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_dec_kv


class LlamaAllTokenRecurrentModel(nn.Module):
    """
    All-Token Recurrence architecture implemented natively for Llama-3 (8B):
    Phase 01: Causal Encoder (16 layers) with prefix memory
    Phase 02: Recurrent Transition Decoder (16 layers) with state H_T and layerwise KV caching
    """
    def __init__(self, config: LlamaRecurrentConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Encoder (first 16 layers of Llama-3)
        self.encoder_layers = nn.ModuleList([LlamaEncoderLayer(config) for _ in range(config.num_encoder_layers)])

        # Recurrent state transition bridge: Residual-Add
        self.h0 = nn.Parameter(torch.zeros(1, config.d_model))
        self.h_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.recurrent_bridge = None
        self.memory_bank = None

        # Decoder (last 16 layers of Llama-3)
        self.decoder_layers = nn.ModuleList([LlamaRecurrentDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.decoder_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute RoPE freqs
        head_dim = config.d_model // config.num_heads
        cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def enable_memory_bank(self, num_slots: int = 8, init_gate_bias: float = -2.0):
        from recurrent_memory_bank import RecurrentMemoryBank
        self.memory_bank = RecurrentMemoryBank(
            d_model=self.config.d_model,
            num_slots=num_slots,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.d_model // self.config.num_heads,
            num_layers=self.config.num_decoder_layers,
            init_gate_bias=init_gate_bias,
        )
        self.memory_bank.to(self.embed_tokens.weight.device).to(self.embed_tokens.weight.dtype)
        return self.memory_bank

    def encode_to_memory_bank(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes passage into K-Slot memory bank tensors (mem_k, mem_v)."""
        B, T = input_ids.shape
        device = input_ids.device
        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        x = self.embed_tokens(input_ids)
        for enc_layer in self.encoder_layers:
            x, _ = enc_layer(x, cos, sin)

        mem_k, mem_v = self.memory_bank(x)
        return mem_k, mem_v

    def enable_gated_bridge(self, hidden_dim: Optional[int] = None, init_gate_bias: float = -4.0):
        self.recurrent_bridge = GatedRecurrentBridge(self.config.d_model, hidden_dim, init_gate_bias)
        self.recurrent_bridge.to(self.embed_tokens.weight.device).to(self.embed_tokens.weight.dtype)
        return self.recurrent_bridge

    def apply_bridge(self, h: torch.Tensor) -> torch.Tensor:
        if self.recurrent_bridge is not None:
            return self.recurrent_bridge(h)
        return self.h_proj(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        initial_h: Optional[torch.Tensor] = None,
        memory_bank_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B, T = input_ids.shape
        device = input_ids.device
        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        # Phase 01: Parallel causal encoder
        x = self.embed_tokens(input_ids)
        prefix_kv_memory = []
        for enc_layer in self.encoder_layers:
            x, layer_kv = enc_layer(x, cos, sin)
            prefix_kv_memory.append(layer_kv)
        e = x  # Direct unperturbed residual stream

        # Phase 02: All-Token Recurrence
        h = initial_h if initial_h is not None else self.h0.expand(B, -1)
        decoder_kv_cache = [None] * len(self.decoder_layers)
        logits_list = []

        for t in range(T):
            e_t = e[:, t, :]
            # Residual-Add bypass: guarantees 100% equivalence at initialization
            u = (e_t + self.apply_bridge(h)).unsqueeze(1)

            new_dec_cache = []
            cur_x = u
            for l_idx, dec_layer in enumerate(self.decoder_layers):
                if memory_bank_kv is not None:
                    pref_kv = memory_bank_kv
                    is_mem = True
                    c_gate = self.memory_bank.get_layer_gate(l_idx) if self.memory_bank is not None else None
                else:
                    pref_kv = prefix_kv_memory[l_idx % len(prefix_kv_memory)]
                    is_mem = False
                    c_gate = None

                past_dec = decoder_kv_cache[l_idx]
                cur_x, new_kv = dec_layer(
                    cur_x, cos[:, :, t : t + 1, :], sin[:, :, t : t + 1, :],
                    past_dec, pref_kv, step_idx=t, cross_gate=c_gate, is_memory_bank=is_mem
                )
                new_dec_cache.append(new_kv)

            decoder_kv_cache = new_dec_cache
            h = self.decoder_norm(cur_x.squeeze(1))
            logits_t = self.lm_head(h)
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=1)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        initial_h: Optional[torch.Tensor] = None,
        memory_bank_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Forward pass across prompt that also returns KV caches and final recurrent state."""
        B, T = input_ids.shape
        device = input_ids.device
        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        # Phase 01: Parallel causal encoder
        x = self.embed_tokens(input_ids)
        prefix_kv_memory = []
        for enc_layer in self.encoder_layers:
            x, layer_kv = enc_layer(x, cos, sin)
            prefix_kv_memory.append(layer_kv)
        e = x

        # Phase 02: All-Token Recurrence
        h = initial_h if initial_h is not None else self.h0.expand(B, -1)
        decoder_kv_cache = [None] * len(self.decoder_layers)
        logits_list = []

        for t in range(T):
            e_t = e[:, t, :]
            u = (e_t + self.apply_bridge(h)).unsqueeze(1)

            new_dec_cache = []
            cur_x = u
            for l_idx, dec_layer in enumerate(self.decoder_layers):
                if memory_bank_kv is not None:
                    pref_kv = memory_bank_kv
                    is_mem = True
                    c_gate = self.memory_bank.get_layer_gate(l_idx) if self.memory_bank is not None else None
                else:
                    pref_kv = prefix_kv_memory[l_idx % len(prefix_kv_memory)]
                    is_mem = False
                    c_gate = None

                past_dec = decoder_kv_cache[l_idx]
                cur_x, new_kv = dec_layer(
                    cur_x, cos[:, :, t : t + 1, :], sin[:, :, t : t + 1, :],
                    past_dec, pref_kv, step_idx=t, cross_gate=c_gate, is_memory_bank=is_mem
                )
                new_dec_cache.append(new_kv)

            decoder_kv_cache = new_dec_cache
            h = self.decoder_norm(cur_x.squeeze(1))
            logits_t = self.lm_head(h)
            logits_list.append(logits_t)

        logits = torch.stack(logits_list, dim=1)
        return logits, prefix_kv_memory, decoder_kv_cache, h

    def step(
        self,
        tok_id: torch.Tensor,
        pos: int,
        enc_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        dec_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        h: torch.Tensor,
        memory_bank_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Fast O(1) auto-regressive generation step with layerwise KV caching and memory bank support.
        tok_id: (B, 1)
        pos: current token position integer
        """
        device = tok_id.device
        cos_t = self.rope_cos[:, :, pos : pos + 1, :].to(device)
        sin_t = self.rope_sin[:, :, pos : pos + 1, :].to(device)

        # 1. Encoder single step
        x = self.embed_tokens(tok_id)
        new_enc = []
        for l_idx, enc_layer in enumerate(self.encoder_layers):
            past = enc_cache[l_idx] if enc_cache is not None else None
            x, nkv = enc_layer(x, cos_t, sin_t, past_kv=past)
            new_enc.append(nkv)

        e_t = x.squeeze(1)

        # 2. Residual-Add recurrent transition
        u = (e_t + self.apply_bridge(h)).unsqueeze(1)

        # 3. Decoder single step
        new_dec = []
        cur_x = u
        for l_idx, dec_layer in enumerate(self.decoder_layers):
            past = dec_cache[l_idx] if dec_cache is not None else None
            if memory_bank_kv is not None:
                pref_kv = memory_bank_kv
                is_mem = True
                c_gate = self.memory_bank.get_layer_gate(l_idx) if self.memory_bank is not None else None
            else:
                pref_kv = new_enc[l_idx % len(new_enc)]
                is_mem = False
                c_gate = None

            cur_x, nkv = dec_layer(
                cur_x, cos_t, sin_t, past_dec_kv=past,
                prefix_kv=pref_kv, step_idx=pos, cross_gate=c_gate, is_memory_bank=is_mem
            )
            new_dec.append(nkv)

        # 4. Decoder norm & LM head
        h_new = self.decoder_norm(cur_x.squeeze(1))
        logits = self.lm_head(h_new)
        return logits, new_enc, new_dec, h_new
