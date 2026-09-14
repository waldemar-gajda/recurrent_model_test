r"""
All-Token Recurrence (ATR) Neural Architecture in pure PyTorch.

Components:
- Phase 01 (Encoder): Parallel causal encoding of prompt tokens x_1, ..., x_T
  producing token representations e_1, ..., e_T and encoder prefix KV memory M_{\le T}^E.
- Phase 02 (All-Token Recurrence): Recurrent state update where prompt and response
  share transition dynamics. Each step combines token embedding with previous recurrent
  state (H_{T-1} \oplus e_T \to H_T), passed through L_D decoder layers with layerwise
  cached KV and prefix memory access.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AllTokenRecurrentConfig:
    r"""Configuration for All-Token Recurrence model."""
    vocab_size: int = 256
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 128
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_weights: bool = False


class CausalSelfAttention(nn.Module):
    r"""
    Multi-head causal self-attention supporting both full-sequence parallel forward
    and step-by-step autoregressive execution with KV cache.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        x: (B, T, d_model)
        past_kv: Optional tuple of (cached_k, cached_v) with shape (B, num_heads, T_past, d_head)
        Returns:
            out: (B, T, d_model)
            (k, v): updated key and value representations (B, num_heads, T_total, d_head)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            past_k = past_k.to(device=x.device, dtype=x.dtype)
            past_v = past_v.to(device=x.device, dtype=x.dtype)
            if past_k.size(0) == 1 and B > 1:
                past_k = past_k.expand(B, -1, -1, -1)
                past_v = past_v.expand(B, -1, -1, -1)
            assert past_k.size(0) == B, (
                f"past_kv batch size {past_k.size(0)} does not match current batch size {B}"
            )
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        T_total = k.size(2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, num_heads, T, T_total)

        if causal and T > 1:
            # Causal mask for sequence length T against total keys
            # When past_kv is present, query at index t corresponds to key index (T_past + t)
            T_past = T_total - T
            mask = torch.triu(torch.full((T, T_total), float('-inf'), device=x.device, dtype=scores.dtype), diagonal=T_past + 1)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights).to(v.dtype)

        attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, T, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(attn_out)

        return out, (k, v)


class PrefixCrossAttention(nn.Module):
    r"""
    Cross-attention mechanism accessing encoder prefix KV memory.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        x: query hidden state (B, 1, d_model)
        prefix_k: encoder prefix keys (B, num_heads, T_prefix, d_head)
        prefix_v: encoder prefix values (B, num_heads, T_prefix, d_head)
        """
        B, T_q, _ = x.shape
        if prefix_k.size(0) == 1 and B > 1:
            prefix_k = prefix_k.expand(B, -1, -1, -1)
            prefix_v = prefix_v.expand(B, -1, -1, -1)
        assert prefix_k.size(0) == B, (
            f"prefix_k batch size {prefix_k.size(0)} does not match query batch size {B}"
        )
        prefix_k = prefix_k.to(device=x.device, dtype=x.dtype)
        prefix_v = prefix_v.to(device=x.device, dtype=x.dtype)

        q = self.q_proj(x).view(B, T_q, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, prefix_k.transpose(-1, -2)) * self.scale  # (B, num_heads, T_q, T_prefix)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights).to(prefix_v.dtype)

        attn_out = torch.matmul(attn_weights, prefix_v)  # (B, num_heads, T_q, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    r"""Two-layer MLP with activation and dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    r"""
    Causal encoder layer producing token representations and prefix KV cache.
    """
    def __init__(self, config: AllTokenRecurrentConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_out, new_kv = self.self_attn(norm_x, past_kv=past_kv, causal=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class CausalEncoder(nn.Module):
    r"""
    Phase 01: Causal Encoder E_\theta.
    Performs parallel causal encoding of observed prompt tokens x_1, ..., x_T
    producing token representations e_1, ..., e_T and prefix KV memory M_{\le T}^E.
    """
    def __init__(self, config: AllTokenRecurrentConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_prefix_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        r"""
        x: (B, T, d_model) token embeddings.
        past_prefix_kv: Optional list of past (k, v) per encoder layer.
        Returns:
            e: token representations (B, T, d_model)
            prefix_kv_memory: list of (k, v) per layer (B, num_heads, T_total, d_head)
        """
        if past_prefix_kv is not None:
            assert len(past_prefix_kv) == len(self.layers), (
                f"past_prefix_kv has {len(past_prefix_kv)} layers, but encoder has {len(self.layers)} layers"
            )
        prefix_kv_memory = []
        h = x
        for i, layer in enumerate(self.layers):
            layer_past = past_prefix_kv[i] if past_prefix_kv is not None else None
            h, layer_kv = layer(h, past_kv=layer_past)
            prefix_kv_memory.append(layer_kv)
        e = self.norm(h)
        return e, prefix_kv_memory


class RecurrentDecoderLayer(nn.Module):
    r"""
    Single layer of the Recurrent Transition Decoder D_\phi.
    Features:
    - Layerwise cached KV self-attention across past decoder steps.
    - Prefix memory cross-attention to encoder prefix KV memory.
    - Feed-forward transformation.
    """
    def __init__(self, config: AllTokenRecurrentConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.cross_attn = PrefixCrossAttention(config.d_model, config.num_heads, config.dropout)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_dec_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        prefix_kv: Tuple[torch.Tensor, torch.Tensor],
        step_idx: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        x: decoder transition candidate (B, 1, d_model)
        past_dec_kv: cached decoder KV from prior recurrent steps
        prefix_kv: encoder prefix KV memory (k, v) for this layer
        step_idx: current step index
        """
        # 1. Decoder Self-Attention with layerwise cached KV
        norm1_x = self.norm1(x)
        attn_self, new_dec_kv = self.self_attn(norm1_x, past_kv=past_dec_kv, causal=False)
        x = x + attn_self

        # 2. Prefix memory access: slice prefix memory causally up to current step
        prefix_k, prefix_v = prefix_kv
        curr_prefix_len = min(step_idx + 1, prefix_k.size(2))
        k_slice = prefix_k[:, :, :curr_prefix_len, :]
        v_slice = prefix_v[:, :, :curr_prefix_len, :]

        norm2_x = self.norm2(x)
        attn_cross = self.cross_attn(norm2_x, k_slice, v_slice)
        x = x + attn_cross

        # 3. Feed-Forward
        x = x + self.mlp(self.norm3(x))

        return x, new_dec_kv


class RecurrentTransitionDecoder(nn.Module):
    r"""
    Phase 02: Recurrent Transition Decoder D_\phi.
    Combines token representation e_t with previous recurrent state H_{t-1}:
        H_{t-1} \oplus e_t \to H_t
        and passes it through L_D decoder layers with layerwise cached KV and prefix memory access.
    """
    def __init__(self, config: AllTokenRecurrentConfig):
        super().__init__()
        self.config = config
        self.h0 = nn.Parameter(torch.zeros(1, config.d_model))
        nn.init.normal_(self.h0, std=0.02)

        self.transition_proj = nn.Linear(2 * config.d_model, config.d_model)
        self.transition_norm = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList([RecurrentDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""Returns initial recurrent state H_0 expanded to (batch_size, d_model)."""
        if device is None:
            device = self.h0.device
        if dtype is None:
            dtype = self.h0.dtype
        return self.h0.to(device=device, dtype=dtype).expand(batch_size, -1)

    def step(
        self,
        h_prev: torch.Tensor,
        e_t: torch.Tensor,
        decoder_kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
        prefix_kv_memory: List[Tuple[torch.Tensor, torch.Tensor]],
        step_idx: int,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        r"""
        Executes one transition step:
            H_{t-1} \oplus e_t \to H_t
        h_prev: (B, d_model)
        e_t: (B, d_model)
        decoder_kv_cache: list of (k, v) per decoder layer
        prefix_kv_memory: list of (k, v) per encoder layer
        step_idx: current time step
        """
        assert prefix_kv_memory is not None and len(prefix_kv_memory) > 0, (
            "prefix_kv_memory must be a non-empty list of layer prefix memories"
        )
        if decoder_kv_cache is not None:
            assert len(decoder_kv_cache) == len(self.layers), (
                f"decoder_kv_cache has {len(decoder_kv_cache)} layers, but decoder has {len(self.layers)} layers"
            )
        else:
            decoder_kv_cache = [None] * len(self.layers)

        # H_{t-1} \oplus e_t -> u_t
        combined = torch.cat([h_prev, e_t], dim=-1)
        u = self.transition_norm(self.transition_proj(combined))
        x = u.unsqueeze(1)  # (B, 1, d_model)

        new_decoder_kv_cache = []
        num_enc_layers = len(prefix_kv_memory)
        for l_idx, layer in enumerate(self.layers):
            pref_kv = prefix_kv_memory[l_idx % num_enc_layers]
            past_dec = decoder_kv_cache[l_idx]
            x, new_dec = layer(x, past_dec, pref_kv, step_idx)
            new_decoder_kv_cache.append(new_dec)

        h_t = self.norm(x.squeeze(1))
        return h_t, new_decoder_kv_cache


class AllTokenRecurrentModel(nn.Module):
    r"""
    All-Token Recurrence Neural Architecture.
    Unifies:
    - Token embedding
    - Phase 01: Causal Encoder E_\theta
    - Phase 02: Recurrent Transition Decoder D_\phi
    - Next-token LM projection head
    """
    def __init__(self, config: Optional[AllTokenRecurrentConfig] = None):
        super().__init__()
        self.config = config or AllTokenRecurrentConfig()

        assert self.config.vocab_size > 0, f"vocab_size must be > 0, got {self.config.vocab_size}"
        assert self.config.d_model > 0, f"d_model must be > 0, got {self.config.d_model}"
        assert self.config.num_heads > 0, f"num_heads must be > 0, got {self.config.num_heads}"
        assert self.config.d_ff > 0, f"d_ff must be > 0, got {self.config.d_ff}"
        assert self.config.d_model % self.config.num_heads == 0, (
            f"d_model ({self.config.d_model}) must be divisible by num_heads ({self.config.num_heads})"
        )
        assert self.config.num_encoder_layers > 0, f"num_encoder_layers must be > 0, got {self.config.num_encoder_layers}"
        assert self.config.num_decoder_layers > 0, f"num_decoder_layers must be > 0, got {self.config.num_decoder_layers}"
        assert self.config.max_seq_len > 0, f"max_seq_len must be > 0, got {self.config.max_seq_len}"
        assert 0.0 <= self.config.dropout < 1.0, f"dropout must be in [0, 1), got {self.config.dropout}"

        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.encoder = CausalEncoder(self.config)
        self.decoder = RecurrentTransitionDecoder(self.config)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        if self.config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        r"""Initializes model weights with standard normal deviations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def init_recurrent_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""Initializes H_0 recurrent state for a batch."""
        return self.decoder.init_state(batch_size, device=device, dtype=dtype)

    def init_cache(self, batch_size: int = 1) -> Tuple[None, List[None]]:
        r"""Initializes empty (prefix_memory, decoder_cache)."""
        return None, [None] * len(self.decoder.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        r"""
        Forward pass over sequence of tokens.
        input_ids: (batch_size, seq_len) or (seq_len,)
        initial_state: Optional initial recurrent state H_0 (batch_size, d_model) or (1, d_model)
        return_state: If True, returns (logits, final_recurrent_state, (prefix_kv_memory, decoder_kv_cache))
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        device = self.tok_embeddings.weight.device
        dtype = self.tok_embeddings.weight.dtype

        if input_ids.ndim == 0:
            input_ids = input_ids.view(1, 1)
        elif input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.ndim == 2, f"input_ids must have shape (batch_size, seq_len), got {input_ids.shape}"

        if input_ids.device != device:
            input_ids = input_ids.to(device=device)

        B, T = input_ids.shape

        # Phase 01 (Encoder): Parallel causal encoding of prompt tokens
        x_embed = self.tok_embeddings(input_ids)
        e, prefix_kv_memory = self.encoder(x_embed)

        if initial_state is not None:
            if initial_state.ndim == 1:
                initial_state = initial_state.unsqueeze(0)
            if initial_state.size(0) == 1 and B > 1:
                initial_state = initial_state.expand(B, -1)
            assert initial_state.size(0) == B, (
                f"initial_state batch size {initial_state.size(0)} does not match input_ids batch size {B}"
            )
            assert initial_state.size(1) == self.config.d_model, (
                f"initial_state dimension {initial_state.size(1)} does not match d_model {self.config.d_model}"
            )
            h = initial_state.to(device=device, dtype=dtype)
        else:
            h = self.init_recurrent_state(B, device=device, dtype=dtype)

        if T == 0:
            empty_logits = self.lm_head(x_embed)
            if return_state:
                return empty_logits, h, (prefix_kv_memory, [None] * len(self.decoder.layers))
            return empty_logits

        # Phase 02 (All-Token Recurrence): Recurrent state update across all observed tokens
        decoder_kv_cache = [None] * len(self.decoder.layers)

        logits_list = []
        for t in range(T):
            e_t = e[:, t, :]
            h, decoder_kv_cache = self.decoder.step(
                h_prev=h,
                e_t=e_t,
                decoder_kv_cache=decoder_kv_cache,
                prefix_kv_memory=prefix_kv_memory,
                step_idx=t,
            )
            logits_t = self.lm_head(h)
            logits_list.append(logits_t)

        logits = torch.stack(logits_list, dim=1)

        if return_state:
            return logits, h, (prefix_kv_memory, decoder_kv_cache)
        return logits

    def step(
        self,
        token_id: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]] = None,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]]:
        r"""
        Step-by-step state transition:
        Computes the recurrent transition for a single token step, updating H_T and KV caches.

        token_id: (B,), (B, 1), or scalar () token tensor
        recurrent_state: Optional (B, d_model) or (1, d_model) previous state H_{t-1}
        kv_cache: Tuple of (prefix_kv_memory, decoder_kv_cache)
        step_idx: Optional explicit step index
        Returns:
            logits: (B, 1, vocab_size) next-token logits
            new_state: (B, d_model) updated recurrent state H_t
            new_cache: updated (prefix_kv_memory, decoder_kv_cache)
        """
        device = self.tok_embeddings.weight.device
        dtype = self.tok_embeddings.weight.dtype

        if token_id.ndim == 0:
            token_id = token_id.view(1, 1)
        elif token_id.ndim == 1:
            token_id = token_id.unsqueeze(1)
        assert token_id.ndim == 2 and token_id.size(1) == 1, (
            f"step() expects single-token input of shape (B, 1), (B,), or (), got {token_id.shape}"
        )
        if token_id.device != device:
            token_id = token_id.to(device=device)

        B, _ = token_id.shape

        if recurrent_state is None:
            recurrent_state = self.init_recurrent_state(B, device=device, dtype=dtype)
        else:
            if recurrent_state.ndim == 1:
                recurrent_state = recurrent_state.unsqueeze(0)
            if recurrent_state.size(0) == 1 and B > 1:
                recurrent_state = recurrent_state.expand(B, -1)
            assert recurrent_state.size(0) == B, (
                f"recurrent_state batch size {recurrent_state.size(0)} does not match token_id batch size {B}"
            )
            assert recurrent_state.size(1) == self.config.d_model, (
                f"recurrent_state dimension {recurrent_state.size(1)} does not match d_model {self.config.d_model}"
            )
            recurrent_state = recurrent_state.to(device=device, dtype=dtype)

        # Align kv_cache device, dtype, and broadcast batch size if needed
        prefix_kv_memory = None
        decoder_kv_cache = [None] * len(self.decoder.layers)
        if kv_cache is not None:
            assert isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2, (
                "kv_cache must be a tuple of (prefix_kv_memory, decoder_kv_cache)"
            )
            raw_prefix_kv, raw_decoder_kv = kv_cache

            def _align_cache_list(cache_list, name):
                if cache_list is None:
                    return None
                aligned = []
                for item in cache_list:
                    if item is None:
                        aligned.append(None)
                    else:
                        k, v = item
                        k = k.to(device=device, dtype=dtype)
                        v = v.to(device=device, dtype=dtype)
                        if k.size(0) == 1 and B > 1:
                            k = k.expand(B, -1, -1, -1)
                            v = v.expand(B, -1, -1, -1)
                        assert k.size(0) == B, (
                            f"{name} batch size {k.size(0)} does not match token batch size {B}"
                        )
                        aligned.append((k, v))
                return aligned

            prefix_kv_memory = _align_cache_list(raw_prefix_kv, "prefix_kv_memory")
            decoder_kv_cache = _align_cache_list(raw_decoder_kv, "decoder_kv_cache") or ([None] * len(self.decoder.layers))

        if step_idx is not None:
            assert step_idx >= 0, f"step_idx must be >= 0, got {step_idx}"
            curr_step = step_idx
        elif decoder_kv_cache is not None and len(decoder_kv_cache) > 0 and decoder_kv_cache[0] is not None:
            curr_step = decoder_kv_cache[0][0].size(2)
        elif prefix_kv_memory is not None and len(prefix_kv_memory) > 0 and prefix_kv_memory[0] is not None:
            curr_step = prefix_kv_memory[0][0].size(2)
        else:
            curr_step = 0

        # Encode single token with past prefix memory
        x_embed = self.tok_embeddings(token_id)
        e, new_prefix_kv_memory = self.encoder(x_embed, past_prefix_kv=prefix_kv_memory)
        e_t = e.squeeze(1)

        # Recurrent decoder transition
        new_h, new_decoder_kv_cache = self.decoder.step(
            h_prev=recurrent_state,
            e_t=e_t,
            decoder_kv_cache=decoder_kv_cache,
            prefix_kv_memory=new_prefix_kv_memory,
            step_idx=curr_step,
        )

        logits = self.lm_head(new_h).unsqueeze(1)
        new_cache = (new_prefix_kv_memory, new_decoder_kv_cache)
        return logits, new_h, new_cache

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Autoregressively generates continuation tokens given a prompt.
        Supports 2D (B, T), 1D (T,), and 0D () prompt tensors, including empty prompts.
        """
        assert temperature >= 0.0, f"temperature must be non-negative, got {temperature}"
        assert prompt_ids.ndim <= 2, f"prompt_ids must have at most 2 dimensions, got shape {prompt_ids.shape}"
        self.eval()

        device = self.tok_embeddings.weight.device
        dtype = self.tok_embeddings.weight.dtype
        if prompt_ids.device != device:
            prompt_ids = prompt_ids.to(device=device)

        orig_ndim = prompt_ids.ndim
        if prompt_ids.ndim == 0:
            prompt_ids = prompt_ids.view(1, 1)
        elif prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if max_new_tokens <= 0:
            if orig_ndim == 0:
                return prompt_ids.squeeze()
            elif orig_ndim == 1:
                return prompt_ids.squeeze(0)
            return prompt_ids

        B, T = prompt_ids.shape

        if T == 0:
            # Empty prompt: generate starting directly from H_0 or initial_state
            if initial_state is not None:
                if initial_state.ndim == 1:
                    initial_state = initial_state.unsqueeze(0)
                if initial_state.size(0) == 1 and B > 1:
                    initial_state = initial_state.expand(B, -1)
                h = initial_state.to(device=device, dtype=dtype)
            else:
                h = self.init_recurrent_state(B, device=device, dtype=dtype)
            logits = self.lm_head(h)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = [curr_token]
            kv_cache = None
            remaining_tokens = max_new_tokens - 1
        else:
            logits, h, kv_cache = self.forward(prompt_ids, initial_state=initial_state, return_state=True)
            if temperature > 0:
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = [curr_token]
            remaining_tokens = max_new_tokens - 1

        for _ in range(remaining_tokens):
            logits_step, h, kv_cache = self.step(curr_token, h, kv_cache)
            if temperature > 0:
                probs = F.softmax(logits_step.squeeze(1) / temperature, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(logits_step.squeeze(1), dim=-1, keepdim=True)
            generated.append(curr_token)

        out = torch.cat([prompt_ids] + generated, dim=1)
        if orig_ndim == 0:
            return out.squeeze()
        elif orig_ndim == 1:
            return out.squeeze(0)
        return out
