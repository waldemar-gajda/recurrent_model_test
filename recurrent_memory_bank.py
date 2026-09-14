import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

class RecurrentMemoryBank(nn.Module):
    """
    K-Slot Recurrent Memory Bank:
    Compresses variable-length document token representations (B, T, d_model)
    into M memory slot vectors (B, M, d_model) via learned cross-attention pooling.
    
    Produces shared Key and Value tensors (B, num_kv_heads, M, head_dim)
    for Cross-Attention in all Decoder layers, maintaining strict O(1) RAM (16 KB total).
    """
    def __init__(
        self,
        d_model: int = 4096,
        num_slots: int = 8,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_layers: int = 16,
        init_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        # Learned latent query vectors for the M memory slots
        self.latent_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.query_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_model)

        # Cross-Attention Pooler: Latent Queries attend to Encoder Tokens
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward refinement for memory slots (SwiGLU)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_down = nn.Linear(d_model * 2, d_model, bias=False)

        # Projection from slots to Decoder Key and Value tensors
        self.slot_norm = RMSNorm(d_model)
        self.to_mem_k = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.to_mem_v = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)

        # Layerwise learnable gates for decoder cross-attention (one scalar per decoder layer)
        self.cross_gates = nn.Parameter(torch.full((num_layers,), init_gate_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.out_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.ffn_gate.weight, gain=0.1)
        nn.init.xavier_normal_(self.ffn_up.weight, gain=0.1)
        nn.init.xavier_normal_(self.ffn_down.weight, gain=0.1)
        nn.init.xavier_normal_(self.to_mem_k.weight, gain=0.1)
        nn.init.xavier_normal_(self.to_mem_v.weight, gain=0.1)

    def forward(self, encoder_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        encoder_tokens: (B, T, d_model)
        Returns: (mem_k, mem_v)
          mem_k: (B, num_kv_heads, num_slots, head_dim)
          mem_v: (B, num_kv_heads, num_slots, head_dim)
        """
        B, T, _ = encoder_tokens.shape
        queries = self.query_norm(self.latent_queries.expand(B, -1, -1)) # (B, M, d_model)
        kv = self.kv_norm(encoder_tokens) # (B, T, d_model)

        pool_dim = self.num_heads * self.head_dim
        q = self.q_proj(queries).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention: latent queries attend to encoder tokens
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        pooled = torch.matmul(attn, v) # (B, num_heads, M, head_dim)
        pooled = pooled.transpose(1, 2).contiguous().view(B, self.num_slots, pool_dim)
        slots = queries + self.out_proj(pooled)

        # FFN refinement (SwiGLU)
        ffn_in = self.ffn_norm(slots)
        gate = F.silu(self.ffn_gate(ffn_in))
        up = self.ffn_up(ffn_in)
        slots = slots + self.ffn_down(gate * up)

        # Project slots to shared Decoder Keys and Values
        s_norm = self.slot_norm(slots)
        mem_k = self.to_mem_k(s_norm).view(B, self.num_slots, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        mem_v = self.to_mem_v(s_norm).view(B, self.num_slots, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        return mem_k, mem_v

    def get_layer_gate(self, layer_idx: int) -> torch.Tensor:
        """Returns sigmoid scalar gate for decoder layer layer_idx."""
        return torch.sigmoid(self.cross_gates[layer_idx])
