import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

class GatedRecurrentBridge(nn.Module):
    """
    Non-linear Gated Recurrent Bridge:
    u_t = e_t + Bridge(H_{t-1})
    where Bridge(H) = sigmoid(gate) * W2(SiLU(W1(Norm(H))))
    
    The learnable vector gate starts initialized with negative bias (e.g. -4.0, ~0.018),
    guaranteeing that at initialization the bridge adds negligible perturbation to LLaMA-3,
    while providing active non-zero gradients across all dimensions.
    """
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, init_gate_bias: float = -4.0):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or (d_model * 2)
        
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=False)
        
        # Per-dimension gating parameter
        self.gate = nn.Parameter(torch.full((d_model,), init_gate_bias))
        
        # Initialize weights with standard Xavier normal
        nn.init.xavier_normal_(self.fc1.weight, gain=0.1)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d_model) or (B, T, d_model)
        Returns: (B, d_model) or (B, T, d_model)
        """
        g = torch.sigmoid(self.gate)
        h_norm = self.norm(h)
        projected = self.fc2(self.act(self.fc1(h_norm)))
        return g * projected
