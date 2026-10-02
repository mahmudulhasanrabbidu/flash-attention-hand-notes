import math
import torch
import torch.nn as nn
from typing import Optional

from .flash_attention import flash_attention


class FlashMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = True, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        B, N, C = x.shape
        is_causal = self.causal if causal is None else causal

        # Shape: (B, N, C) -> (B, N, H, D) -> (B, H, N, D)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o = flash_attention(q, k, v, causal=is_causal, softmax_scale=self.softmax_scale)

        # (B, H, N, D) -> (B, N, H, D) -> (B, N, C)
        o = o.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(o)
