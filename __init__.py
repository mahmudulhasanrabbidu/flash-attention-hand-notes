"""
FlashAttention-2 Triton Pipeline
================================
Modular, educational, and high-performance implementation of FlashAttention-2.
"""

from .naive_attention import (
    naive_scaled_dot_product_attention,
    naive_attention_forward_and_backward,
    memory_footprint_bytes,
)
from .online_softmax import (
    naive_softmax,
    safe_softmax,
    online_softmax_1d,
    tiled_online_attention_simulation,
)
from .fwd_kernel import _attn_fwd_kernel
from .bwd_preprocess import _attn_bwd_preprocess
from .bwd_kernel import _attn_bwd_dq, _attn_bwd_dk_dv
from .flash_attention import FlashAttentionFunction, flash_attention
from .flash_mha import FlashMHA

__all__ = [
    "naive_scaled_dot_product_attention",
    "naive_attention_forward_and_backward",
    "memory_footprint_bytes",
    "naive_softmax",
    "safe_softmax",
    "online_softmax_1d",
    "tiled_online_attention_simulation",
    "_attn_fwd_kernel",
    "_attn_bwd_preprocess",
    "_attn_bwd_dq",
    "_attn_bwd_dk_dv",
    "FlashAttentionFunction",
    "flash_attention",
    "FlashMHA",
]
