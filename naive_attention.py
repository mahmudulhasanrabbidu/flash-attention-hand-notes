import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def memory_footprint_bytes(batch_size: int, num_heads: int, seq_len: int, dtype: torch.dtype = torch.float16) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    # P: (batch_size, num_heads, seq_len, seq_len)
    num_elements = batch_size * num_heads * seq_len * seq_len
    return num_elements * element_size


def naive_scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, softmax_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N_q, D = q.shape
    _, _, N_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # pairwise attention scores S = (Q @ K^T) * scale
    # (B, H, N_q, N_k)
    s = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.tril(torch.ones((N_q, N_k), device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))

    #softmax along key dimension (last dimension)
    p = F.softmax(s.float(), dim=-1).to(q.dtype)

    # (B, H, N_q, D)
    o = torch.matmul(p, v)

    return o, p


def naive_attention_forward_and_backward(q: torch.Tensor, k: torch.Tensor,v: torch.Tensor, do: torch.Tensor, causal: bool = True, oftmax_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_clone = q.detach().clone().requires_grad_(True)
    k_clone = k.detach().clone().requires_grad_(True)
    v_clone = v.detach().clone().requires_grad_(True)

    o, _ = naive_scaled_dot_product_attention(
        q_clone, k_clone, v_clone, causal=causal, softmax_scale=softmax_scale
    )
    o.backward(do)

    return o.detach(), q_clone.grad, k_clone.grad, v_clone.grad


if __name__ == "__main__":
    b, h, seq, dim = 2, 4, 1024, 64
    bytes_p = memory_footprint_bytes(b, h, seq)
    print(f"Memory to materialize P for seq_len={seq}: {bytes_p / (1024 * 1024):.2f} MB")
    
    seq_long = 16384
    bytes_p_long = memory_footprint_bytes(b, h, seq_long)
    print(f"Memory to materialize P for seq_len={seq_long}: {bytes_p_long / (1024 * 1024):.2f} MB ({bytes_p_long / (1024**3):.2f} GB)")
