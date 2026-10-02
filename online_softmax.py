import math
import torch
from typing import Tuple


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def safe_softmax(x: torch.Tensor) -> torch.Tensor:
    m = torch.max(x, dim=-1, keepdim=True).values
    exp_x = torch.exp(x - m)
    l = torch.sum(exp_x, dim=-1, keepdim=True)
    return exp_x / l


def online_softmax_1d(x: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
    n = x.numel()
    m_i = float("-inf")
    l_i = 0.0

    for i in range(0, n, chunk_size):
        chunk = x[i : i + chunk_size]
        m_chunk = torch.max(chunk).item()

        m_new = max(m_i, m_chunk)
        alpha = math.exp(m_i - m_new) if m_i != float("-inf") else 0.0

        l_chunk = torch.sum(torch.exp(chunk - m_new)).item()
        l_new = l_i * alpha + l_chunk

        m_i = m_new
        l_i = l_new
        
    return torch.exp(x - m_i) / l_i


def tiled_online_attention_simulation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_kv: int = 64, softmax_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    N_q, D = q.shape
    N_k, _ = k.shape

    m_i = torch.full((N_q, 1), float("-inf"), device=q.device)
    l_i = torch.zeros((N_q, 1), device=q.device)
    o_i = torch.zeros((N_q, D), device=q.device)

    for start_k in range(0, N_k, block_kv):
        end_k = min(start_k + block_kv, N_k)
        k_chunk = k[start_k:end_k]
        v_chunk = v[start_k:end_k]

        # s_chunk = Q @ K_chunk^T * scale
        s_chunk = torch.matmul(q, k_chunk.T) * softmax_scale

        m_chunk = torch.max(s_chunk, dim=-1, keepdim=True).values
        m_new = torch.maximum(m_i, m_chunk)

        alpha = torch.exp(m_i - m_new)
        alpha = torch.where(torch.isneginf(m_i), torch.zeros_like(alpha), alpha)

        p_chunk = torch.exp(s_chunk - m_new)

        l_chunk = torch.sum(p_chunk, dim=-1, keepdim=True)
        l_new = l_i * alpha + l_chunk

        # o_new = O_old * alpha + P_chunk @ V_chunk
        o_i = o_i * alpha + torch.matmul(p_chunk, v_chunk)

        m_i = m_new
        l_i = l_new

    o_final = o_i / l_i
    logsumexp = m_i + torch.log(l_i)
    return o_final, logsumexp.squeeze(-1)


if __name__ == "__main__":
    x = torch.randn(128)
    p_safe = safe_softmax(x)
    p_online = online_softmax_1d(x, chunk_size=16)
    print("Online Softmax max difference:", torch.max(torch.abs(p_safe - p_online)).item())

    q = torch.randn(32, 64)
    k = torch.randn(128, 64)
    v = torch.randn(128, 64)
    scale = 1.0 / math.sqrt(64)

    s_ref = torch.matmul(q, k.T) * scale
    p_ref = torch.softmax(s_ref, dim=-1)
    o_ref = torch.matmul(p_ref, v)

    o_online, lse = tiled_online_attention_simulation(q, k, v, block_kv=16, softmax_scale=scale)
    print("Tiled Attention max difference:", torch.max(torch.abs(o_ref - o_online)).item())
