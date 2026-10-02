import math
import torch
import triton
from typing import Optional

from .fwd_kernel import _attn_fwd_kernel
from .bwd_preprocess import _attn_bwd_preprocess
from .bwd_kernel import _attn_bwd_dq, _attn_bwd_dk_dv


class FlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal: bool, softmax_scale: float):
        batch_size, num_heads, seq_len, head_dim = Q.shape
        head_dim_k = K.shape[-1]
        head_dim_v = V.shape[-1]
        assert head_dim == head_dim_k and head_dim_k == head_dim_v, "Head dimensions must match"

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # Dynamic launch grid based on autotuned BLOCK_SIZE_Q
        grid = lambda args: (
            triton.cdiv(seq_len, args["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
            1,
        )

        # M stores logsumexp for backward pass
        M = torch.empty(
            (batch_size, num_heads, seq_len), device=Q.device, dtype=torch.float32
        )

        _attn_fwd_kernel[grid](
            Q=Q, K=K, V=V, softmax_scale=softmax_scale,
            M=M, O=O,
            stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1), stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0), stride_K_head=K.stride(1), stride_K_seq=K.stride(2), stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0), stride_V_head=V.stride(1), stride_V_seq=V.stride(2), stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0), stride_O_head=O.stride(1), stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
            BATCH_SIZE=batch_size,
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.softmax_scale = softmax_scale
        ctx.head_dim = head_dim
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous(), "dO must be contiguous"
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        batch_size, num_heads, seq_len = Q.shape[:3]
        stage = 3 if ctx.causal else 1
        D = torch.empty_like(M)

        # autotuned Preprocessing: D = rowsum(dO * O)
        preprocess_grid = lambda args: (triton.cdiv(seq_len, args["BLOCK_SIZE_Q"]), batch_size * num_heads)
        _attn_bwd_preprocess[preprocess_grid](O=O, dO=dO, D=D, SEQ_LEN=seq_len, HEAD_DIM=ctx.head_dim)

        # autotuned dK and dV: Grid 0 parallelized over BLOCK_KV
        grid_dk_dv = lambda args: (triton.cdiv(seq_len, args["BLOCK_KV"]), 1, batch_size * num_heads)
        _attn_bwd_dk_dv[grid_dk_dv](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV,
            M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=ctx.head_dim,
            STAGE=stage,
        )

        #autotuned dQ: Grid 0 parallelized over BLOCK_Q
        grid_dq = lambda args: (triton.cdiv(seq_len, args["BLOCK_Q"]), 1, batch_size * num_heads)
        _attn_bwd_dq[grid_dq](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
            dO=dO, dQ=dQ, dK=dK, dV=dV,
            M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=ctx.head_dim,
            STAGE=stage,
        )

        return dQ, dK, dV, None, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, softmax_scale: Optional[float] = None) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return FlashAttentionFunction.apply(q, k, v, causal, softmax_scale)
