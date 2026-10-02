import torch
import triton
import triton.language as tl


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_Q": bs}, num_warps=nw, num_stages=1)
        for bs in [32, 64, 128]
        for nw in [2, 4, 8]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_preprocess(O, dO, D, SEQ_LEN, BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)

    offs_dim = tl.arange(0, HEAD_DIM)

    # base pointer offset for current batch and head
    batch_head_offset = index_batch_head * HEAD_DIM * SEQ_LEN

    # load BLOCK_SIZE_Q rows of O and dO
    O_block = tl.load(O + batch_head_offset + offs_q[:, None] * HEAD_DIM + offs_dim[None, :])
    dO_block = tl.load(dO + batch_head_offset + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]).to(tl.float32)

    # D_i = sum(dO_i * O_i) across head dimension
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)

    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
