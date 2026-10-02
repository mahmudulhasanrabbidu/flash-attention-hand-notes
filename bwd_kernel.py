import torch
import triton
import triton.language as tl


@triton.autotune(
    [
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_warps=4, num_stages=4),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale,
    dO, dQ, dK, dV,
    M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Computes dQ by fixing a Q block and iterating across KV blocks.
    Grid: (cdiv(SEQ_LEN, BLOCK_Q), 1, BATCH_SIZE * NUM_HEADS)
    """
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_q = tl.program_id(0)

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Load Q block and dO block into SRAM
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    M_block = tl.load(M + offs_q)[:, None]
    Di = tl.load(D + offs_q)

    offs_kv_base = tl.arange(0, BLOCK_KV)

    if STAGE == 3:
        # full unmasked blocks: all keys strictly before start_q
        hi_unmasked = (start_q // BLOCK_KV) * BLOCK_KV
        # total active blocks: up to start_q + BLOCK_Q
        hi_total = (start_q + BLOCK_Q + BLOCK_KV - 1) // BLOCK_KV * BLOCK_KV
        hi_total = tl.minimum(hi_total, SEQ_LEN)
    else:
        hi_unmasked = SEQ_LEN
        hi_total = SEQ_LEN

    kT_ptrs = K + offs_kv_base[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv_base[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    # full unmasked blocks
    for start_kv in range(0, hi_unmasked, BLOCK_KV):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = (P_block * (dP_block - Di[:, None])).to(tl.float16)

        # Factor out softmax_scale to epilogue
        dQ_block += tl.dot(dS_block, tl.trans(K_T_block))

        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    # transition blocks (apply causal mask only where needed)
    if STAGE == 3:
        for start_kv in range(hi_unmasked, hi_total, BLOCK_KV):
            K_T_block = tl.load(kT_ptrs)
            V_T_block = tl.load(vT_ptrs)

            QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
            P_block = tl.math.exp(QK_block - M_block)

            current_offs_kv = start_kv + offs_kv_base
            mask_block = offs_q[:, None] >= current_offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
            dS_block = (P_block * (dP_block - Di[:, None])).to(tl.float16)

            dQ_block += tl.dot(dS_block, tl.trans(K_T_block))

            kT_ptrs += BLOCK_KV * stride_seq
            vT_ptrs += BLOCK_KV * stride_seq

    dQ_block = dQ_block * softmax_scale
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.autotune(
    [
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=4, num_stages=4),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, softmax_scale,
    dO, dQ, dK, dV,
    M, D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Computes dK and dV by fixing a KV block and iterating across Q blocks.
    Grid: (cdiv(SEQ_LEN, BLOCK_KV), 1, BATCH_SIZE * NUM_HEADS)
    """
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # Load K and V blocks (stay in SRAM throughout the loop over Q)
    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    offs_q_base = tl.arange(0, BLOCK_Q)

    # bound Calculations for Two-Stage Causal Loop
    if STAGE == 3:
        # skip query blocks strictly before start_kv
        lo_q = (start_kv // BLOCK_Q) * BLOCK_Q
        # boundary transition ends when all queries in the block exceed start_kv + BLOCK_KV
        hi_transition = ((start_kv + BLOCK_KV + BLOCK_Q - 1) // BLOCK_Q) * BLOCK_Q
        hi_transition = tl.minimum(hi_transition, SEQ_LEN)
    else:
        lo_q = 0
        hi_transition = 0

    curr_q = lo_q
    qT_ptrs = Q + (curr_q + offs_q_base[None, :]) * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + (curr_q + offs_q_base[:, None]) * stride_seq + offs_dim[None, :] * stride_dim

    # for causal
    if STAGE == 3:
        for _ in range(lo_q, hi_transition, BLOCK_Q):
            qT_block = tl.load(qT_ptrs)
            offs_q = curr_q + offs_q_base
            m = tl.load(M + offs_q)

            QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
            P_T_block = tl.math.exp(QK_T_block - m[None, :])

            mask_block = offs_q[None, :] >= offs_kv[:, None]
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

            dO_block = tl.load(dO_ptrs)
            dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

            Di = tl.load(D + offs_q)
            dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

            dS_T_block = (P_T_block * (dpT_block - Di[None, :])).to(tl.float16)
            dK_block += tl.dot(dS_T_block, tl.trans(qT_block))

            curr_q += BLOCK_Q
            qT_ptrs += BLOCK_Q * stride_seq
            dO_ptrs += BLOCK_Q * stride_seq

    # full unmasked blocks
    start_unmasked = hi_transition if STAGE == 3 else 0
    for _ in range(start_unmasked, SEQ_LEN, BLOCK_Q):
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + offs_q_base
        m = tl.load(M + offs_q)

        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offs_q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        dS_T_block = (P_T_block * (dpT_block - Di[None, :])).to(tl.float16)
        dK_block += tl.dot(dS_T_block, tl.trans(qT_block))

        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dK_block = dK_block * softmax_scale

    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)
