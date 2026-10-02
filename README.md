# FlashAttention-2 from First Principles (Triton)

A modular, educational implementation of **FlashAttention-2** using **OpenAI Triton** and **PyTorch**, developed from first principles with a focus on the forward pass, backward pass, online softmax, causal optimization, autotuning, and numerical validation.

This project is built as an educational companion to and extension of the open exercises in [Umar Jamil's `triton-flash-attention` repository](https://github.com/hkproj/triton-flash-attention) and grounded in the research of Tri Dao et al. ([FlashAttention-2](https://arxiv.org/abs/2307.08691)).

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
- [Project Structure](#project-structure)
- [Algorithmic Foundations](#algorithmic-foundations)
- [Solutions to Umar Jamil's Exercises](#solutions-to-umar-jamils-exercises)
  - [Exercise 1: Autotuning the Backward Pass](#exercise-1-autotuning-the-backward-pass)
  - [Exercise 2: Causal Skipping Optimization](#exercise-2-causal-skipping-optimization)
  - [Bonus Kernel Optimizations](#bonus-kernel-optimizations)
- [Requirements & Installation](#requirements--installation)
- [Quick Start](#quick-start)
- [Testing & Numerical Validation](#testing--numerical-validation)
- [Benchmarking](#benchmarking)
- [Attribution & References](#attribution--references)

---

## Overview & Motivation

Standard Scaled Dot-Product Attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V
$$

For a sequence of length $N$, the intermediate pairwise attention matrix $S = Q K^T / \sqrt{d}$ and the normalized probabilities $P = \text{softmax}(S)$ both have shape $(B, H, N, N)$.
- At $N = 16,384$, storing $P$ in GPU High Bandwidth Memory (HBM/DRAM) consumes **over 1 GB per sample**.
- Standard implementations are **memory-bound**: GPUs spend most clock cycles transferring bytes across the memory bus between slow HBM and on-chip SRAM rather than computing floating-point operations.

**FlashAttention-2 addresses this bottleneck by:**
1. **Tiling & Fusing in SRAM**: Partitioning inputs into blocks that fit within on-chip SRAM/registers and streaming key-value tiles while holding query tiles in fast memory.
2. **Online Softmax**: Computing softmax statistics dynamically using running row maximums $m$ and normalization sums $l$, updating the output accumulator $O$ iteratively without ever materializing the full $N \times N$ matrix in DRAM.
3. **Selective Recomputation in Backward**: Discarding $P$ in the forward pass. During the backward pass, attention scores are recomputed on-the-fly in SRAM from $Q, K$, and saved logsumexp vectors $M$, achieving **$O(N)$ memory scaling**.

---

## Project Structure

```
flash_attention_hand_notes/
├── README.md                    # Project documentation and architectural overview
├── __init__.py                  # Package exports
├── naive_attention.py           # Standard PyTorch O(N^2) reference implementation
├── online_softmax.py            # Safe softmax vs. online softmax streaming simulations
├── fwd_kernel.py                # FlashAttention-2 Triton forward kernel (outer Q loop, 2-stage causal)
├── bwd_preprocess.py            # Backward preprocessing kernel: Di = rowsum(dO · O)
├── bwd_kernel.py                # Optimized backward kernels (dQ, dK, dV) with causal skipping
├── flash_attention.py           # torch.autograd.Function wrapper and functional API
├── flash_mha.py                 # Drop-in torch.nn.Module Multi-Head Attention layer
├── test_pipeline.py             # Numerical correctness test suite vs. PyTorch reference
├── benchmark_pipeline.py        # Latency, memory, and TFLOPs/s benchmarking suite
└── pipeline_walkthrough.ipynb   # Interactive step-by-step master walkthrough notebook
```

---

## Algorithmic Foundations

### 1. Online Softmax Rescaling

Standard safe softmax requires a pre-pass to find the global row maximum. The **online softmax** algorithm (Milakov & Gimelshein, 2018) computes the exact result over streaming blocks:

$$
m_{\text{new}} = \max\left(m_{\text{old}},\, \max(x^{(k)})\right)
$$

$$
\alpha = \exp\left(m_{\text{old}} - m_{\text{new}}\right)
$$

$$
l_{\text{new}} = l_{\text{old}} \cdot \alpha + \sum \exp\left(x^{(k)} - m_{\text{new}}\right)
$$

$$
O_{\text{new}} = O_{\text{old}} \cdot \alpha + P^{(k)} V^{(k)}
$$

At loop termination, the final output accumulator is normalized once:

$$
O = \frac{O_{\text{final}}}{l_{\text{final}}}
$$

### 2. Backward Pass Gradient Decoupling ($D_i$)

Rather than instantiating the $N \times N$ Jacobian for the softmax derivative, FlashAttention projects the gradient along rows of the forward output:

$$
D_i = \sum_{d} (dO)_{id} \cdot O_{id} = \text{rowsum}(dO \circ O)
$$

$$
dS = P \circ (dP - D_i)
$$

where $\circ$ denotes element-wise (Hadamard) multiplication. This reduces backpropagation through softmax to cheap element-wise operations inside SRAM.

---

## Solutions to Umar Jamil's Exercises

This pipeline implements complete solutions to the two open exercises from [hkproj/triton-flash-attention](https://github.com/hkproj/triton-flash-attention):

### Exercise 1: Autotuning the Backward Pass

> *“Can you apply autotuning configs to the backwards pass like done for the forward pass?”*  
> — Umar Jamil

In [`bwd_preprocess.py`](./bwd_preprocess.py) and [`bwd_kernel.py`](./bwd_kernel.py), all three backward kernels are decorated with `@triton.autotune`:

1. **`_attn_bwd_preprocess`**: Autotuned over `BLOCK_SIZE_Q` $\in [32, 64, 128]$ and `num_warps` $\in [2, 4, 8]$.
2. **`_attn_bwd_dk_dv`**: Autotuned over candidate pairs of `(BLOCK_Q, BLOCK_KV)`, `num_warps`, and `num_stages`.
3. **`_attn_bwd_dq`**: Autotuned over candidate pairs of `(BLOCK_Q, BLOCK_KV)`, `num_warps`, and `num_stages`.

Because tile dimensions change dynamically during autotuning, the grid launch parameters in [`flash_attention.py`](./flash_attention.py) are formulated as dynamic lambdas:

```python
grid_dk_dv = lambda args: (triton.cdiv(seq_len, args["BLOCK_KV"]), 1, batch_size * num_heads)
grid_dq = lambda args: (triton.cdiv(seq_len, args["BLOCK_Q"]), 1, batch_size * num_heads)
```

---

### Exercise 2: Causal Skipping Optimization

> *“As you can see, during the backwards pass we are going through the entire `SEQ_LEN` even when the attention calculation is `causal`, can you avoid going through all tokens that would not contribute to any change in `dK`, `dQ` and `dV` when the attention calculation is causal?”*  
> — Umar Jamil

In causal attention, $P_{ij} = 0$ for all keys $j > i$. The original baseline streamed across all $N$ tokens regardless:

```
Key token j ────────────────────────►
      0    1    2    3    4    5    6    7
0   [ P    0    0    0    0    0    0    0 ]
1   [ P    P    0    0    0    0    0    0 ]
2   [ P    P    P    0    0    0    0    0 ]
3   [ P    P    P    P    0    0    0    0 ]  ◄── Query i only attends to j <= i
4   [ P    P    P    P    P    0    0    0 ]
5   [ P    P    P    P    P    P    0    0 ]
6   [ P    P    P    P    P    P    P    0 ]
7   [ P    P    P    P    P    P    P    P ]
      ▲
      └──── Key j only receives gradients from i >= j
```

We optimize both backward passes in [`bwd_kernel.py`](./bwd_kernel.py):

1. **Early Exit in `_attn_bwd_dq`**:
   - A query tile covering row indices from `start_q` to `start_q + BLOCK_Q - 1` never attends to keys where $j >$ `start_q + BLOCK_Q - 1`.
   - The loop terminates early at:
     ```python
     hi_total = (start_q + BLOCK_Q + BLOCK_KV - 1) // BLOCK_KV * BLOCK_KV
     ```
     skipping all tiles in the upper triangle.

2. **Late Start in `_attn_bwd_dk_dv`**:
   - A key tile covering column indices from `start_kv` to `start_kv + BLOCK_KV - 1` never receives gradients from queries where $i <$ `start_kv`.
   - The loop skips all leading zero-product iterations and starts directly at:
     ```python
     lo_q = (start_kv // BLOCK_Q) * BLOCK_Q
     ```

*Theoretical Impact:* The causal lower-triangular structure eliminates roughly half of the query-key tile interactions asymptotically (~50% reduction in theoretical tile operations). The measured speedup depends on tile granularity, boundary handling, and hardware autotuning.

---

### Bonus Kernel Optimizations

1. **Two-Stage Causal Splitting**:
   - *Full Tiles* (strictly below the diagonal): Evaluated with **zero `tl.where` masking overhead**, bypassing mask creation entirely.
   - *Boundary Tiles* (intersecting the diagonal): Only the single diagonal transition block evaluates `tl.where(mask, P, 0.0)`.
2. **Factored Epilogue Scaling**:
   `softmax_scale` is factored out of the inner accumulator loops in `_attn_bwd_dq` and `_attn_bwd_dk_dv`, applying a single scalar multiplication in registers during the epilogue before `tl.store`.

---

## Requirements & Installation

### Prerequisites
- Python $\ge 3.10$
- PyTorch $\ge 2.2.0$
- OpenAI Triton $\ge 2.2.0$
- NVIDIA GPU with Compute Capability $\ge 7.0$ (Turing, Ampere, Ada Lovelace, Hopper) and compatible CUDA drivers.

### Setup
```bash
git clone https://github.com/hkproj/triton-flash-attention.git
cd triton-flash-attention

# Install dependencies
pip install torch triton
```

---

## Quick Start

### 1. Functional API (`flash_attention`)

```python
import torch
from flash_attention_hand_notes import flash_attention

batch_size, num_heads, seq_len, head_dim = 4, 8, 2048, 64

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)

# Forward pass
O = flash_attention(Q, K, V, causal=True)

# Backward pass
dO = torch.randn_like(O)
O.backward(dO)

print("dQ:", Q.grad.shape)
print("dK:", K.grad.shape)
print("dV:", V.grad.shape)
```

### 2. Multi-Head Attention Module (`FlashMHA`)

```python
import torch
from flash_attention_hand_notes import FlashMHA

# Accepts standard (BATCH, SEQ_LEN, D_MODEL) sequence inputs
layer = FlashMHA(d_model=512, num_heads=8, causal=True).to("cuda").half()

x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.float16, requires_grad=True)
output = layer(x)

output.sum().backward()
print("Layer output shape:", output.shape)
```

---

## Testing & Numerical Validation

A unit test suite compares outputs and gradients against native PyTorch attention:

```bash
# Run from repository root
python -m flash_attention_hand_notes.test_pipeline
```

Validation criteria:
- Forward outputs: `torch.allclose(ref_O, tri_O, atol=1e-2)`
- Backward gradients: `torch.allclose(ref_dQ, tri_dQ, atol=1e-2)`, `ref_dK`, `ref_dV`
- Verified across both `causal=True` and `causal=False` modes.

---

## Benchmarking

Benchmark latency and compute throughput (TFLOPs/s) against PyTorch's native `F.scaled_dot_product_attention`:

```bash
# Run from repository root
python -m flash_attention_hand_notes.benchmark_pipeline
```

Throughput formula for causal attention:

$$
\text{TFLOPs/s} = \frac{2 \times B \times H \times N^2 \times D}{\text{runtime (seconds)} \times 10^{12}}
$$

> **Running without a local GPU:**  
> If working on a machine without a local NVIDIA GPU, upload the repository to [Google Colab](https://colab.research.google.com) or Kaggle, select a **T4 GPU** runtime, and run [`pipeline_walkthrough.ipynb`](./pipeline_walkthrough.ipynb) interactively.

---

## Attribution & References

- **Umar Jamil**: Created the [hkproj/triton-flash-attention](https://github.com/hkproj/triton-flash-attention) tutorial repository and the lecture series *"Flash Attention from first principles"*, which provided the pedagogical foundation and exercises for this project.
- **Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022).
- **Tri Dao**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023).
- **OpenAI Triton**: [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
