# FlashAttention-2 Triton Pipeline: From First Principles

A modular, production-ready, and educational implementation of **FlashAttention-2** written in **OpenAI Triton** and **PyTorch**, inspired by Umar Jamil's deep-dive course and Tri Dao's original research.

---

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Directory Structure](#directory-structure)
3. [Algorithmic Foundations](#algorithmic-foundations)
4. [Solutions to Exercise 1 & Exercise 2](#solutions-to-exercise-1--exercise-2)
5. [Quick Start & Usage](#quick-start--usage)
6. [Testing & Validation](#testing--validation)
7. [Benchmarking & Performance](#benchmarking--performance)
8. [Hardware & Execution Notes](#hardware--execution-notes)

---

## Overview & Motivation

Standard Multi-Head Attention requires computing:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

For a sequence of length $N$, the intermediate matrix $S = QK^T / \sqrt{d}$ and attention weight matrix $P = \text{softmax}(S)$ both have shape $(B, H, N, N)$. 
- At $N = 16,384$, materializing $P$ in GPU High Bandwidth Memory (HBM/DRAM) consumes **over 1 GB per sample**.
- This memory access pattern is **memory-bound**; GPUs spend the vast majority of time moving bytes between slow HBM and fast SRAM rather than computing floating-point operations.

**FlashAttention-2 solves this by:**
1. **Tiling & Fusing in SRAM**: Keeping query and output tiles in fast on-chip SRAM/registers and streaming key-value tiles.
2. **Online Softmax**: Computing softmax statistics dynamically with running maximum $m$ and running sum $l$, updating output accumulators $O$ without pre-scanning the full sequence.
3. **Recomputation in Backward Pass**: Never saving the $O(N^2)$ attention matrix $P$. Instead, $P$ is recomputed on-the-fly in SRAM from $Q, K$, and saved logsumexp $M$, achieving **$O(N)$ memory complexity**.

---

## Directory Structure

```
flash_attention_hand_notes/
├── __init__.py                  # Package exports for all components
├── readme.md                    # Comprehensive documentation and guide
├── naive_attention.py           # Standard PyTorch O(N^2) reference & memory scaling formulas
├── online_softmax.py            # 3-pass safe vs 2-pass online vs 1-pass tiled accumulator simulation
├── fwd_kernel.py                # FlashAttention-2 forward kernel (outer Q loop, 2-stage causal masking)
├── bwd_preprocess.py            # Backward preprocessing kernel: Di = rowsum(dO * O)
├── bwd_kernel.py                # Backward kernels (dQ, dK, dV) with on-the-fly SRAM recomputation & causal skipping
├── flash_attention.py           # PyTorch autograd Function wrapper & flash_attention() API
├── flash_mha.py                 # Drop-in torch.nn.Module Multi-Head Attention layer
├── test_pipeline.py             # Numerical correctness test suite vs PyTorch reference
├── benchmark_pipeline.py        # Latency, memory, and TFLOPs/s benchmarking suite
└── pipeline_walkthrough.ipynb   # Interactive step-by-step master notebook
```

---

## Algorithmic Foundations

### 1. The Online Softmax Rescaling Trick
Standard 3-pass safe softmax requires knowing the global row maximum beforehand. Online softmax updates running statistics iteratively:

$$m_{\text{new}} = \max(m_{\text{old}}, \max(x^{(k)})), \quad \alpha = e^{m_{\text{old}} - m_{\text{new}}}$$
$$l_{\text{new}} = l_{\text{old}} \cdot \alpha + \sum e^{x^{(k)} - m_{\text{new}}}$$
$$O_{\text{new}} = O_{\text{old}} \cdot \alpha + P^{(k)} V^{(k)}$$

At the end of the loop, output is normalized by $O = O / l$.

### 2. Backward Pass Decoupling ($D_i$)
Instead of backpropagating through the full softmax Jacobian:
$$D_i = \sum_{d} (dO)_{id} \cdot O_{id} = \text{rowsum}(dO \circ O)$$
$$dS = P \circ (dP - D_i)$$
This reduces the softmax derivative to an element-wise scale and subtraction, avoiding any $N \times N$ matrix operations in HBM.

---

## Solutions to Exercise 1 & Exercise 2

### Exercise 1: Autotuning the Backward Pass
In [`bwd_preprocess.py`](bwd_preprocess.py) and [`bwd_kernel.py`](bwd_kernel.py), kernels are decorated with `@triton.autotune`:
- **`_attn_bwd_preprocess`**: Autotuned over `BLOCK_SIZE_Q` $\in [32, 64, 128]$ and `num_warps` $\in [2, 4, 8]$.
- **`_attn_bwd_dk_dv`**: Autotuned over `(BLOCK_Q, BLOCK_KV)`, `num_warps`, and `num_stages` (supporting asynchronous software pipelining with `num_stages=4`).
- **`_attn_bwd_dq`**: Autotuned over `(BLOCK_Q, BLOCK_KV)`, `num_warps`, and `num_stages`.
- In [`flash_attention.py`](flash_attention.py), grid dimensions dynamically adapt to candidate parameters using lambda grids:
  ```python
  grid_dk_dv = lambda args: (triton.cdiv(seq_len, args["BLOCK_KV"]), 1, batch_size * num_heads)
  grid_dq = lambda args: (triton.cdiv(seq_len, args["BLOCK_Q"]), 1, batch_size * num_heads)
  ```

### Exercise 2: Causal Skipping (Cutting 50% of Backward FLOPs)
In causal attention, $P_{ij} = 0$ for all $j > i$.
1. **Early Exit in `_attn_bwd_dq`**: Query tile $i \in [\text{start\_q}, \text{start\_q} + \text{BLOCK\_Q} - 1]$ only interacts with keys $j \le i$. The loop terminates early at `hi_total = (start_q + BLOCK_Q + BLOCK_KV - 1) // BLOCK_KV * BLOCK_KV`, skipping the upper-triangular dead space.
2. **Late Start in `_attn_bwd_dk_dv`**: Key tile $j \in [\text{start\_kv}, \text{start\_kv} + \text{BLOCK\_KV} - 1]$ only interacts with queries $i \ge j$. Query blocks where $i < \text{start\_kv}$ evaluate to zero and are skipped by starting directly at `lo_q = (start_kv // BLOCK_Q) * BLOCK_Q`.

### Bonus Optimizations
- **Two-Stage Causal Splitting**: Active blocks are divided into:
  - *Full Tiles* (strictly below diagonal): Executed with **zero `tl.where` masking overhead**.
  - *Boundary Transition Tile*: Only the single tile intersecting the diagonal applies `tl.where(mask, P, 0.0)`.
- **Factored Epilogue Scaling**: Factored `softmax_scale` out of inner loops and applied once to registers right before `tl.store`.

---

## Quick Start & Usage

### 1. Functional API (`flash_attention`)
```python
import torch
from flash_attention_hand_notes import flash_attention

B, H, N, D = 4, 8, 2048, 64
Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)

# Forward pass
O = flash_attention(Q, K, V, causal=True)

# Backward pass
dO = torch.randn_like(O)
O.backward(dO)

print("dQ shape:", Q.grad.shape)
print("dK shape:", K.grad.shape)
print("dV shape:", V.grad.shape)
```

### 2. PyTorch `nn.Module` Layer (`FlashMHA`)
```python
import torch
from flash_attention_hand_notes import FlashMHA

# Standard (BATCH, SEQ_LEN, D_MODEL) sequence inputs
layer = FlashMHA(d_model=512, num_heads=8, causal=True).to("cuda").half()

x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.float16, requires_grad=True)
out = layer(x)

out.sum().backward()
print("Output shape:", out.shape)
```

---

## Testing & Validation

The test suite compares outputs and gradients against native PyTorch attention:
```bash
python -m flash_attention_hand_notes.test_pipeline
```

Validation criteria:
- Forward output: `torch.allclose(ref_O, tri_O, atol=1e-2)`
- Backward gradients: `torch.allclose(ref_dQ, tri_dQ, atol=1e-2)`, `ref_dK`, `ref_dV`
- Tested across both `causal=True` and `causal=False` modes.

---

## Benchmarking & Performance

Run the benchmark suite to compare latency and TFLOPs/s against native PyTorch `F.scaled_dot_product_attention`:
```bash
python -m flash_attention_hand_notes.benchmark_pipeline
```

Throughput formula for causal attention:
$$\text{TFLOPs/s} = \frac{2 \times B \times H \times N^2 \times D}{\text{runtime (seconds)} \times 10^{12}}$$

---

## Hardware & Execution Notes

- **Runtime Requirement**: Running the Triton kernels requires an NVIDIA GPU (Turing, Ampere, Ada, or Hopper architecture with Compute Capability $\ge 7.0$ for Tensor Cores / fp16 MMA).
- **Running in Google Colab (Free GPU)**:
  1. Open [Google Colab](https://colab.research.google.com).
  2. Set **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**.
  3. Install Triton: `!pip install triton torch`.
  4. Open [`pipeline_walkthrough.ipynb`](pipeline_walkthrough.ipynb) and run all cells interactively.
