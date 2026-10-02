import sys
import math
import torch

from .naive_attention import naive_attention_forward_and_backward
from .flash_attention import flash_attention
from .flash_mha import FlashMHA


def run_numerical_test(
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    causal: bool = True,
    dtype: torch.dtype = torch.float16,
    atol: float = 1e-2,
):
    print(f"\n[TEST] B={batch_size}, H={num_heads}, SEQ={seq_len}, D={head_dim}, causal={causal}")

    if not torch.cuda.is_available():
        print("--> CUDA is not available on this machine. Skipping GPU kernel execution.")
        return

    softmax_scale = 1.0 / math.sqrt(head_dim)

    Q = torch.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(0.0, 0.5)
    K = torch.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(0.0, 0.5)
    V = torch.empty((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda").normal_(0.0, 0.5)
    dO = torch.randn_like(Q)

    ref_O, ref_dQ, ref_dK, ref_dV = naive_attention_forward_and_backward(
        Q, K, V, dO, causal=causal, softmax_scale=softmax_scale
    )

    #flashAttention Triton implementation
    Q_tri = Q.clone().requires_grad_(True)
    K_tri = K.clone().requires_grad_(True)
    V_tri = V.clone().requires_grad_(True)

    tri_O = flash_attention(Q_tri, K_tri, V_tri, causal=causal, softmax_scale=softmax_scale)
    tri_O.backward(dO)

    diff_O = torch.max(torch.abs(ref_O - tri_O)).item()
    diff_dQ = torch.max(torch.abs(ref_dQ - Q_tri.grad)).item()
    diff_dK = torch.max(torch.abs(ref_dK - K_tri.grad)).item()
    diff_dV = torch.max(torch.abs(ref_dV - V_tri.grad)).item()

    print(f"    Max Diff O:  {diff_O:.6f} (atol={atol})")
    print(f"    Max Diff dQ: {diff_dQ:.6f} (atol={atol})")
    print(f"    Max Diff dK: {diff_dK:.6f} (atol={atol})")
    print(f"    Max Diff dV: {diff_dV:.6f} (atol={atol})")

    assert torch.allclose(ref_O, tri_O, atol=atol, rtol=0.0), f"O mismatch: {diff_O}"
    assert torch.allclose(ref_dQ, Q_tri.grad, atol=atol, rtol=0.0), f"dQ mismatch: {diff_dQ}"
    assert torch.allclose(ref_dK, K_tri.grad, atol=atol, rtol=0.0), f"dK mismatch: {diff_dK}"
    assert torch.allclose(ref_dV, V_tri.grad, atol=atol, rtol=0.0), f"dV mismatch: {diff_dV}"
    print("    --> PASSED!")


def run_mha_module_test():
    print("\n[TEST] FlashMHA module forward and backward pass")
    if not torch.cuda.is_available():
        print("--> CUDA is not available on this machine. Skipping GPU kernel execution.")
        return

    mha = FlashMHA(d_model=256, num_heads=4, causal=True).to("cuda").half()
    x = torch.randn(2, 512, 256, device="cuda", dtype=torch.float16, requires_grad=True)

    out = mha(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradient check failed!"
    print("    --> FlashMHA module PASSED!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: Running tests in CPU mode. CUDA kernels require GPU execution.")
    else:
        # causal attention tests
        run_numerical_test(batch_size=2, num_heads=4, seq_len=1024, head_dim=64, causal=True)
        #non-causal Attention tests
        run_numerical_test(batch_size=2, num_heads=4, seq_len=1024, head_dim=64, causal=False)
        # module integration test
        run_mha_module_test()
        print("\nALL FLASHATTENTION-2 PIPELINE TESTS PASSED!")
