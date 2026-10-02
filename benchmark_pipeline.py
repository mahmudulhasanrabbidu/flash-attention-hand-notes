import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.testing
except ImportError:
    triton = None

from .flash_attention import flash_attention


def attention_flops(batch: int, heads: int, seq_len: int, dim: int, causal: bool) -> float:
    """Computes theoretical FLOP count for forward attention."""
    # Q @ K^T is 2 * B * H * N * N * D FLOPs
    # P @ V is 2 * B * H * N * N * D FLOPs
    # Total = 4 * B * H * N * N * D FLOPs (or halved for causal)
    total_flops = 4.0 * batch * heads * seq_len * seq_len * dim
    return total_flops * (0.5 if causal else 1.0)


def benchmark_suite(batch_size: int = 2, num_heads: int = 8, head_dim: int = 64, causal: bool = True, seq_lens = (512, 1024, 2048, 4096, 8192)):
    if not torch.cuda.is_available() or triton is None:
        print("Benchmarking requires an NVIDIA GPU with CUDA and Triton installed.")
        return

    print("=" * 80)
    print(f"FLASH ATTENTION-2 BENCHMARK (Batch={batch_size}, Heads={num_heads}, Dim={head_dim}, Causal={causal})")
    print("=" * 80)
    print(f"{'Seq Len':<10} | {'PyTorch SDPA (ms)':<18} | {'Triton FA2 (ms)':<16} | {'FA2 TFLOPs/s':<14} | {'Speedup':<10}")
    print("-" * 80)

    for seq in seq_lens:
        q = torch.randn(batch_size, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq, head_dim, device="cuda", dtype=torch.float16)

        # PyTorch Native SDPA
        try:
            ms_sdpa = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            )
        except Exception:
            ms_sdpa = float("nan")

        # Triton FlashAttention-2
        try:
            ms_fa2 = triton.testing.do_bench(
                lambda: flash_attention(q, k, v, causal=causal)
            )
            flops = attention_flops(batch_size, num_heads, seq, head_dim, causal)
            tflops = (flops / (ms_fa2 * 1e-3)) / 1e12
            speedup = ms_sdpa / ms_fa2 if not math.isnan(ms_sdpa) else 1.0
            print(f"{seq:<10} | {ms_sdpa:<18.3f} | {ms_fa2:<16.3f} | {tflops:<14.2f} | {speedup:<10.2f}x")
        except Exception as e:
            print(f"{seq:<10} | Error: {e}")

    print("=" * 80)


if __name__ == "__main__":
    benchmark_suite()
