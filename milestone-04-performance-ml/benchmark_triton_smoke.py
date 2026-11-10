import csv
import torch
import triton
import triton.testing

from triton_matmul import matmul_kernel

torch.manual_seed(42)

# One workload for our initial benchmark
M, N, K = 1024, 1024, 1024

A = torch.rand((M, K), device="cuda", dtype=torch.float16) - 0.5
B = torch.rand((K, N), device="cuda", dtype=torch.float16) - 0.5

# Preallocate outputs for fairer timing
C_ref = torch.empty((M, N), device="cuda", dtype=torch.float16)
C = torch.empty_like(C_ref)

# PyTorch reference
torch.mm(A, B, out=C_ref)

configs = [
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 8),
]

# Measure PyTorch with a preallocated output
pytorch_ms = triton.testing.do_bench(
    lambda: torch.mm(A, B, out=C_ref),
    warmup=50,
    rep=200,
    return_mode="median"
)

print(f"PyTorch FP16 latency: {pytorch_ms:.6f} ms")

results = []

for BM, BN, BK, warps in configs:

    grid = (
        triton.cdiv(M, BM),
        triton.cdiv(N, BN)
    )

    def run_kernel():
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            BM, BN, BK,
            num_warps=warps
        )

    # Compile and test before timing
    run_kernel()

    torch.testing.assert_close(
        C, C_ref,
        atol=0.02,
        rtol=0.01
    )

    # Benchmark the already-compiled configuration
    latency_ms = triton.testing.do_bench(
        run_kernel,
        warmup=50,
        rep=200,
        return_mode="median"
    )

    speedup = pytorch_ms / latency_ms

    results.append({
        "M": M,
        "N": N,
        "K": K,
        "block_m": BM,
        "block_n": BN,
        "block_k": BK,
        "num_warps": warps,
        "latency_ms": latency_ms,
        "pytorch_ms": pytorch_ms,
        "speedup": speedup
    })

    print(
        f"BM={BM}, BN={BN}, BK={BK}, Warps={warps} | "
        f"Latency={latency_ms:.6f} ms | "
        f"Speedup={speedup:.3f}x"
    )

with open("triton_config_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nConfiguration benchmark completed.")
