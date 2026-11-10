import sys
import torch
import triton

from triton_matmul import matmul_kernel

# One workload for controlled profiling
M, N, K = 512, 1024, 256

CONFIGS = {
    "A": (32, 64, 32, 4),
    "B": (64, 64, 32, 4)
}

if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
    raise SystemExit("Usage: python3 profile_triton.py A|B")

name = sys.argv[1]
BM, BN, BK, warps = CONFIGS[name]

torch.manual_seed(42)

A = torch.rand(
    (M, K), device="cuda", dtype=torch.float16
) - 0.5

B = torch.rand(
    (K, N), device="cuda", dtype=torch.float16
) - 0.5

C = torch.empty(
    (M, N), device="cuda", dtype=torch.float16
)

reference = A @ B

grid = (
    triton.cdiv(M, BM),
    triton.cdiv(N, BN)
)

matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BM, BN, BK,
    num_warps=warps
)

torch.cuda.synchronize()

torch.testing.assert_close(
    C, reference,
    atol=0.02,
    rtol=0.01
)

print(f"Configuration {name}: {(BM, BN, BK, warps)}")
print("Correctness: PASSED")
