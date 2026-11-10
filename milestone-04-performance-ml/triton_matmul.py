import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Identify which output tile this program computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Accumulate the output tile in FP32
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32
    )

    # Process the reduction dimension in tiles
    for k_block in range(tl.cdiv(K, BLOCK_K)):
        ks = k_block * BLOCK_K + k_offsets

        a_ptrs = A + rows[:, None] * K + ks[None, :]
        b_ptrs = B + ks[:, None] * N + cols[None, :]

        a = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0
        )

        b = tl.load(
            b_ptrs,
            mask=(ks[:, None] < K) & (cols[None, :] < N),
            other=0.0
        )

        accumulator = tl.dot(a, b, accumulator)

    # Write the output tile
    c_ptrs = C + rows[:, None] * N + cols[None, :]

    mask = (rows[:, None] < M) & (cols[None, :] < N)

    tl.store(
        c_ptrs,
        accumulator.to(tl.float16),
        mask=mask
    )


def triton_matmul(A, B, block_m=32, block_n=64,
                  block_k=32, num_warps=4):

    assert A.ndim == 2 and B.ndim == 2
    assert A.dtype == B.dtype == torch.float16
    assert A.is_cuda and B.is_cuda
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n)
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        block_m, block_n, block_k,
        num_warps=num_warps
    )

    return C

if __name__ == "__main__":
    # Test several matrix shapes
    torch.manual_seed(42)

    shapes = [
        (128, 128, 128),
        (192, 256, 128),
        (130, 250, 96)
    ]

    for M, N, K in shapes:

        A = torch.rand(
            (M, K), device="cuda", dtype=torch.float16
        ) - 0.5

        B = torch.rand(
            (K, N), device="cuda", dtype=torch.float16
        ) - 0.5

        # Trusted reference implementation
        reference = A @ B

        # Our Triton implementation
        result = triton_matmul(A, B)

        max_error = (
            result.float() - reference.float()
        ).abs().max().item()

        torch.testing.assert_close(
            result,
            reference,
            atol=0.02,
            rtol=0.01
        )

        print(f"Shape: M={M}, N={N}, K={K}")
        print("Output shape:", tuple(result.shape))
        print(f"Max absolute error: {max_error:.6f}")
        print("CORRECTNESS PASSED\n")

    print("ALL TRITON TESTS PASSED")

    if __name__ == "__main__":

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
