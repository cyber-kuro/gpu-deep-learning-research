
import csv
import random
from pathlib import Path

import torch
import triton
import triton.testing

from triton_matmul import matmul_kernel


# =============================================
# 1. Experimental configurations
# =============================================


# Load existing workload shapes
BASE_DIR = Path(__file__).resolve().parent

with open(
    BASE_DIR / "data/processed/workload_summary.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    ALL_WORKLOADS = [
        (
            int(row["M"]),
            int(row["N"]),
            int(row["K"])
        )
        for row in reader
    ]

assert len(ALL_WORKLOADS) == 120
assert len(set(ALL_WORKLOADS)) == 120

# Switch between smoke test and full dataset
SMOKE_TEST = False

if SMOKE_TEST:
    WORKLOADS = ALL_WORKLOADS[:12]
else:
    WORKLOADS = ALL_WORKLOADS

print("Selected workloads:", len(WORKLOADS))

assert len(WORKLOADS) == 120

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
CONFIGS = [
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 8),
]

NUM_TRIALS = 3
WARMUP_MS = 50
REPETITION_MS = 200
SEED = 42

OUTPUT_FILE = BASE_DIR / (
    "triton_autotuning_smoke.csv"
    if SMOKE_TEST
    else "triton_autotuning_data.csv"
)

FIELDNAMES = [
    "trial",
    "M",
    "N",
    "K",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "latency_ms",
    "pytorch_ms",
    "speedup",
]


# =============================================
# 2. Benchmarking helper
# =============================================

def time_gpu(fn):
    """Measure steady-state GPU latency using CUDA Graph replay."""

    return triton.testing.do_bench_cudagraph(
        fn,
        rep=100,
        return_mode="median",
    )

# =============================================
# 3. Triton kernel launcher
# =============================================

def run_triton(A, B, C, M, N, K, config):

    bm, bn, bk, warps = config

    grid = (
        triton.cdiv(M, bm),
        triton.cdiv(N, bn),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        bm, bn, bk,
        num_warps=warps,
    )


# =============================================
# 4. Main experiment
# =============================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this benchmark"
        )

    torch.manual_seed(SEED)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("CUDA build:", torch.version.cuda)
    print("Input dtype: torch.float16")

    print("Benchmark method: CUDA Graph replay")
    print("Repetition budget: 100 ms")

    results = []

    # =========================================
    # 5. Iterate over workloads
    # =========================================

    for workload_idx, (M, N, K) in enumerate(WORKLOADS):

        print(
            f"\n=== Workload M={M}, N={N}, K={K} ==="
        )

        # Create FP16 matrices on GPU
        A = torch.rand(
            (M, K),
            device="cuda",
            dtype=torch.float16
        ) - 0.5

        B = torch.rand(
            (K, N),
            device="cuda",
            dtype=torch.float16
        ) - 0.5

        # Preallocate output tensors
        C_ref = torch.empty(
            (M, N),
            device="cuda",
            dtype=torch.float16
        )

        C_triton = torch.empty_like(C_ref)

        # Calculate PyTorch reference
        torch.mm(A, B, out=C_ref)

        def run_pytorch():
            torch.mm(A, B, out=C_ref)

        # =====================================
        # 6. Compile and verify configurations
        # =====================================

        for config in CONFIGS:

            run_triton(
                A, B, C_triton,
                M, N, K,
                config
            )

            torch.testing.assert_close(
                C_triton,
                C_ref,
                atol=0.02,
                rtol=0.01
            )

        torch.cuda.synchronize()

        print(
            "All 4 configurations passed correctness tests."
        )

        # =====================================
        # 7. Repeated trial batches
        # =====================================

        for trial in range(1, NUM_TRIALS + 1):

            # Include PyTorch baseline in randomized order
            methods = [("pytorch", None)] + [
                ("triton", config)
                for config in CONFIGS
            ]

            rng = random.Random(
                SEED + 100 * workload_idx + trial
            )

            rng.shuffle(methods)

            pytorch_ms = None
            triton_times = {}

            # =================================
            # 8. Measure randomized methods
            # =================================

            for method, config in methods:

                if method == "pytorch":

                    pytorch_ms = time_gpu(
                        run_pytorch
                    )

                else:

                    def run_candidate(cfg=config):
                        run_triton(
                            A, B, C_triton,
                            M, N, K, cfg
                        )

                    latency_ms = time_gpu(
                        run_candidate
                    )

                    triton_times[config] = latency_ms

            assert pytorch_ms is not None
            assert pytorch_ms > 0
            assert len(triton_times) == len(CONFIGS)

            print(
                f"\nTrial {trial} | "
                f"PyTorch: {pytorch_ms:.6f} ms"
            )

            # =================================
            # 9. Calculate and store speedups
            # =================================

            for bm, bn, bk, warps in CONFIGS:

                latency_ms = triton_times[
                    (bm, bn, bk, warps)
                ]

                speedup = pytorch_ms / latency_ms

                result = {
                    "trial": trial,
                    "M": M,
                    "N": N,
                    "K": K,
                    "block_m": bm,
                    "block_n": bn,
                    "block_k": bk,
                    "num_warps": warps,
                    "latency_ms": latency_ms,
                    "pytorch_ms": pytorch_ms,
                    "speedup": speedup,
                }

                results.append(result)

                print(
                    f"BM={bm}, BN={bn}, BK={bk}, "
                    f"Warps={warps} | "
                    f"Latency={latency_ms:.6f} ms | "
                    f"Speedup={speedup:.3f}x"
                )

    # =========================================
    # 10. Validate result count
    # =========================================

    expected = (
        len(WORKLOADS)
        * NUM_TRIALS
        * len(CONFIGS)
    )

    assert len(results) == expected, (
        f"Expected {expected} rows; "
        f"got {len(results)}"
    )

    # =========================================
    # 11. Export results to CSV
    # =========================================

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nBenchmark complete. "
        f"Saved {len(results)} rows to {OUTPUT_FILE}"
    )

    print(
        "Expected CSV lines (including header):",
        len(results) + 1
    )


if __name__ == "__main__":
    main()
