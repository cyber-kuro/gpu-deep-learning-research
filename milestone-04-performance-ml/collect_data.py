import torch
import statistics
import random
from itertools import product
import csv

torch.set_float32_matmul_precision("highest")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("FP32 precision:", torch.get_float32_matmul_precision())

# Each tuple represents (M, N, K)
dimensions = [128, 256, 384, 512, 768, 1024, 1536, 2048]

all_workloads = list(product(dimensions, repeat=3))

rng = random.Random(42)

workloads = rng.sample(all_workloads, 120)

results = []

for trial in range(1, 6):

    for M, N, K in workloads:

        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")

        # Warm up the GPU
        for _ in range(10):
            C = A @ B

        torch.cuda.synchronize()

        times_ms = []

        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            # Perform matrix multiplication
            C = A @ B

            end.record()
            end.synchronize()

            # Calculate elapsed time in milliseconds
            elapsed_ms = start.elapsed_time(end)

            times_ms.append(elapsed_ms)

        # Calculate median latency
        median_ms = statistics.median(times_ms)

        # calculate total floating-point operations
        flops = 2 * M * N * K  # For matrix multiplication, the number of FLOPs is 2 * M * N * K

        # Calculate achieved TFLOP/s
        latency_seconds = median_ms / 1000  # Convert milliseconds to seconds
        tflops = flops / latency_seconds / 1e12

        # Print the results
        print(f"Workload: M={M}, N={N}, K={K}")
        print(f"Median latency (ms): {median_ms}")
        print(f"Achieved TFLOP/s: {tflops}\n")

        # Store the results
        result = {
            "trial": trial,
            "M": M,
            "N": N,
            "K": K,
            "flops": flops,
            "latency_ms": median_ms,
            "tflops": tflops
        }

        results.append(result)

with open("gpu_performance_data.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "trial",
            "M",
            "N",
            "K",
            "flops",
            "latency_ms",
            "tflops"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("Dataset collected successfully!")