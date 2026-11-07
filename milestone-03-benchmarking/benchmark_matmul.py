import torch
import statistics
import csv

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Input dtype:", torch.float32)
print("Matmul precision:", torch.get_float32_matmul_precision())
print("")

sizes = [256, 512, 1024, 2048, 4096]

results = []

for trial in range(1, 6):

    print("Starting trial:", trial)

    for N in sizes:
        # Create A and B on the GPU
        A = torch.randn(N, N, device='cuda')
        B = torch.randn(N, N, device='cuda')

        # Warm up the GPU 10 times
        for _ in range(10):
            C = A @ B

        torch.cuda.synchronize()

        times_ms = []

        # Measure 20 runs
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            # Perform GPU matrix multiplication
            C = A @ B

            end.record()
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)
            times_ms.append(elapsed_ms)

        # Calculate the median latency
        median_ms = statistics.median(times_ms)

        print("GPU:", torch.cuda.get_device_name(0))
        print("Matrix size:", N, "x", N)
        print("Median latency (ms):", median_ms)

        # Calculate total floating-point operations
        flops = 2 * N * N * N  # For matrix multiplication, the number of FLOPs is 2 * N^3


        # Convert latency from milliseconds to seconds
        latency_seconds = median_ms / 1000

        # Calculate achieved TFLOP/s
        tflops = flops / latency_seconds / 1e12
        print("Achieved TFLOP/s:", tflops)
        print("")

        
        # Add this after calculating latency and TFLOP/s
        result = {
            "trial": trial,
            "matrix_size": N,
            "latency_ms": median_ms,
            "tflops": tflops
        }

        results.append(result)

# Save benchmark results to CSV
with open("matmul_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["trial", "matrix_size", "latency_ms", "tflops"]
    )

    writer.writeheader()
    writer.writerows(results)

print("Benchmark results saved successfully!")
