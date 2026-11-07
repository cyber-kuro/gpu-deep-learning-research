import csv
import statistics
from collections import defaultdict

# Store latency measurements grouped by matrix size
latencies_by_size = defaultdict(list)

with open("matmul_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        # TODO 1: Convert matrix_size to an integer
        N = int(row["matrix_size"])

        # TODO 2: Convert latency_ms to a float
        latency = float(row["latency_ms"])

        latencies_by_size[N].append(latency)

# Initialize the summary list BEFORE the loop
summary_results = []

for N, latencies in sorted(latencies_by_size.items()):

    # Step 1: Calculate statistics
    median_latency = statistics.median(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)
    std_latency = statistics.stdev(latencies)

    # Step 2: Store the calculated statistics
    summary_results.append({
        "matrix_size": N,
        "median_latency_ms": median_latency,
        "min_latency_ms": minimum_latency,
        "max_latency_ms": maximum_latency,
        "std_latency_ms": std_latency
    })

    # Step 3: Print results
    print("\nMatrix size:", N)
    print("Median latency (ms):", median_latency)
    print("Minimum latency (ms):", minimum_latency)
    print("Maximum latency (ms):", maximum_latency)
    print("Standard deviation (ms):", std_latency)
# Save statistical summary to CSV
with open("matmul_summary.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "matrix_size",
            "median_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "std_latency_ms"
        ]
    )

    writer.writeheader()
    writer.writerows(summary_results)

print("Statistical summary saved successfully!")