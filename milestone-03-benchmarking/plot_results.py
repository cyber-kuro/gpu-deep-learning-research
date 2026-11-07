import csv
import matplotlib.pyplot as plt

sizes = []
latencies = []
throughputs = []

sizes = []
latencies = []
throughputs = []

with open("matmul_summary.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        size = int(row["matrix_size"])
        latency = float(row["median_latency_ms"])

        # Calculate achieved throughput
        tflops = (2 * size**3) / (latency * 1e9)

        sizes.append(size)
        latencies.append(latency)
        throughputs.append(tflops)

print("Sizes:", sizes)
print("Latencies:", latencies)
print("Throughputs:", throughputs)

# Plotting Latency vs Matrix Size
import matplotlib.pyplot as plt

plt.plot(sizes, throughputs, marker="o")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Achieved Throughput (TFLOP/s)")
plt.title("RTX 5090: Matrix Size vs Throughput")
plt.grid(True)

# Save the graph as an image
plt.savefig("throughput.png", dpi=300)
plt.close()

# Graph 2: Matrix Size vs Latency

plt.figure(figsize=(8, 5))

# Plot matrix sizes against latencies
plt.plot(sizes, latencies, marker="o")

# Label your axes
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Latency (ms)")

# Give the graph an appropriate title
plt.title("RTX 5090: Matrix Size vs Latency")

plt.grid(True)

# Save your latency graph
plt.savefig("latency.png", dpi=300)
plt.close()
