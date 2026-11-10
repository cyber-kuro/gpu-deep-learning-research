
import csv
import math
import random
import statistics
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "gpu_performance_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Group measurements by workload shape
groups = defaultdict(dict)

with open(INPUT_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        M = int(row["M"])
        N = int(row["N"])
        K = int(row["K"])
        trial = int(row["trial"])
        latency = float(row["latency_ms"])

        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid latency detected")

        if int(row["flops"]) != 2 * M * N * K:
            raise ValueError("Incorrect FLOP count")

        key = (M, N, K)

        if trial in groups[key]:
            raise ValueError("Duplicate trial detected")

        groups[key][trial] = latency

# Step 2: Calculate summary statistics
summary = []

for (M, N, K), trial_data in sorted(groups.items()):

    if set(trial_data) != {1, 2, 3, 4, 5}:
        raise ValueError("Incomplete trial data")

    latencies = list(trial_data.values())

    median_latency = statistics.median(latencies)
    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies)

    summary.append({
        "M": M,
        "N": N,
        "K": K,
        "flops": 2 * M * N * K,
        "median_latency_ms": median_latency,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "cv_percent": 100 * std_latency / mean_latency
    })

assert len(summary) == 120

# Step 3: Randomly split unique workloads
rng = random.Random(42)
rng.shuffle(summary)

train_data = summary[:84]
val_data = summary[84:102]
test_data = summary[102:]

# Step 4: Write CSV files
def save_csv(filename, rows):
    path = OUTPUT_DIR / filename

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

save_csv("workload_summary.csv", summary)
save_csv("train.csv", train_data)
save_csv("val.csv", val_data)
save_csv("test.csv", test_data)

# Step 5: Report
print("Dataset preparation completed!")
print("Total unique workloads:", len(summary))
print("Training workloads:", len(train_data))
print("Validation workloads:", len(val_data))
print("Testing workloads:", len(test_data))

high_variability = [
    row for row in summary if row["cv_percent"] > 10
]

print("Workloads with CV > 10%:", len(high_variability))
print("DATA PREPARATION PASSED")
