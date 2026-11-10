import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "triton_autotuning_data.csv"
SPLIT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "autotuning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    (32, 64, 32, 4),
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 64, 8),
}


# Step 1: Load the existing workload splits
def load_shapes(filename):
    with open(SPLIT_DIR / filename, newline="") as f:
        reader = csv.DictReader(f)

        return {
            (int(r["M"]), int(r["N"]), int(r["K"]))
            for r in reader
        }


train_shapes = load_shapes("train.csv")
val_shapes = load_shapes("val.csv")
test_shapes = load_shapes("test.csv")

assert len(train_shapes) == 84
assert len(val_shapes) == 18
assert len(test_shapes) == 18

assert not (train_shapes & val_shapes)
assert not (train_shapes & test_shapes)
assert not (val_shapes & test_shapes)


# Step 2: Group measurements by workload and config
measurements = defaultdict(dict)

with open(RAW_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 1440

for r in rows:
    shape = (
        int(r["M"]),
        int(r["N"]),
        int(r["K"])
    )

    config = (
        int(r["block_m"]),
        int(r["block_n"]),
        int(r["block_k"]),
        int(r["num_warps"])
    )

    assert config in CONFIGS

    trial = int(r["trial"])
    latency = float(r["latency_ms"])
    pytorch_ms = float(r["pytorch_ms"])
    speedup = float(r["speedup"])

    assert math.isfinite(latency) and latency > 0
    assert math.isfinite(pytorch_ms) and pytorch_ms > 0
    assert math.isclose(
        speedup, pytorch_ms / latency,
        rel_tol=1e-3
    )

    key = shape + config

    if trial in measurements[key]:
        raise ValueError(f"Duplicate trial for {key}")

    measurements[key][trial] = (latency, pytorch_ms)


# Step 3: Verify dataset completeness
assert len(measurements) == 480

all_shapes = {key[:3] for key in measurements}

assert len(all_shapes) == 120
assert all_shapes == (
    train_shapes | val_shapes | test_shapes
)

for shape in all_shapes:
    observed_configs = {
        key[3:] for key in measurements
        if key[:3] == shape
    }
    assert observed_configs == CONFIGS

for trial_data in measurements.values():
    assert set(trial_data) == {1, 2, 3}


# Step 4: Aggregate repeated trials
summary = []

for key, trials in sorted(measurements.items()):

    M, N, K, BM, BN, BK, warps = key

    latencies = [trials[t][0] for t in (1, 2, 3)]
    baselines = [trials[t][1] for t in (1, 2, 3)]

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies)

    pytorch_ms = statistics.median(baselines)

    summary.append({
        "M": M,
        "N": N,
        "K": K,
        "block_m": BM,
        "block_n": BN,
        "block_k": BK,
        "num_warps": warps,
        "median_latency_ms": median_ms,
        "std_latency_ms": std_ms,
        "cv_percent": 100 * std_ms / mean_ms,
        "pytorch_ms": pytorch_ms,
        "speedup": pytorch_ms / median_ms
    })


# Step 5: Write the processed datasets
def save_csv(filename, data):
    with open(OUTPUT_DIR / filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(data)


train_data = [
    r for r in summary
    if (r["M"], r["N"], r["K"]) in train_shapes
]

val_data = [
    r for r in summary
    if (r["M"], r["N"], r["K"]) in val_shapes
]

test_data = [
    r for r in summary
    if (r["M"], r["N"], r["K"]) in test_shapes
]

assert len(train_data) == 336
assert len(val_data) == 72
assert len(test_data) == 72

save_csv("autotuning_summary.csv", summary)
save_csv("train.csv", train_data)
save_csv("val.csv", val_data)
save_csv("test.csv", test_data)


# Step 6: Print validation results
high_cv = sum(
    r["cv_percent"] > 10 for r in summary
)

print("=== AUTOTUNING DATASET SUMMARY ===")
print("Raw measurements:", len(rows))
print("Unique workload shapes:", len(all_shapes))
print("Unique workload-config pairs:", len(summary))
print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))
print("Testing pairs:", len(test_data))
print("Pairs with CV > 10%:", high_cv)

print("\nAUTOTUNING DATA PREPARATION PASSED")
