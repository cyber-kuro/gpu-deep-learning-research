import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "autotuning"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "M", "N", "K",
    "block_m", "block_n", "block_k",
    "num_warps"
]


# 1. Read the processed datasets
def load_rows(filename):
    with open(DATA_DIR / filename, newline="") as f:
        return list(csv.DictReader(f))


train_rows = load_rows("train.csv")
val_rows = load_rows("val.csv")


# 2. Build numerical features and targets
def get_features(rows):
    X = np.array([
        [float(row[name]) for name in FEATURES]
        for row in rows
    ])

    return np.log2(X)


def get_targets(rows):
    return np.array([
        float(row["median_latency_ms"])
        for row in rows
    ])


X_train = get_features(train_rows)
X_val = get_features(val_rows)

y_train = get_targets(train_rows)
y_val = get_targets(val_rows)

assert len(train_rows) == 336
assert len(val_rows) == 72

# Ensure workload shapes are separated
train_shapes = {
    (r["M"], r["N"], r["K"])
    for r in train_rows
}

val_shapes = {
    (r["M"], r["N"], r["K"])
    for r in val_rows
}

assert not train_shapes.intersection(val_shapes)


# 3. Train Random Forest latency predictor
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, np.log(y_train))


# 4. Predict validation latencies
predictions = np.exp(model.predict(X_val))

mae = mean_absolute_error(y_val, predictions)

print("=== CONFIGURATION PREDICTION ===")
print("Training pairs:", len(train_rows))
print("Validation pairs:", len(val_rows))
print(f"Validation MAE: {mae:.6f} ms")


# 5. Group validation records by workload
groups = defaultdict(list)

for i, row in enumerate(val_rows):
    shape = (
        int(row["M"]),
        int(row["N"]),
        int(row["K"])
    )

    groups[shape].append(i)


# 6. Evaluate recommended configurations
selection_results = []

for shape, indices in sorted(groups.items()):

    assert len(indices) == 4

    # Configuration with lowest predicted latency
    selected_idx = min(
        indices,
        key=lambda i: predictions[i]
    )

    # Best measured configuration
    optimal_idx = min(
        indices,
        key=lambda i: y_val[i]
    )

    selected_ms = y_val[selected_idx]
    optimal_ms = y_val[optimal_idx]

    regret_percent = (
        selected_ms / optimal_ms - 1
    ) * 100

    selected_row = val_rows[selected_idx]

    selection_results.append({
        "M": shape[0],
        "N": shape[1],
        "K": shape[2],
        "block_m": selected_row["block_m"],
        "block_n": selected_row["block_n"],
        "block_k": selected_row["block_k"],
        "num_warps": selected_row["num_warps"],
        "selected_latency_ms": selected_ms,
        "optimal_latency_ms": optimal_ms,
        "regret_percent": regret_percent,
        "exact_best": int(selected_idx == optimal_idx),
        "within_5_percent": int(
            regret_percent <= 5.0 + 1e-9
        )
    })


# 7. Summarize configuration selection
regrets = [
    r["regret_percent"]
    for r in selection_results
]

exact_best = sum(
    r["exact_best"]
    for r in selection_results
)

near_best = sum(
    r["within_5_percent"]
    for r in selection_results
)

print("\n=== AUTOTUNING RECOMMENDATIONS ===")

print("Evaluated workloads:", len(selection_results))

print(
    "Exact best configuration:",
    f"{exact_best}/{len(selection_results)}"
)

print(
    "Within 5% of best:",
    f"{near_best}/{len(selection_results)}"
)

print(
    f"Mean selection regret: "
    f"{np.mean(regrets):.2f}%"
)

print(
    f"Median selection regret: "
    f"{np.median(regrets):.2f}%"
)


# 8. Save validation decisions
output_file = RESULT_DIR / "rf_autotuner_validation.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(selection_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(selection_results)


# 9. Save the trained model
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "feature_transform": "log2",
        "target_transform": "natural_log"
    },
    RESULT_DIR / "rf_autotuner.joblib"
)

print("\nAutotuner model and results saved.")
