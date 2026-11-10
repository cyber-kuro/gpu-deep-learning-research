import csv
import copy
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(filename):
    with open(DATA_DIR / filename, newline="") as f:
        rows = list(csv.DictReader(f))

    X = np.array([
        [float(r["M"]), float(r["N"]), float(r["K"])]
        for r in rows
    ])
    y = np.array([
        float(r["median_latency_ms"])
        for r in rows
    ])
    return X, y


class PerformanceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


X_train, y_train = load_data("train.csv")
X_val, y_val = load_data("val.csv")

fractions = [0.10, 0.25, 0.50, 1.00]
seeds = [42, 43, 44]
results = []

for seed in seeds:
    torch.manual_seed(seed)

    # Shuffle training indices deterministically
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))

    for fraction in fractions:
        count = max(1, round(len(X_train) * fraction))
        selected = indices[:count]

        X_subset = X_train[selected]
        y_subset = y_train[selected]

        # Fit preprocessing only on selected training data
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(np.log2(X_subset))
        X_val_scaled = scaler.transform(np.log2(X_val))

        X_t = torch.tensor(
            X_sub_scaled, dtype=torch.float32, device=device
        )
        y_t = torch.tensor(
            np.log(y_subset).reshape(-1, 1),
            dtype=torch.float32, device=device
        )
        X_val_t = torch.tensor(
            X_val_scaled, dtype=torch.float32, device=device
        )

        # Reset model for each experiment
        torch.manual_seed(seed)
        model = PerformanceMLP().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        best_mae = float("inf")
        best_epoch = 0
        best_state = None
        no_improvement = 0

        for epoch in range(1, 501):
            model.train()

            prediction = model(X_t)
            loss = loss_fn(prediction, y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_predictions = torch.exp(
                    model(X_val_t)
                ).cpu().numpy().flatten()

            val_mae = mean_absolute_error(y_val, val_predictions)

            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= 50:
                break

        results.append({
            "seed": seed,
            "fraction": fraction,
            "train_workloads": count,
            "best_epoch": best_epoch,
            "validation_mae_ms": best_mae
        })

        print(
            f"Seed={seed} | Data={fraction:.0%} | "
            f"Shapes={count} | "
            f"Best epoch={best_epoch} | "
            f"Val MAE={best_mae:.6f} ms"
        )


# Save individual experiment results
with open(RESULT_DIR / "data_efficiency.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)


# Summarize across seeds
print("\n=== DATA EFFICIENCY SUMMARY ===")

summary_rows = []

for fraction in fractions:
    values = [
        r["validation_mae_ms"]
        for r in results
        if r["fraction"] == fraction
    ]

    mean_mae = statistics.mean(values)
    std_mae = statistics.stdev(values)

    print(
        f"Data={fraction:.0%} | "
        f"Mean MAE={mean_mae:.6f} ms | "
        f"Std={std_mae:.6f} ms"
    )

    summary_rows.append({
        "fraction": fraction,
        "mean_mae_ms": mean_mae,
        "std_mae_ms": std_mae
    })

with open(RESULT_DIR / "data_efficiency_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nData-efficiency experiment completed.")
