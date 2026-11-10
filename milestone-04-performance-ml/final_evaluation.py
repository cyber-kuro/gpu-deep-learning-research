
import csv
import numpy as np
import torch

from pathlib import Path
from torch import nn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

def load_data(filename):
    with open(DATA_DIR / filename, newline="") as f:
        rows = list(csv.DictReader(f))

    X = np.array([
        [float(r["M"]), float(r["N"]), float(r["K"])]
        for r in rows
    ])
    y = np.array([
        float(r["median_latency_ms"]) for r in rows
    ])

    return X, y

X_train, y_train = load_data("train.csv")
X_val, _ = load_data("val.csv")
X_test, y_test = load_data("test.csv")

# Check that distinct shapes belong to only one split
train_shapes = {tuple(x) for x in X_train}
val_shapes = {tuple(x) for x in X_val}
test_shapes = {tuple(x) for x in X_test}

assert not train_shapes & val_shapes
assert not train_shapes & test_shapes
assert not val_shapes & test_shapes

print("Held-out test workloads:", len(X_test))

# 1. Constant baseline
constant = DummyRegressor(strategy="median")
constant.fit(X_train, y_train)
constant_pred = constant.predict(X_test)

# 2. Log-linear regression
ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(np.log2(X_train), np.log(y_train))
ridge_pred = np.exp(ridge.predict(np.log2(X_test)))

# 3. Random Forest
forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
forest.fit(X_train, np.log(y_train))
forest_pred = np.exp(forest.predict(X_test))

# 4. Previously trained MLP
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

# Load only your own locally generated checkpoint.
# weights_only=False is needed because the earlier checkpoint
# saved NumPy scaler arrays alongside the model weights.
checkpoint = torch.load(
    BASE_DIR / "mlp_checkpoint.pt",
    map_location="cpu",
    weights_only=False
)

mlp = PerformanceMLP()
mlp.load_state_dict(checkpoint["model_state_dict"])
mlp.eval()

X_test_scaled = (
    np.log2(X_test) - checkpoint["scaler_mean"]
) / checkpoint["scaler_scale"]

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

with torch.no_grad():
    mlp_pred = torch.exp(
        mlp(X_test_tensor)
    ).numpy().flatten()

# Final held-out comparisons
predictions = {
    "Constant": constant_pred,
    "Log-linear": ridge_pred,
    "Random Forest": forest_pred,
    "MLP": mlp_pred
}

print("\n=== HELD-OUT TEST RESULTS ===")

for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    relative_error = np.mean(
        np.abs(pred - y_test) / y_test
    ) * 100

    print(
        f"{name:14s} | "
        f"MAE={mae:.6f} ms | "
        f"Mean relative error={relative_error:.2f}%"
    )

# Save per-workload predictions
with open(
    RESULT_DIR / "final_test_predictions.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "M", "N", "K", "actual_ms",
        "constant_ms", "loglinear_ms",
        "forest_ms", "mlp_ms"
    ])

    for i in range(len(X_test)):
        writer.writerow([
            *X_test[i].astype(int),
            y_test[i],
            constant_pred[i],
            ridge_pred[i],
            forest_pred[i],
            mlp_pred[i]
        ])

print("\nFinal predictions saved.")
