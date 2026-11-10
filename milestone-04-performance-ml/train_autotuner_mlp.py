import csv
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "autotuning"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "M", "N", "K",
    "block_m", "block_n", "block_k",
    "num_warps"
]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Load datasets
# ==========================================

def load_rows(filename):
    with open(DATA_DIR / filename, newline="") as f:
        return list(csv.DictReader(f))


train_rows = load_rows("train.csv")
val_rows = load_rows("val.csv")

assert len(train_rows) == 336
assert len(val_rows) == 72

train_shapes = {
    (r["M"], r["N"], r["K"])
    for r in train_rows
}

val_shapes = {
    (r["M"], r["N"], r["K"])
    for r in val_rows
}

assert not train_shapes & val_shapes


def make_features(rows):
    return np.array([
        [float(r[name]) for name in FEATURES]
        for r in rows
    ], dtype=np.float64)


def make_targets(rows):
    return np.array([
        float(r["median_latency_ms"])
        for r in rows
    ], dtype=np.float64)


X_train = make_features(train_rows)
X_val = make_features(val_rows)

y_train = make_targets(train_rows)
y_val = make_targets(val_rows)

# Transform input dimensions and configurations
X_train = np.log2(X_train)
X_val = np.log2(X_val)

# Fit preprocessing only on training data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_t = torch.tensor(
    X_train, dtype=torch.float32, device=device
)

X_val_t = torch.tensor(
    X_val, dtype=torch.float32, device=device
)

y_train_t = torch.tensor(
    np.log(y_train).reshape(-1, 1),
    dtype=torch.float32,
    device=device
)


# ==========================================
# 3. Define the neural network
# ==========================================

class AutotunerMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


model = AutotunerMLP().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()


# ==========================================
# 4. Training
# ==========================================

best_mae = float("inf")
best_epoch = 0
best_state = None

patience = 100
no_improvement = 0

for epoch in range(1, 801):

    model.train()

    train_pred = model(X_train_t)
    loss = loss_fn(train_pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_pred = np.exp(
            model(X_val_t).cpu().numpy().flatten()
        )

    val_mae = mean_absolute_error(
        y_val, val_pred
    )

    if val_mae < best_mae:
        best_mae = val_mae
        best_epoch = epoch
        best_state = copy.deepcopy(
            model.state_dict()
        )
        no_improvement = 0
    else:
        no_improvement += 1

    if epoch == 1 or epoch % 100 == 0:
        print(
            f"Epoch={epoch} | "
            f"Loss={loss.item():.6f} | "
            f"Val MAE={val_mae:.6f} ms"
        )

    if no_improvement >= patience:
        print("Early stopping triggered.")
        break


# Restore best checkpoint
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    predictions = np.exp(
        model(X_val_t).cpu().numpy().flatten()
    )


# ==========================================
# 5. Evaluate configuration selection
# ==========================================

groups = defaultdict(list)

for i, row in enumerate(val_rows):
    shape = (
        int(row["M"]),
        int(row["N"]),
        int(row["K"])
    )
    groups[shape].append(i)


selection_results = []

for shape, indices in sorted(groups.items()):

    assert len(indices) == 4

    # Lowest predicted latency
    selected_idx = min(
        indices,
        key=lambda i: predictions[i]
    )

    # Lowest measured latency
    optimal_idx = min(
        indices,
        key=lambda i: y_val[i]
    )

    selected_ms = y_val[selected_idx]
    optimal_ms = y_val[optimal_idx]

    regret = (
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
        "regret_percent": regret,
        "exact_best": int(
            selected_idx == optimal_idx
        ),
        "within_5_percent": int(
            regret <= 5.0 + 1e-9
        )
    })


# ==========================================
# 6. Report results
# ==========================================

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

print("\n=== MLP AUTOTUNER RESULTS ===")
print("Training device:", device)
print("Training pairs:", len(train_rows))
print("Validation pairs:", len(val_rows))
print("Best epoch:", best_epoch)
print(f"Validation MAE: {best_mae:.6f} ms")

print("\n=== CONFIGURATION SELECTION ===")
print("Evaluated workloads:", len(selection_results))
print(f"Exact best: {exact_best}/18")
print(f"Within 5%: {near_best}/18")
print(f"Mean regret: {np.mean(regrets):.2f}%")
print(f"Median regret: {np.median(regrets):.2f}%")


# ==========================================
# 7. Save evaluation results
# ==========================================

output = RESULT_DIR / "mlp_autotuner_validation.csv"

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(selection_results[0].keys())
    )
    writer.writeheader()
    writer.writerows(selection_results)


# ==========================================
# 8. Save model checkpoint
# ==========================================

torch.save({
    "model_state_dict": {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    },
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "features": FEATURES,
    "best_epoch": best_epoch,
}, RESULT_DIR / "mlp_autotuner.pt")

print("\nMLP autotuner and results saved.")
