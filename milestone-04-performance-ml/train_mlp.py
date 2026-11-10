
import csv
import copy
import numpy as np
import torch

from pathlib import Path
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load prepared workload data
def load_data(filename):
    with open(DATA_DIR / filename, "r") as file:
        rows = list(csv.DictReader(file))

    X = np.array([
        [float(r["M"]), float(r["N"]), float(r["K"])]
        for r in rows
    ])

    y = np.array([
        float(r["median_latency_ms"])
        for r in rows
    ])

    return X, y


X_train, y_train = load_data("train.csv")
X_val, y_val = load_data("val.csv")

# Log-transform input dimensions
X_train = np.log2(X_train)
X_val = np.log2(X_val)

# Fit scaler using training data ONLY
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert data to PyTorch tensors
X_train_t = torch.tensor(
    X_train, dtype=torch.float32, device=device
)

X_val_t = torch.tensor(
    X_val, dtype=torch.float32, device=device
)

y_train_t = torch.tensor(
    np.log(y_train),
    dtype=torch.float32,
    device=device
).reshape(-1, 1)

# Define neural network
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


model = PerformanceMLP().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

best_mae = float("inf")
best_state = None
best_epoch = 0
patience = 50
epochs_without_improvement = 0

# Training loop
for epoch in range(1, 501):

    model.train()

    predictions = model(X_train_t)
    loss = loss_fn(predictions, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()

    with torch.inference_mode():
        val_log_predictions = model(X_val_t)
        val_predictions = torch.exp(
            val_log_predictions
        ).cpu().numpy().flatten()

    val_mae = mean_absolute_error(
        y_val, val_predictions
    )

    if val_mae < best_mae:
        best_mae = val_mae
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch % 50 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | "
            f"Train loss: {loss.item():.6f} | "
            f"Val MAE: {val_mae:.6f} ms"
        )

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

# Restore best validation checkpoint
model.load_state_dict(best_state)
model.eval()

print("\n=== MLP VALIDATION RESULTS ===")
print("Training device:", device)
print("Best epoch:", best_epoch)
print(f"Best validation MAE: {best_mae:.6f} ms")

# Save model and preprocessing information
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "best_epoch": best_epoch,
}, BASE_DIR / "mlp_checkpoint.pt")

print("MLP checkpoint saved.")
