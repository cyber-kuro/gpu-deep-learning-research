import csv
import numpy as np
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"


# Read a dataset split
def load_data(filename):
    with open(DATA_DIR / filename, "r") as file:
        rows = list(csv.DictReader(file))

    # Inputs: workload dimensions
    X = np.array([
        [float(row["M"]), float(row["N"]), float(row["K"])]
        for row in rows
    ])

    # Target: measured median latency
    y = np.array([
        float(row["median_latency_ms"])
        for row in rows
    ])

    return X, y


# Load training and validation workloads
X_train, y_train = load_data("train.csv")
X_val, y_val = load_data("val.csv")

print("Training workloads:", len(X_train))
print("Validation workloads:", len(X_val))


######### Model 1: Constant baseline #########
constant_model = DummyRegressor(strategy="median")
constant_model.fit(X_train, y_train)

constant_predictions = constant_model.predict(X_val)

constant_mae = mean_absolute_error(
    y_val, constant_predictions
)


######### Model 2: Log-linear regression #########
# Log transforms help model multiplicative scaling
X_train_log = np.log2(X_train)
X_val_log = np.log2(X_val)

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Train to predict log latency
model.fit(X_train_log, np.log(y_train))

# Predict log latency and convert back to milliseconds
predictions = np.exp(model.predict(X_val_log))

regression_mae = mean_absolute_error(
    y_val, predictions
)


print("\n=== VALIDATION RESULTS ===")
print(f"Constant baseline MAE: {constant_mae:.6f} ms")
print(f"Log-linear regression MAE: {regression_mae:.6f} ms")

print("\n=== SAMPLE PREDICTIONS USING LOG-LINEAR REGRESSION ===")
for actual, predicted in zip(y_val[:5], predictions[:5]):
    print(
        f"Actual: {actual:.5f} ms | "
        f"Predicted: {predicted:.5f} ms"
    )

######### Model 3: Random Forest Regression #########

forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train using only training workloads
forest_model.fit(X_train, np.log(y_train))

# Predict validation latency
forest_predictions = np.exp(
    forest_model.predict(X_val)
)

# Evaluate using MAE
forest_mae = mean_absolute_error(
    y_val,
    forest_predictions
)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Random Forest MAE: {forest_mae:.6f} ms")

print("\n=== MODEL COMPARISON ===")

print(f"Constant baseline: {constant_mae:.6f} ms")
print(f"Log-linear regression: {regression_mae:.6f} ms")
print(f"Random forest: {forest_mae:.6f} ms")