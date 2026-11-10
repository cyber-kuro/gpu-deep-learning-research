"""Offline validation comparison of fixed, random, ML-guided and exhaustive
Triton configuration selection. Does NOT load held-out test data.

Run after train_autotuner.py and train_autotuner_mlp.py have produced their saved models."""
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "autotuning"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

FEATURES = ["M", "N", "K", "block_m", "block_n", "block_k", "num_warps"]
SHAPE_FIELDS = ["M", "N", "K"]
CONFIG_FIELDS = ["block_m", "block_n", "block_k", "num_warps"]
MONTE_CARLO_RUNS = 2000  # Simulated random selections, NOT fresh GPU benchmarks
SEED = 42


def read_rows(filename):
    with (DATA_DIR / filename).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {filename}")
    return rows


def shape_key(row):
    return tuple(int(row[field]) for field in SHAPE_FIELDS)


def config_key(row):
    return tuple(int(row[field]) for field in CONFIG_FIELDS)


def latency(row):
    return float(row["median_latency_ms"])


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[shape_key(row)].append(row)
    for shape, group in grouped.items():
        configs = [config_key(r) for r in group]
        if len(group) != 4 or len(set(configs)) != 4:
            raise ValueError(f"Expected four unique configurations for {shape}")
    return dict(sorted(grouped.items()))


def feature_matrix(rows):
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float64)
    if not np.isfinite(X).all() or (X <= 0).any():
        raise ValueError("Features must be positive and finite for log2 preprocessing")
    return np.log2(X)


class AutotunerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_predictions(rows):
    X_log = feature_matrix(rows)

    # Load only artifacts generated locally by your own scripts.
    rf_saved = joblib.load(RESULT_DIR / "rf_autotuner.joblib")
    if rf_saved["features"] != FEATURES:
        raise ValueError("Random Forest feature order does not match")
    rf_pred = np.exp(rf_saved["model"].predict(X_log))

    checkpoint = torch.load(
        RESULT_DIR / "mlp_autotuner.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["features"] != FEATURES:
        raise ValueError("MLP feature order does not match")
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    X_scaled = (X_log - mean) / scale
    mlp = AutotunerMLP()
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()
    with torch.no_grad():
        mlp_pred = torch.exp(
            mlp(torch.tensor(X_scaled, dtype=torch.float32))
        ).numpy().reshape(-1)

    if not np.isfinite(rf_pred).all() or not np.isfinite(mlp_pred).all():
        raise ValueError("Model predictions must be finite")
    return rf_pred, mlp_pred


def regret_pct(selected_ms, best_ms):
    return 100.0 * (selected_ms / best_ms - 1.0)


def measure_selected(group, chosen):
    best = min(latency(r) for r in group)
    picked = latency(chosen)
    regret = regret_pct(picked, best)
    return {
        "regret": regret,
        "within_5": float(regret <= 5.0 + 1e-9),
        "exact_best": float(abs(picked - best) <= 1e-12),
    }


def summarize(name, records, benchmark_evaluations):
    return {
        "strategy": name,
        "workloads": len(records),
        "evaluations_per_workload": benchmark_evaluations,
        "mean_regret_percent": statistics.mean(r["regret"] for r in records),
        "median_regret_percent": statistics.median(r["regret"] for r in records),
        "exact_best_rate_percent": 100.0 * statistics.mean(r["exact_best"] for r in records),
        "within_5_rate_percent": 100.0 * statistics.mean(r["within_5"] for r in records),
    }


def main():
    train_rows = read_rows("train.csv")
    val_rows = read_rows("val.csv")
    train = group_rows(train_rows)
    val = group_rows(val_rows)
    assert len(train) == 84 and len(val) == 18
    assert not set(train).intersection(val)
    assert len(train_rows) == 336 and len(val_rows) == 72

    configs = {config_key(r) for r in train_rows}
    if any({config_key(r) for r in group} != configs for group in val.values()):
        raise ValueError("Train and validation configuration spaces differ")

    # Choose one fixed configuration using TRAINING measurements only.
    train_fixed_scores = {}
    for cfg in sorted(configs):
        train_fixed_scores[cfg] = statistics.mean(
            measure_selected(group, next(r for r in group if config_key(r) == cfg))["regret"]
            for group in train.values()
        )
    fixed_cfg = min(train_fixed_scores, key=train_fixed_scores.get)
    print("Training-selected fixed configuration:", fixed_cfg)

    rf_pred, mlp_pred = load_predictions(val_rows)
    predicted = {}
    for i, row in enumerate(val_rows):
        predicted[(shape_key(row), config_key(row))] = (rf_pred[i], mlp_pred[i])

    results = []
    fixed_records, rf_records, mlp_records, oracle_records = [], [], [], []
    for shape, group in val.items():
        fixed_records.append(measure_selected(
            group, next(r for r in group if config_key(r) == fixed_cfg)
        ))
        rf_chosen = min(group, key=lambda r: predicted[(shape, config_key(r))][0])
        mlp_chosen = min(group, key=lambda r: predicted[(shape, config_key(r))][1])
        rf_records.append(measure_selected(group, rf_chosen))
        mlp_records.append(measure_selected(group, mlp_chosen))
        oracle_records.append(measure_selected(group, min(group, key=latency)))

    results.extend([
        summarize("Fixed (selected on train)", fixed_records, 1),
        summarize("Random Forest recommendation", rf_records, 1),
        summarize("MLP recommendation", mlp_records, 1),
        summarize("Exhaustive (offline oracle)", oracle_records, 4),
    ])

    # Offline simulations. Each chosen configuration is looked up in the
    # previously measured CSV, not executed again on the GPU.
    rng = random.Random(SEED)
    for budget in (1, 2, 3):
        sampled_records = []
        for _ in range(MONTE_CARLO_RUNS):
            for group in val.values():
                candidates = rng.sample(group, budget)
                sampled_records.append(measure_selected(group, min(candidates, key=latency)))
        results.append(summarize(f"Random search k={budget} (simulation)",
                                 sampled_records, budget))

    print("\n=== OFFLINE VALIDATION POLICY COMPARISON ===")
    print("18 unseen validation shapes, four premeasured configs per shape")
    print("Strategy                          Eval   Mean regret   Within 5%")
    for r in results:
        print(f"{r['strategy'][:33]:33s} {r['evaluations_per_workload']:>4d}  "
              f"{r['mean_regret_percent']:>10.2f}%  {r['within_5_rate_percent']:>9.1f}%")

    path = RESULT_DIR / "search_policy_validation_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved:", path)
    print("NOTE: Offline lookup/simulation only. No online tuning-time claim.")
    print("Held-out test.csv was not read.")


if __name__ == "__main__":
    main()
