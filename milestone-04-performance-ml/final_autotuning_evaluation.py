"""Frozen, offline held-out evaluation of Triton configuration selection.

Run ONCE after choosing models/policies on validation data.
Does not train or retune models. Requires compare_search_strategies.py,
model checkpoints, and data/autotuning/{train,val,test}.csv.
"""
import csv
import random
import statistics

from compare_search_strategies import (
    RESULT_DIR, SEED, MONTE_CARLO_RUNS,
    config_key, group_rows, latency, load_predictions,
    measure_selected, read_rows, shape_key, summarize,
)


def main():
    train_rows = read_rows("train.csv")
    val_rows = read_rows("val.csv")
    test_rows = read_rows("test.csv")

    train = group_rows(train_rows)
    val = group_rows(val_rows)
    test = group_rows(test_rows)
    assert (len(train), len(val), len(test)) == (84, 18, 18)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))

    configs = sorted({config_key(r) for r in train_rows})
    assert len(configs) == 4
    assert all({config_key(r) for r in group} == set(configs)
               for group in list(val.values()) + list(test.values()))

    # FIXED POLICY: derive using training measurements alone.
    # This repeats the exact rule in compare_search_strategies.py.
    fixed_cfg = min(
        configs,
        key=lambda cfg: statistics.mean(
            measure_selected(group, next(r for r in group if config_key(r) == cfg))["regret"]
            for group in train.values()
        ),
    )

    # Frozen models: no fitting, checkpoint selection, or hyperparameter changes.
    rf_pred, mlp_pred = load_predictions(test_rows)
    pred_by_pair = {
        (shape_key(row), config_key(row)): (rf_pred[i], mlp_pred[i])
        for i, row in enumerate(test_rows)
    }

    fixed_records, rf_records, mlp_records, oracle_records = [], [], [], []
    decision_rows = []
    for shape, group in test.items():
        fixed = next(r for r in group if config_key(r) == fixed_cfg)
        rf = min(group, key=lambda r: pred_by_pair[(shape, config_key(r))][0])
        mlp = min(group, key=lambda r: pred_by_pair[(shape, config_key(r))][1])
        oracle = min(group, key=latency)

        fixed_records.append(measure_selected(group, fixed))
        rf_records.append(measure_selected(group, rf))
        mlp_records.append(measure_selected(group, mlp))
        oracle_records.append(measure_selected(group, oracle))

        for policy, row in (("Fixed", fixed), ("Random Forest", rf),
                            ("MLP", mlp), ("Offline oracle", oracle)):
            metrics = measure_selected(group, row)
            decision_rows.append({
                "policy": policy,
                "M": shape[0], "N": shape[1], "K": shape[2],
                "config": str(config_key(row)),
                "selected_latency_ms": latency(row),
                "best_latency_ms": latency(oracle),
                "regret_percent": metrics["regret"],
                "within_5": int(metrics["within_5"]),
            })

    results = [
        summarize("Fixed (selected on train)", fixed_records, 1),
        summarize("Random Forest recommendation", rf_records, 1),
        summarize("MLP recommendation", mlp_records, 1),
        summarize("Exhaustive (offline oracle)", oracle_records, 4),
    ]

    # Simulate random choices from the same four PREMEASURED configs.
    # These are lookups, not new GPU benchmarks or measured search time.
    rng = random.Random(SEED)
    for budget in (1, 2, 3):
        sampled_records = []
        for _ in range(MONTE_CARLO_RUNS):
            for group in test.values():
                candidates = rng.sample(group, budget)
                sampled_records.append(
                    measure_selected(group, min(candidates, key=latency))
                )
        results.append(summarize(
            f"Random search k={budget} (simulation)", sampled_records, budget
        ))

    print("Frozen, held-out OFFLINE policy evaluation (18 test shapes)")
    print("Training-selected fixed configuration:", fixed_cfg)
    print("Strategy                          Eval  Mean regret  Within 5%")
    for r in results:
        print(f"{r['strategy'][:33]:33s} {r['evaluations_per_workload']:>4d} "
              f"{r['mean_regret_percent']:>10.2f}% "
              f"{r['within_5_rate_percent']:>9.1f}%")

    summary_path = RESULT_DIR / "search_policy_test_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    decision_path = RESULT_DIR / "search_policy_test_decisions.csv"
    with decision_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)

    print("Saved:", summary_path)
    print("Saved:", decision_path)
    print("OFFLINE evaluation only. No online tuning-time claim.")
    print("Do not retune models based on these test results.")


if __name__ == "__main__":
    main()
