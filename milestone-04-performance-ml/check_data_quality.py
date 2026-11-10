import csv
import statistics

with open("data/processed/workload_summary.csv", "r") as file:
    rows = list(csv.DictReader(file))

# Convert numerical fields
for row in rows:
    row["M"] = int(row["M"])
    row["N"] = int(row["N"])
    row["K"] = int(row["K"])
    row["cv_percent"] = float(row["cv_percent"])
    row["median_latency_ms"] = float(row["median_latency_ms"])

# Sort by highest variability first
rows.sort(key=lambda row: row["cv_percent"], reverse=True)

print("=== TOP 10 HIGHEST VARIABILITY WORKLOADS ===")

for row in rows[:10]:
    print(
        f"M={row['M']}, "
        f"N={row['N']}, "
        f"K={row['K']}, "
        f"Latency={row['median_latency_ms']:.5f} ms, "
        f"CV={row['cv_percent']:.2f}%"
    )

all_cv = [row["cv_percent"] for row in rows]

print("\nMedian CV:", statistics.median(all_cv))
print("Maximum CV:", max(all_cv))
print("Workloads with CV > 10%:", sum(cv > 10 for cv in all_cv))
