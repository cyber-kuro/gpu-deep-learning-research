# Milestone 4 — ML-Guided GPU Performance Prediction and Autotuning

A research prototype studying whether machine learning can predict GPU matrix-multiplication latency and select efficient Triton kernel configurations. It includes a controlled, profiling-informed optimization case study inspired by the evidence-and-validation principles of **OPTIMAS**; it is **not** a reproduction of the full OPTIMAS system.

## Research questions

1. How accurately can conventional ML and a small neural network predict GPU execution latency for unseen matrix shapes?
2. How does training-data availability affect prediction error?
3. Can a learned model choose a near-optimal Triton configuration for a new matrix shape more effectively than fixed or random selection?
4. Can profiler evidence, a proposed configuration change, correctness tests, and fresh timings form a transparent optimization decision?

## Hardware and software

Experiments reported here used an **NVIDIA GeForce RTX 5090** (compute capability 12.0) running under **WSL2 Ubuntu**, **PyTorch 2.15.0+cu132**, **CUDA build 13.2**, **Triton 3.9.0**, and **Nsight Compute 2026.4.0**. The PyTorch latency-prediction study used FP32 with `torch.set_float32_matmul_precision("highest")`; Triton configuration studies used FP16 inputs with FP32 accumulation. Their runtime measurements are **not directly comparable** because both precision and timing methodology differ.

## What was implemented

### Part A — GPU performance data

- `collect_data.py`: PyTorch matrix-multiplication measurements for **120 distinct `(M, N, K)` shapes × 5 trial batches = 600 batch-median records**. Each batch used warmups and CUDA-event timings.
- `prepare_dataset.py`: aggregates repeats into workload-level median latency, measurement variability, and deterministic **84 train / 18 validation / 18 test** shape splits.
- `check_data_quality.py`: audits latency variability. In the original PyTorch dataset, **49/120 workloads** had coefficient of variation above 10% across batches. These cases were retained, not silently discarded.

### Part B — Latency prediction and data efficiency

- `train_baselines.py`: constant, log-linear Ridge, and Random Forest predictors.
- `train_mlp.py`: a small PyTorch MLP with training-only feature scaling and log-latency targets.
- `data_efficiency.py`: MLP experiments with 10%, 25%, 50%, and 100% of training shapes, repeated across three seeds.
- `final_evaluation.py`: final comparison on the frozen 18-shape test split.

**Held-out PyTorch latency-prediction results:**

| Model | Test MAE (ms) | Mean relative error |
| --- | ---: | ---: |
| Constant predictor | 0.027749 | 30.15% |
| Log-linear Ridge | 0.013174 | 16.57% |
| Random Forest | **0.010719** | **11.56%** |
| PyTorch MLP | 0.017672 | 32.02% |

The Random Forest was strongest on this test set. This finding is specific to the measured data and should not be interpreted as a general comparison of model families.

**Data-efficiency results (validation MAE averaged over three seeds):**

| Training shapes | Mean MAE (ms) | Across-seed std. (ms) |
| ---: | ---: | ---: |
| 8 (10%) | 0.040945 | 0.009177 |
| 21 (25%) | 0.024434 | 0.013588 |
| 42 (50%) | 0.011739 | 0.003029 |
| 84 (100%) | 0.011672 | 0.002624 |

Similar average errors for 42 and 84 shapes are a **preliminary observation**, not proof that half the data is sufficient in general.

### Part C — Triton kernels and learned configuration selection

- `triton_matmul.py`: configurable tiled Triton matrix multiplication with masking and numerical correctness checks.
- `benchmark_triton.py`: CUDA Graph-based latency measurements for **120 shapes × 4 candidate configurations × 3 trials = 1,440 measurements**.
- `prepare_autotuning.py`: checks completeness and aggregates to **480 workload–configuration pairs**, partitioned by original shape: **336 train / 72 validation / 72 test pairs**.
- `train_autotuner.py`: Random Forest latency predictor and configuration recommendation.
- `train_autotuner_mlp.py`: 7-input PyTorch MLP configuration predictor.
- `compare_search_strategies.py`: offline validation comparing fixed, random, learned, and exhaustive policies.
- `final_autotuning_evaluation.py`: **frozen offline** evaluation on 18 held-out shapes.

The four candidates were `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)` = `(32,64,32,4)`, `(64,64,32,4)`, `(64,128,32,4)`, `(64,128,64,8)`.

**Held-out offline configuration-selection results:**

| Policy | Candidate evaluations* | Mean selection regret | Within 5% of measured best |
| --- | ---: | ---: | ---: |
| MLP recommendation | 1 | **3.12%** | **83.3%** |
| Random Forest recommendation | 1 | 3.95% | 72.2% |
| Training-selected fixed configuration | 1 | 9.19% | 66.7% |
| Random search, budget 1 | 1 | 45.29% | 40.6% |
| Random search, budget 2 | 2 | 19.34% | 66.9% |
| Random search, budget 3 | 3 | 5.28% | 84.7% |
| Exhaustive oracle, four choices | 4 | 0.00% | 100% |

\*The counts are **simulated candidate-selection budgets on premeasured results**, not measured end-to-end online costs. Random results are simulation averages. The 0%-regret oracle is the best of **four candidates**, not a globally optimal GPU implementation.

### Part D — Profile-informed optimization case study

- `profile_triton.py`: correctness checks and targeted NVIDIA Nsight Compute profiling.
- `evidence_guided_optimizer.py`: reads saved profiling evidence and frozen MLP predictions, tests a **predefined A→B tile change**, checks correctness with three input seeds, benchmarks in five paired rounds, and logs the resulting decision.

For `(M,N,K) = (512,1024,256)`, A = `(32,64,32,4)` and B = `(64,64,32,4)`:

| Evidence | A | B |
| --- | ---: | ---: |
| Nsight achieved occupancy | 12.51% | 8.31% |
| Nsight registers/thread | 54 | 86 |
| Nsight grid size | 256 blocks | 128 blocks |
| Fresh CUDA Graph median latency | 3.098 µs | **2.768 µs** |

Both passed correctness across three input seeds. B was faster in **5/5 paired rounds**, producing a **1.119×** measured speedup versus A in this controlled run. The frozen MLP instead predicted A to be faster. Nsight displayed **4.74 µs for both profiled launches**, which must not be confused with the CUDA Graph timings. The physical reason for B's advantage remains **a hypothesis**, not established causality.

See [the full case study](../docs/optimization_case_study.md) and [research report](../docs/research_report.md).

## Reproducing the workflow

Run from `milestone-04-performance-ml/` in a compatible CUDA/PyTorch/Triton environment. These commands describe the existing scripts; collecting fresh measurements will take additional time and may produce different values.

```bash
# Part A — FP32 PyTorch workload measurements and splits
python3 collect_data.py
python3 prepare_dataset.py
python3 check_data_quality.py

# Part B — Baselines, MLP, data efficiency, final evaluation
python3 train_baselines.py
python3 train_mlp.py
python3 data_efficiency.py
python3 final_evaluation.py

# Part C — Triton configurations (set SMOKE_TEST appropriately
# inside benchmark_triton.py before running the full dataset)
python3 triton_matmul.py
python3 benchmark_triton.py
python3 prepare_autotuning.py
python3 train_autotuner.py
python3 train_autotuner_mlp.py
python3 compare_search_strategies.py
python3 final_autotuning_evaluation.py

# Part D — Existing profiling reports and controlled experiment
python3 profile_triton.py A
python3 profile_triton.py B
python3 evidence_guided_optimizer.py
```

Nsight Compute profiling reports were obtained separately using `ncu --set basic`; collection requires a supported NVIDIA driver and permission to access GPU performance counters. The evidence-guided script expects the relevant saved `.ncu-repz` reports and MLP checkpoint. When running the train/test evaluation sequence, do not use test results to change models or hyperparameters.

## Key output artifacts

- `gpu_performance_data.csv`, `data/processed/`: FP32 PyTorch study and workload splits.
- `triton_autotuning_data.csv`, `data/autotuning/`: FP16 Triton configuration study.
- `results/search_policy_validation_summary.csv`, `results/search_policy_test_summary.csv`: offline policy comparisons.
- `results/rf_autotuner.joblib`, `results/mlp_autotuner.pt`: saved model artifacts. Load locally produced/trusted serialized models only.
- `results/profile_A.ncu-repz`, `results/profile_B.ncu-repz`: Nsight hardware reports.
- `results/evidence_guided_report.json`, `results/evidence_guided_timings.csv`: final A/B case study evidence.

## Limitations and next research directions

This is a **single-GPU**, **single kernel-family**, **four-configuration** study with 18 test workload shapes. Runtime labels and recommendation rankings can be sensitive to timing noise and near ties. The policy comparison is **offline**: it does not demonstrate lower end-to-end tuning time once model training, inference, compilation, and profiling overheads are included. The final profile-informed demonstration uses a **predefined configuration change**, not autonomous code rewriting or an OPTIMAS multi-agent system.

Potential follow-ups include larger configuration spaces, direct ranking losses instead of latency regression, uncertainty-aware selection, cross-GPU generalization, controlled online tuning-cost measurement, and more detailed GPU stall/memory evidence.

## Research relationship

This independent learning project is motivated by performance modeling and evidence-based GPU optimization research, including the Per4ML group's work and **OPTIMAS: An Intelligent Analytics-Informed Generative AI Framework for Performance Optimization** ([preprint](https://arxiv.org/abs/2604.23892)). No affiliation, direct reproduction, or endorsement is implied.
