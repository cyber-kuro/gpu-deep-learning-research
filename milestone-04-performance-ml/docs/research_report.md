# Research Report: Data-Efficient GPU Performance Prediction and Evidence-Guided Optimization

**Type:** Independent research prototype
**Platform:** NVIDIA GeForce RTX 5090 (compute capability 12.0), WSL2 Ubuntu
**Key tools:** PyTorch 2.15.0+cu132, Triton 3.9.0, CUDA build 13.2, NVIDIA Nsight Compute 2026.4.0

## Abstract

This project examines two linked problems in GPU performance optimization: predicting execution time from workload features and recommending kernel configurations that approach the fastest of a restricted candidate set. A PyTorch matrix-multiplication dataset was used to compare constant, log-linear, Random Forest, and neural-network runtime predictors, including data-efficiency experiments. A separate FP16 Triton benchmark covered 120 workload shapes and four tile/warp configurations. An MLP-based configuration selector achieved **3.12% mean selection regret** on **18 held-out shapes**, compared with **9.19%** for a fixed configuration, in an offline comparison. A final case study combined Nsight Compute metrics, model predictions, numerical correctness checks, and fresh CUDA Graph timing to evaluate a predefined tile change, finding a **1.119×** speedup for one workload. These results demonstrate a reproducible prototype, **not** an online tuning-time improvement or reproduction of the full OPTIMAS framework.

## 1. Motivation and questions

GPU kernel performance varies with problem shape, execution configuration, and hardware resource use. Exhaustively benchmarking candidate configurations can be expensive as the search space grows. This project explores:

1. Can simple models and small neural networks predict the latency of new GPU matrix shapes?
2. How does reducing the number of observed training workloads affect model accuracy?
3. Can a learned selector recommend good Triton configurations using fewer candidate measurements than exhaustive enumeration in a restricted search space?
4. Can GPU diagnostics and execution-based validation support transparent optimization decisions, including cases where the ML model's preference is wrong?

## 2. Experimental methods

### PyTorch latency prediction

A dataset of **120 unique `(M,N,K)` shapes** was sampled reproducibly. Each shape had **five trial-batch medians**, producing **600 raw records**. Experiments used FP32 (`highest` PyTorch matmul precision) and CUDA events. Measurements were summarized at the workload level and split by unique shape into **84 training, 18 validation, and 18 held-out test** workloads. Timing variability was audited; **49 of 120 shapes** exceeded 10% coefficient of variation across batches and were retained.

Models comprised a constant median predictor, log-linear Ridge regression, Random Forest, and a small MLP trained to predict log latency. Separately, MLP training was repeated with 8, 21, 42, and 84 training workloads across three random seeds, using the same validation shapes.

### Triton configuration selection

A correctness-checked FP16 tiled matrix multiplication used four configurations `(BLOCK_M,BLOCK_N,BLOCK_K,num_warps)`:

- A `(32,64,32,4)`
- B `(64,64,32,4)`
- C `(64,128,32,4)`
- D `(64,128,64,8)`

CUDA Graph replay benchmarking produced **120 shapes × 4 configurations × 3 trials = 1,440 measurements**. Aggregation yielded **480 shape–configuration pairs**, partitioned into **336/72/72 train/validation/test pairs** with no shape overlap. A Random Forest and a 7-input, two-hidden-layer MLP predicted candidate latency. Both were evaluated as configuration-ranking policies. Fixed, random-budget, and exhaustive-oracle policies were simulated from premeasured configurations. Policy selection occurred before examining held-out test results.

### Diagnostic-guided case study

For FP16 matrix multiplication `(512,1024,256)`, Nsight Compute profiled A and B. A controlled script recorded profiler evidence and frozen MLP predictions, tested the predefined change `BLOCK_M: 32→64`, verified both outputs against PyTorch for three input seeds, measured five paired CUDA Graph trials, and saved the decision and timing data. The script does **not** independently synthesize new kernel code.

## 3. Results

### 3.1 Held-out PyTorch latency prediction

| Method | Test MAE (ms) | Mean relative error |
| --- | ---: | ---: |
| Constant | 0.027749 | 30.15% |
| Log-linear Ridge | 0.013174 | 16.57% |
| **Random Forest** | **0.010719** | **11.56%** |
| MLP | 0.017672 | 32.02% |

Random Forest was best for this dataset and test split. The result does not establish that one model family is universally better.

### 3.2 Data efficiency (MLP validation)

| Training shapes | Mean MAE (ms), 3 seeds | Std. (ms) |
| ---: | ---: | ---: |
| 8 | 0.040945 | 0.009177 |
| 21 | 0.024434 | 0.013588 |
| 42 | 0.011739 | 0.003029 |
| 84 | 0.011672 | 0.002624 |

The similar 42- and 84-shape means motivate further study, but only three seeds and one validation split were evaluated.

### 3.3 Held-out offline Triton selection

| Strategy | Premeasured candidates per decision | Mean regret | Within 5% of measured best |
| --- | ---: | ---: | ---: |
| **MLP recommendation** | **1** | **3.12%** | **83.3%** |
| Random Forest recommendation | 1 | 3.95% | 72.2% |
| Training-selected fixed config | 1 | 9.19% | 66.7% |
| Random search, 1 candidate | 1 | 45.29% | 40.6% |
| Random search, 2 candidates | 2 | 19.34% | 66.9% |
| Random search, 3 candidates | 3 | 5.28% | 84.7% |
| Exhaustive oracle (4) | 4 | 0.00% | 100% |

**Selection regret** is `100 × (selected measured latency / best measured candidate latency − 1)`. On these 18 test shapes, the MLP's mean regret was about **66% lower relative to the fixed policy's mean regret**. However, random search with three candidates had a slightly higher within-5% rate; different metrics give different views. The comparison is **offline** and does not include end-to-end tuning costs.

### 3.4 Profiling-informed A/B case study

| Metric | A `(32,64,32,4)` | B `(64,64,32,4)` |
| --- | ---: | ---: |
| Grid size | 256 | 128 |
| Registers/thread | 54 | 86 |
| Dynamic shared memory/block | 12.29 KB | 16.38 KB |
| Achieved occupancy | 12.51% | 8.31% |
| MLP predicted latency | 4.144 µs | 4.401 µs |
| Fresh CUDA Graph median latency | 3.098 µs | **2.768 µs** |
| Nsight profiled duration | 4.74 µs | 4.74 µs |

Despite A's higher occupancy and the MLP preference for A, B was faster in all five paired timing rounds. B's **1.119× speedup** relative to A was accepted under a predeclared practical decision rule (at least 2% lower median latency and faster in at least four of five rounds). This is a repeatable, controlled observation rather than proof of a specific causal performance mechanism. Profiling-instrumented duration must not be equated with CUDA Graph latency.

## 4. Limitations

- One RTX 5090 and one kernel family; no cross-architecture validation.
- Four candidate configurations; the oracle is only the best **of these four**.
- Only 18 held-out matrix shapes, with sensitivity to close runtime ties.
- Runtime-prediction and tuning studies use different data types and timing conditions.
- Offline policy simulations do not measure inference, compilation, profiling, or end-to-end search overhead.
- Profiling evidence supports a resource-use hypothesis but does not isolate why B is faster.
- The final decision pipeline tests a predefined change and is not a complete LLM multi-agent optimization system.

## 5. Next research directions

A natural progression is to (i) expand workloads and kernel configurations, (ii) optimize a ranking or regret loss instead of latency prediction alone, (iii) quantify prediction uncertainty and close-tie sensitivity, (iv) measure **online** tuning cost and break-even reuse, and (v) test transfer to other GPU architectures. Hardware-counter and stall-level evidence could then guide a more general candidate-generation process.

## 6. Relationship to Per4ML and OPTIMAS

This independent prototype follows themes of learning-based HPC performance modeling and diagnostic-grounded GPU optimization associated with the Per4ML research program. It is inspired by, but **does not replicate**, **OPTIMAS: An Intelligent Analytics-Informed Generative AI Framework for Performance Optimization** ([preprint](https://arxiv.org/abs/2604.23892)). The immediate research question is whether learned configuration-ranking policies can remain effective when performance data is limited and execution feedback is required to validate recommendations.

## Reproducibility

See [`milestone-04-performance-ml/README.md`](../milestone-04-performance-ml/README.md) for script entry points, dataset sizes, software versions, and artifact names. The final experiment's saved evidence is in `results/evidence_guided_report.json`, `results/evidence_guided_timings.csv`, and the two Nsight profiling reports.
