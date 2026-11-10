# Case Study: Profiler-Informed Triton Tile Change on the RTX 5090

## Purpose and scope

Demonstrate a reproducible **evidence → hypothesis → correctness → measurement → decision** workflow on a single GPU matrix multiplication. This is a controlled, predefined configuration comparison inspired by diagnostic-grounded optimization ideas; it is **not** autonomous source-code rewriting or a reproduction of the full OPTIMAS framework.

## Setup

- **GPU:** NVIDIA GeForce RTX 5090, compute capability 12.0, 170 SMs reported by Nsight Compute.
- **Software:** PyTorch 2.15.0+cu132, Triton 3.9.0, Nsight Compute 2026.4.0.
- **Workload:** `M=512`, `N=1024`, `K=256`; FP16 input and output with FP32 accumulation inside the Triton kernel.
- **Config A (baseline):** `(BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4)`.
- **Config B (candidate):** `(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4)`.
- **Only changed parameter:** `BLOCK_M`, from 32 to 64.

## 1. Observed hardware evidence

The previously collected `ncu --set basic` reports contain:

| Nsight metric | A | B |
| --- | ---: | ---: |
| Grid size (blocks) | 256 | 128 |
| Threads/block | 128 | 128 |
| Registers/thread | 54 | 86 |
| Dynamic shared memory/block | 12.29 KB | 16.38 KB |
| Theoretical occupancy | 58.33% | 41.67% |
| Achieved occupancy | 12.51% | 8.31% |
| Compute (SM) throughput utilization | 27.53% | 27.42% |
| Memory throughput utilization | 45.47% | 30.77% |
| DRAM throughput utilization | 9.67% | 9.80% |
| Nsight profiled duration | 4.74 µs | 4.74 µs |

Both grids are small relative to the device. B's 128 blocks are fewer than the 170 SMs, potentially leaving some SMs without blocks. B also requires more registers and shared memory per block. The profiler identifies low achieved occupancy and limited grid size as concerns, **but low occupancy alone is not proof of longer runtime**.

## 2. Hypothesis and model recommendation

Larger output tiles may reduce scheduling work or improve reuse, potentially offsetting greater per-block resource use. This is a **plausible hypothesis**, not a hardware mechanism demonstrated by the current `basic` counters. Additional traffic, stall and instruction diagnostics would be needed for causal analysis.

The **frozen** trained MLP predicted **A = 4.144 µs** and **B = 4.401 µs**, preferring A. Model preference was treated as supporting information and **not** as an unconditional optimization decision.

## 3. Correctness and independent measurement

The `evidence_guided_optimizer.py` script:

1. Read the saved Nsight reports and the existing MLP checkpoint.
2. Tested the predeclared A→B tile change, keeping other configuration settings fixed.
3. Compared both Triton outputs against PyTorch with numerical tolerances on **three input seeds**; all correctness tests passed.
4. Executed **five paired CUDA Graph timing rounds** under matching workload and dtype conditions.
5. Saved the raw timings and a structured result record.

Measured output:

| Quantity | A | B |
| --- | ---: | ---: |
| Median CUDA Graph latency | 3.098 µs | **2.768 µs** |
| Result | Correct | Correct |

Calculated speedup of B against A:

`3.098 / 2.768 ≈ 1.119×` (roughly **10.65% less latency**).

B was faster in **all five paired rounds**. These timings are **not interchangeable** with Nsight's instrumented 4.74 µs profiler durations.

## 4. Decision rule and outcome

The script's predeclared practical acceptance rule was:

- Median speedup of B vs A at least `1.02×`.
- B faster than A in at least `4 of 5` paired timing rounds.
- Numerical correctness tests must pass.

**Outcome:** `ACCEPT_B` for this specific workload and comparison. This is an experimental policy decision, **not a statistical significance test or evidence of global optimality**.

## 5. Interpretation and limits

This case highlights a mismatch between (a) an MLP recommendation, (b) occupancy as a single performance indicator, and (c) observed latency. The data supports choosing B for the tested conditions, but does **not** isolate why B is faster. In particular, no claim is made that lower occupancy causes speedup or that B is best for other shapes.

The pipeline was not asked to autonomously discover the optimization. It evaluated a **predefined** change and preserved evidence, uncertainty, and execution outcomes. Future studies could add deeper memory/warp-stall profiling, confidence-aware decisions, online tuning-cost accounting, and controlled candidate generation.

## Saved experiment artifacts

- `milestone-04-performance-ml/results/profile_A.ncu-repz`
- `milestone-04-performance-ml/results/profile_B.ncu-repz`
- `milestone-04-performance-ml/results/profile_A_details.txt`
- `milestone-04-performance-ml/results/profile_B_details.txt`
- `milestone-04-performance-ml/results/evidence_guided_timings.csv`
- `milestone-04-performance-ml/results/evidence_guided_report.json`

Related methods and broader results: [research report](research_report.md) and [Milestone 4 README](../milestone-04-performance-ml/README.md).
