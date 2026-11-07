
# GPU-Accelerated Deep Learning Research

**Exploring GPU Architecture, Performance Benchmarking, Kernel Optimization, and Deep Learning Systems**

## Overview

This repository documents my self-paced learning and experimental work in GPU-accelerated deep learning and high-performance computing.

As an undergraduate Computer Science student at Texas State University, I am developing practical skills in GPU programming, parallel computing, deep learning systems, and performance optimization, with the long-term goal of pursuing research in these areas.

The project follows a milestone-based approach, combining theoretical foundations with hands-on experiments using NVIDIA GPUs, PyTorch, CUDA, and eventually Triton.

My primary interests include:

- GPU architecture and parallel computing
- Deep learning workload optimization
- GPU kernel development and performance profiling
- Memory efficiency and computational throughput
- GPU autotuning and machine-learning-based performance prediction
- Deep learning compilers and heterogeneous computing

## Hardware and Software

My experiments are conducted locally using the following hardware and software environment.

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| CPU | AMD Ryzen 7 7800X3D |
| System RAM | 32 GB DDR5 |
| Operating System | Windows 11 with WSL2 Ubuntu |
| Programming Language | Python |
| Deep Learning Framework | PyTorch 2.15.0+cu132 |
| CUDA Build | 13.2 |
| FP32 Matmul Precision | Highest |
| Tools | Git, VS Code, Matplotlib, PyTorch Profiler |

Software versions reflect the environment used for Milestone 3.

### Milestone 01 — GPU Research Environment

Established a Python and PyTorch development environment with CUDA GPU acceleration.

**Key activities:**
- Configured the development environment.
- Verified GPU availability through PyTorch.
- Executed GPU-accelerated tensor operations.

### Milestone 02 — GPU Architecture and Tensor Fundamentals

Studied essential GPU computing concepts, including threads, warps, streaming multiprocessors, memory hierarchy, and tiling.

**Implementation:**
- Created and manipulated PyTorch tensors.
- Performed matrix multiplication on CPU and GPU.
- Verified numerical similarity between CPU and GPU results.
- Investigated device placement, tensor shapes, and floating-point data types.

**Key takeaway:** Efficient GPU execution depends on parallelism, memory access patterns, and effective data reuse.

### Milestone 03 — GPU Benchmarking and Profiling

Developed a reproducible introductory benchmarking workflow to examine how matrix multiplication performance changes with workload size on an NVIDIA RTX 5090.

**Methodology:**
- Tested square matrices ranging from 256 × 256 to 4096 × 4096.
- Used CUDA events to measure GPU execution latency.
- Performed 10 warmup iterations before measurement.
- Collected 20 timing measurements per batch across five trial batches.
- Calculated achieved throughput in TFLOP/s.
- Analyzed median latency, minimum, maximum, and sample standard deviation across batch medians.
- Used PyTorch Profiler to inspect CPU-side and GPU-side operations.
- Generated performance graphs using Matplotlib.

#### Experimental Results

| Matrix Size | Median Latency (ms) | Achieved Throughput (TFLOP/s) |
|---|---:|---:|
| 256 × 256 | 0.0322 | 1.04 |
| 512 × 512 | 0.0286 | 9.37 |
| 1024 × 1024 | 0.0666 | 32.23 |
| 2048 × 2048 | 0.3294 | 52.16 |
| 4096 × 4096 | 2.0200 | 68.04 |

*Throughput is estimated from the conventional matrix multiplication FLOP count and the median of five batch-median latencies.*

#### Performance Visualizations

**Matrix Size vs. Achieved Throughput**

![Matrix Multiplication Throughput](milestone-03-benchmarking/throughput.png)

**Matrix Size vs. Latency**

![Matrix Multiplication Latency](milestone-03-benchmarking/latency.png)

#### Key Observations

- Achieved computational throughput generally increased as matrix size increased.
- Larger matrix multiplications required more execution time despite achieving higher throughput.
- Small workloads exhibited proportionally greater sensitivity to timing variability and overhead.
- Profiling revealed the distinction between CPU-side operation dispatch and GPU kernel execution.

These results represent an introductory benchmarking study rather than an exhaustive hardware characterization. Further experiments would be required to identify specific performance bottlenecks.

For implementation details, see [Milestone 03](milestone-03-benchmarking/).

## Repository Structure

```text
gpu-deep-learning-research/
│
├── milestone-01-environment/
│   ├── gpu_check.py
│   └── README.md
│
├── milestone-02-foundations/
│   ├── tensor_experiments.py
│   └── concepts.md
│
├── milestone-03-benchmarking/
│   ├── benchmark_matmul.py
│   ├── profile_matmul.py
│   ├── summary_result.py
│   ├── plot_results.py
│   ├── matmul_results.csv
│   ├── matmul_summary.csv
│   ├── throughput.png
│   ├── latency.png
│   └── README.md
│
└── README.md
```

The repository will expand as additional milestones are completed.

## Reproducing the Benchmarks

A compatible NVIDIA GPU, CUDA-enabled PyTorch installation, and Python environment are required.

Install the relevant Python dependencies:

```bash
python -m pip install matplotlib
```

For PyTorch, follow the [official installation instructions](https://pytorch.org/get-started/locally/) to install a compatible CUDA-enabled build.

Navigate to the benchmarking directory:

```bash
cd milestone-03-benchmarking
```

Run the benchmark:

```bash
python benchmark_matmul.py
```

Generate statistical summaries:

```bash
python summary_result.py
```

Generate performance visualizations:

```bash
python plot_results.py
```

Run the GPU profiler:

```bash
python profile_matmul.py
```

Results may vary depending on GPU hardware, software versions, precision settings, and system conditions.

## Future Research Directions

### GPU Kernel Optimization

Implementing custom GPU kernels using Triton and studying how tiling, memory access patterns, and execution configurations affect performance.

### GPU Autotuning

Investigating how kernel parameters can be automatically selected for different workloads.

### Performance Modeling

Exploring whether lightweight machine learning models can predict efficient kernel configurations and reduce tuning overhead.

### Deep Learning Compilers

Understanding how compiler transformations and hardware-aware optimizations improve the execution of neural network workloads.

These are planned learning and research directions, not completed research contributions.

## References and Learning Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Triton GPU Programming](https://triton-lang.org/)
- [CoffeeBeforeArch — CUDA Programming Examples](https://github.com/CoffeeBeforeArch/cuda_programming)
- [Texas State University Compilers Research Lab](https://compilers.cs.txstate.edu/)

## Project Goals

This repository serves as a record of my technical development and experimental learning in GPU computing.

My goal is to build a strong foundation in deep learning systems, gain experience conducting reproducible performance experiments, and develop the skills necessary to contribute meaningfully to academic research in GPU optimization and high-performance computing.
