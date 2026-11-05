
# Milestone 2: GPU Architecture and Deep Learning Fundamentals

## 1. What happens when PyTorch executes matrix multiplication on the GPU?

When PyTorch performs matrix multiplication on a GPU, it typically uses highly optimized libraries such as NVIDIA's cuBLAS instead of the naive matrix multiplication algorithm.

In a naive GPU implementation, each thread may be assigned to calculate one element of the output matrix. To do this, the thread repeatedly accesses global memory to fetch input values and performs a series of fused multiply-add (FMA) operations.

However, different threads may need some of the same input values, causing redundant global memory accesses. For large matrices, this can create significant memory traffic and potentially make the program memory-bound.

We can improve this using techniques such as tiling and shared-memory reuse, which reduce unnecessary global memory accesses and improve computational efficiency.

## 2. What is a warp?

A warp is a group of 32 threads on an NVIDIA GPU. These threads are scheduled together and execute instructions using the SIMT (Single Instruction, Multiple Threads) execution model.

Warps allow the GPU to efficiently execute many threads in parallel.

## 3. Why is GPU memory behavior important?

GPUs have access to several types of memory, including global memory, shared memory, registers, and caches. Each has different characteristics in terms of access latency, capacity, and how the data can be accessed.

Registers are very fast and private to individual threads. Shared memory is also fast and allows threads within the same block to share data. Global memory offers much larger capacity but generally has higher access latency.

Using these memory types appropriately is important for GPU performance. If a program repeatedly fetches data from global memory instead of reusing data from faster memory, it can create unnecessary memory traffic and become a performance bottleneck.

Therefore, efficient memory access and data reuse are important parts of GPU optimization.

## 4. What is tiling, and why does it help?

Tiling is an optimization technique that divides a computational workload, such as matrix multiplication, into smaller blocks called tiles.

Instead of repeatedly fetching the same input data from global memory, a tiled GPU kernel can load smaller portions of the input matrices into shared memory. Threads within a block can then reuse these values multiple times while performing their calculations.

After completing the calculations for the current tiles, the program loads the next portions of the matrices and repeats the process.

The main advantage is that data can be reused from faster, on-chip memory rather than repeatedly accessing global memory. This reduces redundant global memory accesses and can improve overall performance.

However, the actual improvement depends on factors such as tile size, memory access patterns, workload size, and synchronization overhead.

## 5. Why is a GPU not necessarily faster for every operation?

GPU performance depends on several factors, including memory access, data reuse, parallelism, instruction efficiency, synchronization, and resource usage.

Although GPUs are designed to execute many operations in parallel, they also have overhead associated with launching kernels and potentially transferring data between the CPU and GPU.

For example, if we only need to perform a small operation, such as adding two arrays containing 10 elements, the CPU might complete the task faster because the overhead of using the GPU can outweigh the computational benefits.

However, for larger workloads with enough parallelism, GPUs can provide significant performance improvements.

Therefore, we cannot assume that a GPU is always faster without considering the workload, implementation, and execution overhead.
