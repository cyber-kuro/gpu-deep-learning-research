import torch

A = torch.randn(2, 3)
B = torch.randn(3, 4)

C_cpu = A @ B

A_gpu = A.to("cuda")
B_gpu = B.to("cuda")

C_gpu = A_gpu @ B_gpu

print("Comparison of CPU and GPU results:")
print(torch.allclose(C_cpu, C_gpu.cpu()))

print("Details of the results:")
print("shape of A:", A.shape)
print("shape of B:", B.shape)
print("shape of C (CPU):", C_cpu.shape)
print("shape of C (GPU):", C_gpu.shape)
print("Device storage of C (CPU):", C_cpu.device)
print("Device storage of C (GPU):", C_gpu.device)
print("Data type of A:", A.dtype)
print("Data type of B:", B.dtype)
print("Data type of C (CPU):", C_cpu.dtype)
print("Data type of C (GPU):", C_gpu.dtype)
