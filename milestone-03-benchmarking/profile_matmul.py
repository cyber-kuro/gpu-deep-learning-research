import torch
from torch.profiler import profile, ProfilerActivity

N = 1024

A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")

# Warmup
for _ in range(10):
    C = A @ B

torch.cuda.synchronize()

# Profile 10 matrix multiplications
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]
) as prof:

    for _ in range(10):
        C = A @ B

    torch.cuda.synchronize()

print("\n=== GPU OPERATIONS ===")
print(
    prof.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=15
    )
)

print("\n=== CPU OPERATIONS ===")
print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15
    )
)