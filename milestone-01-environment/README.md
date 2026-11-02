# Milestone 01 — GPU Environment

## Goal

Configure Ubuntu/WSL with Python and a CUDA-enabled PyTorch build, then verify
that the NVIDIA GeForce RTX 5090 can perform GPU matrix multiplication.

## Hardware and software (observed November 1, 2026)

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| OS | Ubuntu 24.04.5 LTS (Noble Numbat), under WSL |
| Python | 3.12.3 (reported by `python --version` in the repository shell) |
| PyTorch | `2.15.0+cu134` (from successful GPU test) |
| PyTorch CUDA build/runtime | CUDA 13.4 (`torch.version.cuda`) |
| CUDA available | `True` |
| GPU compute capability | `(12, 0)` — Blackwell / SM 12.0 |
| NVIDIA-SMI version | `620.67.01` |
| Kernel-mode driver (KMD) version | `621.01` |
| Driver-reported CUDA UMD version | `13.5` |
| Reported GPU memory | 32,607 MiB (from `nvidia-smi`) |

**CUDA version clarification:** The `13.5` reported by `nvidia-smi` is
a driver-side CUDA version indicator, whereas `torch.version.cuda` reports
`13.4` for the installed PyTorch build. These can legitimately differ.
Neither command establishes that a separate CUDA Toolkit was installed.

**Python environment clarification:** The Python 3.12.3 version was observed
in the GitHub repository shell. The Python version and interpreter path of the
original `~/research/gpu-dl/.venv` environment have not been independently
recorded.

## Installation and setup steps

The setup procedure used for this project was:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential

mkdir -p ~/research/gpu-dl
cd ~/research/gpu-dl

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch wheel compatible with the RTX 5090 using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The exact `pip install torch ...` command used during setup was not recorded;
it is not reconstructed here.

## GPU verification test

The repository copy of the script is [`gpu_check.py`](gpu_check.py).
After activating a PyTorch-equipped Python environment, run it from the
repository root:

```bash
python milestone-01-environment/gpu_check.py
```

The successful test was run from the original working directory:

```bash
cd ~/research/gpu-dl
python gpu_check.py
```

### Actual test output

```text
PyTorch version: 2.15.0+cu134
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
Result shape: torch.Size([1024, 1024])
Result device: cuda:0
```

**Result: PASS.** PyTorch detected the RTX 5090 and completed a
1,024 × 1,024 matrix multiplication, keeping the result on the GPU
(`cuda:0`).

### Additional system checks

On November 1, 2026, these commands were run from
`~/research/gpu-deep-learning-research`:

```bash
python --version
cat /etc/os-release
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

They reported Python 3.12.3, Ubuntu 24.04.5 LTS, the NVIDIA driver versions
listed above, and PyTorch CUDA 13.4.

## Troubleshooting / lessons learned

- Git initially rejected a commit because the author identity was missing.
  Set `git config --global user.name` and `git config --global user.email`;
  the private email address is intentionally not recorded here.
- The original `~/research/gpu-dl` workspace and cloned
  `~/research/gpu-deep-learning-research` repository are separate directories.
  The successful test ran from the former; the documented script is stored in
  the latter.

## Checklist

- [x] Ubuntu and Python version recorded
- [x] RTX 5090, compute capability, and CUDA availability verified
- [x] PyTorch version and CUDA build recorded
- [x] NVIDIA driver information recorded
- [x] GPU matrix multiplication completed successfully
- [x] Results documented in GitHub
- [ ] Optional: capture the exact PyTorch installation command and the
      interpreter path from the original `.venv` for full reproducibility

> Do not commit virtual environments, secrets, access tokens, or credentials.
