# Milestone 01 — GPU Environment

## Goal

Set up a Python and CUDA-enabled PyTorch environment under Ubuntu/WSL and verify
that the NVIDIA GeForce RTX 5090 can execute matrix multiplication on the GPU.

## Hardware and software

| Item | Verified value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| OS | Ubuntu on WSL |
| Python | Version not yet recorded |
| PyTorch | `2.15.0+cu134` |
| PyTorch CUDA build | `cu134` (CUDA 13.4) |
| GPU compute capability | `(12, 0)` |
| CUDA available to PyTorch | `True` |

> The `cu134` suffix identifies the CUDA build used by PyTorch. The NVIDIA
> driver version and any system-wide CUDA Toolkit installation have not yet been
> recorded.

## Environment setup

The project setup instructions are:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential

mkdir -p ~/research/gpu-dl
cd ~/research/gpu-dl

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch compatible with the RTX 5090 using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).

> The exact PyTorch installation command used has not been recorded, so this
> README does not claim a specific pip command.

## GPU test

The test script in this repository is
[`gpu_check.py`](gpu_check.py).

Run it from this repository's root:

```bash
python milestone-01-environment/gpu_check.py
```

The observed test was executed in the original setup folder:

```bash
cd ~/research/gpu-dl
python gpu_check.py
```

### Observed output

```text
PyTorch version: 2.15.0+cu134
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
Result shape: torch.Size([1024, 1024])
Result device: cuda:0
```

**Result: PASS.** PyTorch detected the RTX 5090 and completed a
1024 × 1024 matrix multiplication with the result stored on `cuda:0`.


