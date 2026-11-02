# Milestone 01 — GPU Environment

## Goal

Set up Python, PyTorch, and CUDA support under Ubuntu/WSL and confirm the GPU
can execute a PyTorch matrix multiplication.

## Target hardware

- GPU: NVIDIA GeForce RTX 5090
- OS: Ubuntu through WSL
- Framework: PyTorch with a Blackwell-compatible CUDA build

## Test

From the repository root, with the virtual environment activated:

```bash
python milestone-01-environment/gpu_check.py
```

The script reports:
1. PyTorch version
2. Whether CUDA is available
3. Detected GPU name and compute capability
4. Shape and device of a 1024 x 1024 matrix multiplication result

## My results (fill in after running)

- Date:
- PyTorch version:
- CUDA available:
- GPU detected:
- Compute capability:
- Result shape:
- Result device:

## Issues and fixes

Document any installation issues or GPU errors here, along with the fix.

## Completion checklist

- [ ] Python virtual environment created
- [ ] CUDA-enabled PyTorch installed
- [ ] `gpu_check.py` run successfully
- [ ] Observed results recorded above
- [ ] Milestone committed to GitHub
