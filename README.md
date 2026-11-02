# GPU Deep Learning Research

A milestone-based repository for learning GPU programming and deep learning with PyTorch.

## Progress

| Milestone | Description | Status |
| --- | --- | --- |
| 01 | Set up the Python environment and verify GPU acceleration | Completed — GPU test passed; hardware/software documented |

## Project structure

```text
gpu-deep-learning-research/
├── milestone-01-environment/
│   ├── gpu_check.py
│   └── README.md
├── README.md
└── .gitignore
```

## Getting started

Use Ubuntu (WSL) and open this folder in VS Code.

```bash
cd ~/research/gpu-deep-learning-research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a compatible **CUDA-enabled PyTorch** build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/).
Make sure the selected build supports your GPU (RTX 5090 / Blackwell).

Then run:

```bash
python milestone-01-environment/gpu_check.py
```

If the test succeeds, it should report `CUDA available: True`, identify your
NVIDIA GPU, and show a result tensor on `cuda:0`.

## Documentation approach

Keep each milestone in its own folder. Add a short README describing the goal,
commands used, results you actually observed, and any problems you solved.
Do not commit virtual environments, environment secrets, or large datasets.
