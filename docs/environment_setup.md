# Environment Setup

## Preflight

- Operating system: Windows 11 Home, 64-bit.
- Python command used: `py -3.11`.
- Python version: 3.11.9, 64-bit.
- Compute target: CUDA.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
- NVIDIA driver: 596.08.
- CUDA reported by `nvidia-smi`: 13.2.
- Virtual environment: `.venv`.

## Install Commands Run

```powershell
py -3.11 -m venv --clear .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements\runtime.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## Verification

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --output docs\environment_report.json --compute-target cuda
.\.venv\Scripts\python.exe scripts\smoke_test_pyg_xai.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Notes

- PyTorch was installed before PyTorch Geometric.
- No GraphXAI, Attack-XGNN, Captum, dataset, or checkpoint was installed.
- No system CUDA toolkit or global Python configuration was modified.
- Pytest currently reports a PyTorch deprecation warning for `torch.jit.script`; tests still pass.
