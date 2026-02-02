# Environment Setup (Conda + pip, Slurm cluster)

This project must be installed in a way that avoids MKL / Intel OpenMP conflicts
with pip-installed CUDA PyTorch. Follow the steps below on the cluster.

## 1) Create and activate a clean Conda env

```bash
conda create -n chess-dl python=3.10 -y
conda activate chess-dl
```

## 2) Install NumPy with OpenBLAS (no MKL)

Install NumPy from conda-forge with OpenBLAS, then ensure MKL is not present.

```bash
conda install -c conda-forge numpy "libblas=*=*openblas" "libopenblas" -y

# Remove any MKL / Intel OpenMP packages if they were pulled in
conda remove -y mkl mkl-service mkl_fft mkl_random intel-openmp || true
```

Verify NumPy is using OpenBLAS:

```bash
python - <<'PY'
import numpy as np
np.show_config()
PY
```

You should see `openblas` in the BLAS/LAPACK config and no MKL / intel-openmp.

## 3) Install CUDA PyTorch via pip (required on the cluster)

Conda PyTorch does **not** correctly detect GPUs on this cluster. Use pip with
CUDA 11.8 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 \
  torchvision==0.22.1+cu118 \
  torchaudio==2.7.1+cu118
```

Verify CUDA works (run on a GPU node if required by Slurm):

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Expected: `True` for `torch.cuda.is_available()`.

## 4) Install remaining Python deps

```bash
pip install -r requirements.txt
```

## 5) Reproducibility checks (recommended)

If you see errors like:

```
undefined symbol: __kmpc_global_thread_num
```

it means MKL / Intel OpenMP is still being loaded. Remove those packages and
reinstall NumPy with OpenBLAS as in Step 2.

---

## Notes for Slurm

- Run CUDA verification and training on a GPU compute node (not the login node).
- Do **not** install PyTorch from Conda in this environment.
- Keep the environment free of MKL / Intel OpenMP packages.

---

## Submission / Evaluation entrypoint

`predict_board` lives in `scripts/07_infer_single_image.py`.

Single-image inference + visualization:

```bash
python scripts/07_infer_single_image.py --image-path path/to/image.jpg
```

Outputs are written to `results/`:
- `<image_stem>_pred.png`: rendered board with red X on OOD squares
- `<image_stem>_side_by_side.png`: input image next to rendered board
