#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v python >/dev/null 2>&1 && [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi

python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Keep the CUDA-enabled torch package from the AutoDL image.
grep -viE '^(torch|torchvision|torchaudio)([<=> ]|$)' requirements.txt > /tmp/patchtst_requirements_no_torch.txt
python -m pip install -r /tmp/patchtst_requirements_no_torch.txt

python - <<'PY'
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. Choose an AutoDL PyTorch CUDA image.")
PY

echo "Environment setup complete."
