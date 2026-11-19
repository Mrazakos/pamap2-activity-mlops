#!/usr/bin/env bash
# Checks that the pinned torch build sees the GPU and runs a deterministic matmul.
# Adapted from ~/cluster-onboarding/gpu_check.sh.   sbatch slurm/gpu_check.sh
#SBATCH --job-name=gpu-check
#SBATCH --account=mlops_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x-%j.out

source slurm/common.sh

python - <<'EOF'
import sys

import torch

print("PyTorch:", torch.__version__, "| built for CUDA:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
x = torch.randn(4096, 4096, device="cuda")
a = (x @ x).sum().item()
b = (x @ x).sum().item()
torch.cuda.synchronize()
print("Matrix multiply on GPU: OK | repeated result identical:", a == b)
EOF
