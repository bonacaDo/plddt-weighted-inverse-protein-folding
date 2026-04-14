#!/bin/bash
# ============================================================================
# setup_env.slurm  –  One-time environment setup on Arnes HPC (SLING)
# ============================================================================
#
# Run this FIRST before any other SLURM jobs.
# Submission:  sbatch slurm/setup_env.slurm
# ============================================================================

#SBATCH --job-name=setup_env
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/setup_env_%j.txt

echo "======================================================="
echo "  Environment Setup for pLDDT-Weighted ProteinMPNN"
echo "======================================================="
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "-------------------------------------------------------"

module purge
module load Python/3.11
module load CUDA
echo "Python : $(python --version)"
echo "CUDA   : $(nvcc --version | grep release)"

# ── Create virtual environment ───────────────────────────────────────────────
VENV="$HOME/plddt_mpnn_venv"
echo "Creating virtual environment at $VENV ..."
python -m venv "$VENV"
source "$VENV/bin/activate"

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install --quiet numpy python-dateutil requests matplotlib seaborn

# Verify installation
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
import numpy; print(f'NumPy {numpy.__version__}')
import matplotlib; print(f'Matplotlib {matplotlib.__version__}')
print('All dependencies OK!')
"

# ── Create directory structure ───────────────────────────────────────────────
REPO_DIR="$HOME/plddt-weighted-proteinmpnn"
mkdir -p "${REPO_DIR}/plddt_weighted/results"
mkdir -p "${REPO_DIR}/plddt_weighted/data"

echo ""
echo "Setup complete: $(date)"
echo "======================================================="
echo ""
echo "Next steps:"
echo "  1. sbatch slurm/prepare_data.slurm    # Download training/eval datasets"
echo "  2. sbatch slurm/train_pLDDT_results.slurm  # Train the model"
echo "  3. sbatch slurm/eval_pLDDT_results.slurm   # Evaluate results"
