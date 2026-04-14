#!/bin/bash
# ============================================================================
# train_pLDDT_results.slurm  –  Train pLDDT ProteinMPNN
#                               on the combined ProteinMPNN + CATH S40 dataset
# ============================================================================
#
# Submission:  sbatch slurm/train_pLDDT_results.slurm
# Log:         results/train_pLDDT_results_<JOBID>.txt
#
# Prerequisites:
#   1. Set up 2FA: ssh <user>@hpc-otp.arnes.si
#   2. Clone repo and set up environment: sbatch slurm/setup_env.slurm
#   3. Prepare both datasets:  sbatch slurm/prepare_data.slurm
#      (this downloads pdb_2021aug02, CATH S40, and produces the merged dir)
# ============================================================================

#SBATCH --job-name=train_plddt
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=results/train_pLDDT_results_%j.txt

# ── System info ──────────────────────────────────────────────────────────────
echo "======================================================="
echo "  pLDDT ProteinMPNN Training"
echo "======================================================="
echo "Date        : $(date)"
echo "Node        : $(hostname)"
echo "User        : $USER"
echo "Job ID      : $SLURM_JOB_ID"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Memory      : $SLURM_MEM_PER_NODE MB"
echo "-------------------------------------------------------"
echo "GPU info:"
nvidia-smi
echo "-------------------------------------------------------"

# ── Modules ──────────────────────────────────────────────────────────────────
module purge
module load Python/3.11
module load CUDA
echo "Python : $(python --version)"
echo "CUDA   : $(nvcc --version | grep release)"

# ── Virtual environment ──────────────────────────────────────────────────────
VENV="$HOME/plddt_mpnn_venv"
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV ..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118
    pip install --quiet numpy python-dateutil
else
    source "$VENV/bin/activate"
fi

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
echo "-------------------------------------------------------"

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/plddt-weighted-proteinmpnn"
TRAIN_DIR="${REPO_DIR}/plddt_weighted/training"
# Merged dataset produced by prepare_data.slurm (ProteinMPNN + CATH S40)
DATA_DIR="${REPO_DIR}/plddt_weighted/data/merged"

# Vanilla checkpoint for initialisation (from original ProteinMPNN)
VANILLA_CKPT="${REPO_DIR}/ProteinMPNN/training/exp_020/model_weights/epoch_last.pt"

mkdir -p "${REPO_DIR}/plddt_weighted/results"

# ── Sanity-check merged dataset ───────────────────────────────────────────────
echo ""
echo "Checking merged dataset at ${DATA_DIR} ..."
python - "${DATA_DIR}" << 'PY'
import sys, csv
from pathlib import Path

data_dir = Path(sys.argv[1])
for name in ("list.csv", "valid_clusters.txt", "test_clusters.txt"):
    if not (data_dir / name).exists():
        raise SystemExit(f"ERROR: missing {name} in {data_dir}")

with open(data_dir / "list.csv") as f:
    rows = list(csv.DictReader(f))

print(f"  Entries total   : {len(rows):,}")
has_src = 'source_dir' in rows[0] if rows else False
print(f"  Has source_dir  : {has_src}")

n_val  = sum(1 for l in open(data_dir / "valid_clusters.txt") if l.strip())
n_test = sum(1 for l in open(data_dir / "test_clusters.txt")  if l.strip())
print(f"  Valid clusters  : {n_val:,}")
print(f"  Test  clusters  : {n_test:,}")
PY
echo "-------------------------------------------------------"

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo "Starting pLDDT training on combined ProteinMPNN + CATH S40 dataset..."
echo "-------------------------------------------------------"

cd "${TRAIN_DIR}"

python training.py \
    --path_for_training_data "${DATA_DIR}" \
    --path_for_outputs "${REPO_DIR}/plddt_weighted/results/exp_combined_lowconf_focus" \
    --from_vanilla_checkpoint "${VANILLA_CKPT}" \
    --warmup_new_params 5 \
    --num_epochs 200 \
    --save_model_every_n_epochs 10 \
    --reload_data_every_n_epochs 1 \
    --num_examples_per_epoch 5000 \
    --batch_size 10000 \
    --max_protein_length 10000 \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --num_neighbors 48 \
    --dropout 0.1 \
    --backbone_noise 0.2 \
    --mixed_precision True \
    --gradient_norm 1.0 \
    --lowconf_threshold 0.7 \
    --lowconf_aux_weight 0.75 \
    --lowconf_power 1.0

echo ""
echo "Training complete: $(date)"
echo "======================================================="
