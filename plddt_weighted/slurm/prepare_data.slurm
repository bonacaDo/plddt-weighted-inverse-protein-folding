#!/bin/bash
# ============================================================================
# prepare_data.slurm  –  Download and preprocess training data on Arnes HPC
# ============================================================================
#
# Submission:  sbatch slurm/prepare_data.slurm
# Log:         results/prepare_data_<JOBID>.txt
#
# This is a CPU-only job that prepares all datasets used by this project:
# 1) ProteinMPNN training set (pdb_2021aug02) for fair training comparison
# 2) CATH S40 non-redundant domain dataset
# 3) Merge both into a single combined dataset
# 4) AlphaFold structures for evaluation
# ============================================================================

set -euo pipefail

#SBATCH --job-name=prep_data
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/prepare_data_%j.txt

echo "======================================================="
echo "  Data Preparation: pdb_2021aug02 + CATH S40 + merge"
echo "======================================================="
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "-------------------------------------------------------"

module purge
module load Python/3.11

VENV="$HOME/plddt_mpnn_venv"
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet numpy torch python-dateutil requests
else
    source "$VENV/bin/activate"
fi

REPO_DIR="$HOME/plddt-weighted-proteinmpnn"
cd "${REPO_DIR}"

mkdir -p plddt_weighted/results plddt_weighted/data
mkdir -p ProteinMPNN/training

# ── Step 1: Download ProteinMPNN training dataset ───────────────────────────
echo ""
echo "[Step 1] Downloading ProteinMPNN dataset (pdb_2021aug02)..."
PDB2021_TAR="ProteinMPNN/training/pdb_2021aug02.tar.gz"
PDB2021_URL="https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz"
PDB2021_DIR="ProteinMPNN/training/pdb_2021aug02"

if [ ! -f "${PDB2021_TAR}" ]; then
    echo "  Downloading ${PDB2021_URL}"
    wget -O "${PDB2021_TAR}" "${PDB2021_URL}" || \
    curl -fL "${PDB2021_URL}" -o "${PDB2021_TAR}"
else
    echo "  Archive already exists: ${PDB2021_TAR}"
fi

if [ ! -f "${PDB2021_DIR}/list.csv" ]; then
    echo "  Extracting dataset to ProteinMPNN/training/..."
    tar -xzf "${PDB2021_TAR}" -C "ProteinMPNN/training"
else
    echo "  Extracted dataset already present: ${PDB2021_DIR}"
fi

# ── Step 2: Validate ProteinMPNN training dataset structure ─────────────────
echo ""
echo "[Step 2] Validating pdb_2021aug02 structure..."
python - << 'PY'
from pathlib import Path

root = Path("ProteinMPNN/training/pdb_2021aug02")
required = ["list.csv", "valid_clusters.txt", "test_clusters.txt"]
missing = [name for name in required if not (root / name).exists()]

if missing:
    raise SystemExit(f"Missing required files in {root}: {missing}")

print(f"Dataset OK: {root}")
for name in required:
    print(f"  found: {root / name}")
PY

# ── Step 3: Download CATH S40 dataset ───────────────────────────────────────
echo ""
echo "[Step 3] Downloading CATH S40 non-redundant domain dataset..."
CATH_DIR="plddt_weighted/data/cath_s40"
bash plddt_weighted/scripts/download_cath_s40.sh "${CATH_DIR}"

# ── Step 4: Preprocess CATH S40 through the training-data pipeline ──────────
echo ""
echo "[Step 4] Preprocessing CATH S40 structures..."
CATH_PDB_DIR="${CATH_DIR}/pdbs/dompdb"
CATH_PROCESSED_DIR="${CATH_DIR}/processed"
CATH_LIST="${CATH_DIR}/cath-dataset-nonredundant-S40.list"

python -m plddt_weighted.scripts.prepare_training_data \
    --pdb_dir   "${CATH_PDB_DIR}" \
    --output_dir "${CATH_PROCESSED_DIR}" \
    --cath_list  "${CATH_LIST}" \
    --val_fraction  0.1 \
    --test_fraction 0.1 \
    --seed 42

# ── Step 5: Merge ProteinMPNN + CATH S40 into a combined dataset ─────────────
echo ""
echo "[Step 5] Merging ProteinMPNN + CATH S40 datasets..."
MERGED_DIR="plddt_weighted/data/merged"

python -m plddt_weighted.scripts.merge_datasets \
    --dir1         "${PDB2021_DIR}" \
    --dir2         "${CATH_PROCESSED_DIR}" \
    --output_dir   "${MERGED_DIR}" \
    --val_fraction  0.1 \
    --test_fraction 0.1 \
    --seed 42

# ── Step 6: Validate merged dataset ─────────────────────────────────────────
echo ""
echo "[Step 6] Validating merged dataset..."
python - << 'PY'
from pathlib import Path

root = Path("plddt_weighted/data/merged")
required = ["list.csv", "valid_clusters.txt", "test_clusters.txt"]
missing = [name for name in required if not (root / name).exists()]

if missing:
    raise SystemExit(f"Missing required files in {root}: {missing}")

import csv
with open(root / "list.csv") as f:
    rows = list(csv.DictReader(f))
print(f"Merged dataset OK: {root}")
print(f"  Total entries: {len(rows):,}")
has_source_dir = 'source_dir' in rows[0] if rows else False
print(f"  Has source_dir column: {has_source_dir}")
PY

# ── Step 7: Download AlphaFold structures for evaluation ─────────────────────
echo ""
echo "[Step 7] Downloading AlphaFold structures..."
python -m plddt_weighted.scripts.download_alphafold_plddt \
    --output_dir plddt_weighted/data/alphafold_eval

echo ""
echo "Data preparation complete: $(date)"
echo "======================================================="
