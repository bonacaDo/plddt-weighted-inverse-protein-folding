#!/bin/bash
# ============================================================================
# download_cath_s40.sh  –  Download CATH S40 non-redundant domain dataset
# ============================================================================
# This script downloads the official CATH S40 representative domain list and
# the corresponding bulk PDB archive from the current CATH release.
#
# Usage:
#   bash scripts/download_cath_s40.sh [output_dir]
#
# Default output_dir: ./data/cath_s40
# ============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-./data/cath_s40}"
PDB_DIR="${OUTPUT_DIR}/pdbs"
DOMAIN_LIST="${OUTPUT_DIR}/cath-dataset-nonredundant-S40.list"
PDB_ARCHIVE="${OUTPUT_DIR}/cath-dataset-nonredundant-S40.pdb.tgz"

CATH_BASE_URL="https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets"
DOMAIN_LIST_URL="${CATH_BASE_URL}/cath-dataset-nonredundant-S40.list"
PDB_ARCHIVE_URL="${CATH_BASE_URL}/cath-dataset-nonredundant-S40.pdb.tgz"

mkdir -p "${PDB_DIR}"

echo "============================================="
echo "  CATH S40 Dataset Download"
echo "============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# --- Step 1: Download CATH S40 domain list ---
echo "[1/3] Downloading CATH S40 domain list..."
if [ ! -f "${DOMAIN_LIST}" ]; then
    wget -q -O "${DOMAIN_LIST}" "${DOMAIN_LIST_URL}" || \
    curl -fsSL "${DOMAIN_LIST_URL}" -o "${DOMAIN_LIST}"
    echo "  Downloaded $(wc -l < "${DOMAIN_LIST}") domains"
else
    echo "  Domain list already exists, skipping download"
fi

# --- Step 2: Download bulk PDB archive ---
echo "[2/3] Downloading bulk CATH S40 PDB archive..."
if [ ! -f "${PDB_ARCHIVE}" ]; then
    wget -q -O "${PDB_ARCHIVE}" "${PDB_ARCHIVE_URL}" || \
    curl -fsSL "${PDB_ARCHIVE_URL}" -o "${PDB_ARCHIVE}"
    echo "  Downloaded archive: $(du -h "${PDB_ARCHIVE}" | awk '{print $1}')"
else
    echo "  PDB archive already exists, skipping download"
fi

# --- Step 3: Extract archive ---
echo "[3/3] Extracting PDB structures..."
if [ -d "${PDB_DIR}/dompdb" ] && [ "$(find "${PDB_DIR}/dompdb" -type f | wc -l)" -gt 0 ]; then
    echo "  Extracted structure files already present, skipping extraction"
else
    tar -xzf "${PDB_ARCHIVE}" -C "${PDB_DIR}"
fi

echo ""
echo "============================================="
echo "  Download complete!"
echo "  Domains in list: $(wc -l < "${DOMAIN_LIST}")"
echo "  Extracted files: $(find "${PDB_DIR}" -type f | wc -l)"
echo "============================================="
echo ""
echo "Next step: Run the preprocessing script:"
echo "  python scripts/prepare_training_data.py --pdb_dir ${PDB_DIR} --output_dir ${OUTPUT_DIR}/processed"
