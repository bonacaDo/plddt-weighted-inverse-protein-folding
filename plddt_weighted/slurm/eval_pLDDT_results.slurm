#!/bin/bash
# ============================================================================
# eval_pLDDT_results.slurm  –  Evaluate pLDDT model vs Vanilla
# ============================================================================
#
# Submission:  sbatch slurm/eval_pLDDT_results.slurm
# Log:         results/eval_pLDDT_results_<JOBID>.txt
# ============================================================================

#SBATCH --job-name=eval_plddt_results
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/eval_pLDDT_results_%j.txt

echo "======================================================="
echo "  pLDDT Evaluation"
echo "======================================================="
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "-------------------------------------------------------"

module purge
module load Python/3.11
module load CUDA

VENV="$HOME/plddt_mpnn_venv"
source "$VENV/bin/activate"

REPO_DIR="$HOME/plddt-weighted-proteinmpnn"
cd "${REPO_DIR}/plddt_weighted"

mkdir -p results/evaluation_pLDDT_results

PLDDT_CKPT_TRAINED="${REPO_DIR}/plddt_weighted/results/exp_combined_lowconf_focus/model_weights/epoch_last.pt"
PLDDT_CKPT_FALLBACK="${REPO_DIR}/plddt_weighted/weights/epoch_last.pt"

if [ -f "${PLDDT_CKPT_TRAINED}" ]; then
    PLDDT_CKPT="${PLDDT_CKPT_TRAINED}"
    echo "Using latest trained checkpoint: ${PLDDT_CKPT}"
elif [ -f "${PLDDT_CKPT_FALLBACK}" ]; then
    PLDDT_CKPT="${PLDDT_CKPT_FALLBACK}"
    echo "Using fallback checkpoint: ${PLDDT_CKPT}"
else
    echo "ERROR: no pLDDT checkpoint found." >&2
    echo "Checked: ${PLDDT_CKPT_TRAINED}" >&2
    echo "Checked: ${PLDDT_CKPT_FALLBACK}" >&2
    exit 1
fi

# ── Run evaluation ───────────────────────────────────────────────────────────
python evaluation/evaluate_models.py \
    --vanilla_weights "${REPO_DIR}/ProteinMPNN/vanilla_model_weights/v_48_020.pt" \
    --plddt_weights "${PLDDT_CKPT}" \
    --test_pdbs "${REPO_DIR}/plddt_weighted/data/alphafold_eval" \
    --output_dir "${REPO_DIR}/plddt_weighted/results/evaluation_pLDDT_results"

echo ""
echo "Evaluation complete: $(date)"
echo "======================================================="
