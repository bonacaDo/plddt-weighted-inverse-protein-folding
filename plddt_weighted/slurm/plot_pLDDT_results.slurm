#!/bin/bash
# ============================================================================
# plot_pLDDT_results.slurm  -  Plot pLDDT model results
# ============================================================================
#
# Submission:  sbatch slurm/plot_pLDDT_results.slurm
# Log:         results/plot_plddt_results_<JOBID>.txt
# ============================================================================

#SBATCH --job-name=plot_plddt
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/plot_plddt_results_%j.txt

echo "======================================================="
echo "  Plot pLDDT Results"
echo "======================================================="
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "-------------------------------------------------------"

module purge
module load Python/3.11

VENV="$HOME/plddt_mpnn_venv"
source "$VENV/bin/activate"

REPO_DIR="$HOME/plddt-weighted-proteinmpnn"
cd "${REPO_DIR}/plddt_weighted"

python evaluation/plot_exported_results.py \
    --export_dir results/evaluation_pLDDT_results \
    --output_dir figures/generated_pLDDT_results \
    --new_only

echo ""
echo "Generated figures:"
find figures/generated_pLDDT_results -maxdepth 1 -type f | sort
echo ""
echo "Plotting complete: $(date)"
echo "======================================================="
