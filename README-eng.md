# pLDDT-Weighted ProteinMPNN

Arnes Hackathon 2026 project focused on confidence-aware protein sequence design.

This repository extends ProteinMPNN with pLDDT/B-factor-aware message passing so the model can trust high-confidence structural regions more than low-confidence or disordered regions.

## Project Goal

ProteinMPNN treats all residues in the input structure uniformly. For AlphaFold structures, this is often suboptimal because confidence varies strongly across regions.

Our approach introduces confidence-aware weighting directly into the graph message-passing pipeline:

1. Confidence Node Embedding: each residue receives a learned embedding of its confidence score.
2. Confidence Edge Gate: each edge message is softly weighted using confidence from both connected residues.

This lets the model down-weight noisy regions without hard-thresholding and keeps training differentiable and transfer-learning friendly.

## What Is In This Repository

- `plddt_weighted/`: confidence-weighted implementation, training pipeline, scripts, evaluation, and SLURM jobs.
- `ProteinMPNN/`: original upstream code and assets kept for reference and compatibility.
- `streamlit_demo/`: web app for interactive sequence design demos.
- `demo_cases/`: sample case definitions.
- `outputs/`: generated examples and run outputs.

## Core Model Changes

Main architectural updates are implemented in `plddt_weighted/training/model_utils.py`:

- Added confidence embedding MLP for per-residue confidence values.
- Added edge-gating MLP (sigmoid) for confidence-weighted message propagation.
- Added a confidence-weighted encoder layer variant for graph updates.

Data loading and preprocessing updates include:

- Extraction of B-factors from PDB files (used as pLDDT for AlphaFold structures).
- Normalization of confidence values before feeding the model.
- Passing confidence tensors through training/inference paths.

## Data Workflow

The weighted training pipeline in `plddt_weighted/` is optimized for HPC usage:

1. Download CATH S40 domains/PDBs.
2. Preprocess structures into `.pt` tensors.
3. Extract and normalize confidence (B-factor/pLDDT).
4. Train confidence-weighted model (initialized from base ProteinMPNN weights).
5. Evaluate against vanilla ProteinMPNN across confidence bins.

## Quick Start

### Option A: Run the Streamlit Demo

From repository root:

```bash
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py
```

### Option B: Run Weighted Demo Inference

From repository root:

```bash
python plddt_weighted/scripts/demo_inference.py \
	--pdb_file plddt_weighted/data/alphafold_eval/AF-P04637-F1.pdb \
	--model_weights plddt_weighted/weights/epoch_last.pt \
	--temperature 0.1 \
	--num_sequences 3
```

Note: exact data and checkpoint availability depends on whether preparation/training has been run.

## SLURM Workflow (Arnes HPC)

Inside `plddt_weighted/`, submit jobs in this order:

1. Environment setup

```bash
sbatch slurm/setup_env.slurm
```

2. Data preparation

```bash
sbatch slurm/prepare_data.slurm
```

3. Training

```bash
sbatch slurm/train_pLDDT_results.slurm
```

4. Evaluation

```bash
sbatch slurm/eval_pLDDT_results.slurm
```

5. Plotting

```bash
sbatch slurm/plot_pLDDT_results.slurm
```

## Evaluation and Figures

Evaluation scripts are in `plddt_weighted/evaluation/`.

Useful commands:

```bash
python plddt_weighted/evaluation/plot_exported_results.py \
	--export_dir plddt_weighted/results/evaluation_pLDDT_results \
	--output_dir plddt_weighted/figures/generated_pLDDT_results
```

```bash
python plddt_weighted/evaluation/plot_results.py --mock --output_dir plddt_weighted/figures/
```

## Why Soft Confidence Gating

We selected soft edge gating over hard masking and over naive feature concatenation because it:

- Preserves differentiability and stable optimization.
- Supports transfer learning from base ProteinMPNN checkpoints.
- Learns context-dependent confidence behavior per edge instead of fixed thresholds.

## Repository Notes

- Some large datasets and generated artifacts are intentionally excluded from GitHub and are produced/downloaded through scripts.
- `plddt_weighted/weights/epoch_last.pt` is used as a convenient local checkpoint for demo and fallback evaluation.

## Acknowledgements

- Original ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- This project builds on that architecture for confidence-aware sequence design.