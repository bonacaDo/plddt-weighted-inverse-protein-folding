#!/usr/bin/env bash
set -euo pipefail

PDB_FILE="./ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb"
MODEL_DIR="plddt_weighted/weights"
MODEL_NAME="epoch_last"
OUT_FOLDER=""
NUM_SEQ=8
TEMP="0.1"
DATA_SOURCE="auto"
CHAINS=""

usage() {
    cat <<'EOF'
Usage:
  ./run_design.sh [pdb_file] [options]

Options:
  --pdb FILE          Input PDB backbone file.
  --out DIR           Output directory. Default: outputs/local_design_<pdb_name>
  --num N             Number of sequences to generate. Default: 8
  --temp "T"          Sampling temperature(s). Default: "0.1"
  --chains "A B"      Chains to design. Default: all chains
  --data-source SRC   auto, xray, or alphafold. Default: auto
  --model-name NAME   Checkpoint name without .pt. Default: epoch_last
  --help              Show this help.

Examples:
  ./run_design.sh
  ./run_design.sh ./ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb
  ./run_design.sh --pdb inputs/AF-P04637-F1.pdb --data-source alphafold --num 3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pdb)
            PDB_FILE="$2"
            shift 2
            ;;
        --out)
            OUT_FOLDER="$2"
            shift 2
            ;;
        --num)
            NUM_SEQ="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --chains)
            CHAINS="$2"
            shift 2
            ;;
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            PDB_FILE="$1"
            shift
            ;;
    esac
done

if [[ ! -f "$PDB_FILE" ]]; then
    echo "Input PDB not found: $PDB_FILE" >&2
    exit 1
fi

if [[ ! -f "$MODEL_DIR/$MODEL_NAME.pt" ]]; then
    echo "Model checkpoint not found: $MODEL_DIR/$MODEL_NAME.pt" >&2
    exit 1
fi

if [[ -z "$OUT_FOLDER" ]]; then
    pdb_name="$(basename "$PDB_FILE" .pdb)"
    OUT_FOLDER="outputs/local_design_${pdb_name}"
fi

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

cmd=(
    python plddt_weighted/protein_mpnn_run_plddt.py
    --pdb_path "$PDB_FILE"
    --out_folder "$OUT_FOLDER"
    --num_seq_per_target "$NUM_SEQ"
    --sampling_temp "$TEMP"
    --path_to_model_weights "$MODEL_DIR"
    --model_name "$MODEL_NAME"
    --data_source "$DATA_SOURCE"
)

if [[ -n "$CHAINS" ]]; then
    cmd+=(--pdb_path_chains "$CHAINS")
fi

echo "Input PDB : $PDB_FILE"
echo "Model     : $MODEL_DIR/$MODEL_NAME.pt"
echo "Output    : $OUT_FOLDER"
echo

"${cmd[@]}"

echo
echo "Done. Designed sequences are in: $OUT_FOLDER/seqs/"
