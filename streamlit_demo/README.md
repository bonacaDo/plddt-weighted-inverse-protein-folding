# Streamlit Demo for pLDDT-Weighted ProteinMPNN

This folder contains a small Streamlit app for presenting a comparison between the standard ProteinMPNN model and our confidence-weighted model.

The app does not reimplement inference. It calls the existing command-line runners:

- `ProteinMPNN/protein_mpnn_run.py` for the baseline ProteinMPNN model.
- `plddt_weighted/protein_mpnn_run_plddt.py` for the confidence-weighted model.

## Input

The input is a `.pdb` file with a 3D protein structure. The model reads the protein backbone atoms from the `ATOM` records, especially `N`, `CA`, `C`, and `O`.

For the confidence-weighted model, the confidence signal comes from the PDB B-factor column:

- AlphaFold PDB files usually store pLDDT in the B-factor column on a 0-100 scale.
- Experimental structures use real B-factors. In that case, choose `xray`, where lower B-factor is treated as a more reliable part of the structure.
- `auto` tries to infer the meaning of the column automatically.

## 3D View

The app includes a simplified 3D structure view. It is not a full molecular cartoon renderer. It shows the CA backbone: each point is one CA atom from the PDB file, and the line connects consecutive residues in the same chain.

For AlphaFold structures, the points are colored by pLDDT:

- orange: very low confidence, below 50
- yellow: low confidence, 50-70
- light blue: good confidence, 70-90
- dark blue: very high confidence, above 90

For experimental PDB structures, the same colors show a relative confidence signal derived from the B-factor column, where lower B-factor is treated as more reliable. This is useful for presentation, but it is not an official pLDDT score.

## Running the App

From the repository root:

```bash
cd ~/plddt-weighted-proteinmpnn
source .venv/bin/activate
```

If Streamlit is not installed yet, run this once:

```bash
pip install -r streamlit_demo/requirements.txt
```

Then start the app:

```bash
bash streamlit_demo/run_app.sh
```

Streamlit will print a local browser URL, usually something like:

```text
Local URL: http://localhost:8501
```

Open that URL in a browser. If the port is already taken, Streamlit will print a different port.

## Demo Cases

Presentation-ready PDB files are stored in the repository root:

```text
demo_cases/
```

In the app, select them from the sidebar under:

```text
Primeri za predstavitev
```

The main default case is:

```text
demo_cases/AF-P15502-F1.pdb
```

A faster live-demo case is:

```text
demo_cases/AF-P05114-F1.pdb
```

A backup experimental PDB case is:

```text
demo_cases/4GYT.pdb
```

For `AF-*` cases, use:

```text
AlphaFold: pLDDT v B-factor stolpcu
```

For `4GYT`, use:

```text
Eksperimentalni PDB: B-factor
```

## Model Locations

Baseline ProteinMPNN model:

```text
ProteinMPNN/vanilla_model_weights/v_48_020.pt
```

Confidence-weighted model:

```text
plddt_weighted/weights/epoch_last.pt
```

The app passes checkpoint names without the `.pt` extension:

```text
v_48_020
epoch_last
```

## Output Locations

Each app run is saved under:

```text
outputs/streamlit_demo/runs/
```

Uploaded PDB files are saved under:

```text
outputs/streamlit_demo/uploads/
```
