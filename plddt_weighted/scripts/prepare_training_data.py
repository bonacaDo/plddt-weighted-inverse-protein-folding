#!/usr/bin/env python3
"""
prepare_training_data.py
========================
Preprocess PDB files into the .pt format expected by the ProteinMPNN
training pipeline, **including per-residue B-factors** for confidence
weighting.

This script produces:
- Per-chain .pt files with keys: seq, xyz, mask, bfac, occ
- Per-structure metadata .pt files
- list.csv, valid_clusters.txt, test_clusters.txt

Optional CATH S40 support (``--cath_list``)
-------------------------------------------
When ``--cath_list`` is supplied the script reads the CATH S40 domain-list
file (``cath-dataset-nonredundant-S40.list``) and groups domains into
clusters by their CATH superfamily (H-level, the fourth component of the
CATH code, e.g. ``1.10.8.10``).  Domains with the same superfamily code
share a cluster ID.  This ensures that structurally related domains are
always in the same train/val/test split.

AlphaFold detection
-------------------
A subset of CATH releases include AlphaFold structures whose B-factor
column carries pLDDT scores (high = confident) rather than crystallographic
B-factors (low = confident).  Including them without correction would invert
the confidence signal.  By default such entries are **excluded** from the
output.  Pass ``--include_alphafold`` to retain them; they will be stored
with an ``is_alphafold: True`` flag in their ``.pt`` file so that downstream
code can handle the inversion correctly.

Usage:
    python prepare_training_data.py \\
        --pdb_dir ./data/cath_s40/pdbs/dompdb \\
        --output_dir ./data/cath_s40/processed \\
        --cath_list ./data/cath_s40/cath-dataset-nonredundant-S40.list \\
        --val_fraction 0.1 \\
        --test_fraction 0.1
"""

import argparse
import os
import csv
import random
import sys
import numpy as np
import torch
from collections import defaultdict

if __package__ in (None, ""):
    # Allow direct execution via
    #   python plddt_weighted/scripts/prepare_training_data.py
    # in addition to module execution via
    #   python -m plddt_weighted.scripts.prepare_training_data
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from plddt_weighted.training.pdb_utils import (
    AA_3TO1, parse_resseq_sort_key, infer_confidence_source,
)

BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
# Standard atom names for the 14-atom representation
ATOM14_NAMES = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2',
                'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2']


# ---------------------------------------------------------------------------
#  CATH S40 domain-list helpers
# ---------------------------------------------------------------------------

def parse_cath_list(cath_list_path):
    """Parse a CATH S40 domain list file.

    Each non-comment line starts with a domain ID followed by a CATH code::

        1abcA00 1.10.8.10 ...

    Returns
    -------
    domain_to_superfamily : dict[str, str]
        Maps domain ID (e.g. ``"1abcA00"``) to its 4-level CATH code
        (e.g. ``"1.10.8.10"``).
    superfamily_to_cluster : dict[str, int]
        Maps each unique superfamily code to a sequential integer cluster ID.
    """
    domain_to_superfamily = {}
    superfamilies = {}
    cluster_counter = 0

    with open(cath_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            domain_id = parts[0]
            cath_code = parts[1]
            domain_to_superfamily[domain_id] = cath_code
            if cath_code not in superfamilies:
                superfamilies[cath_code] = cluster_counter
                cluster_counter += 1

    superfamily_to_cluster = superfamilies
    print(f"  CATH list: {len(domain_to_superfamily):,} domains, "
          f"{len(superfamily_to_cluster):,} superfamilies")
    return domain_to_superfamily, superfamily_to_cluster


def is_alphafold_pdb(pdb_path):
    """Return True if *pdb_path* looks like an AlphaFold model.

    Heuristics (applied in order):
    1. Filename contains the ``AF-`` prefix used by the AlphaFold DB.
    2. No ``REMARK   3`` record is present (absent in AF2 mmCIF-derived PDB).
    3. All B-factors are in [0, 100] with a median ≥ 60 (typical pLDDT).

    Using the file-system hint first keeps this fast for the common case.
    """
    basename = os.path.basename(pdb_path).upper()
    if basename.startswith('AF-') and '-F' in basename:
        return True

    has_remark3 = False
    bfactors = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('REMARK   3'):
                    has_remark3 = True
                    break
                if line[:6] in ('ATOM  ', 'HETATM'):
                    try:
                        bfactors.append(float(line[60:66]))
                    except ValueError:
                        pass
    except OSError:
        return False

    if has_remark3:
        return False  # Crystallographic refinement record -> X-ray

    if not bfactors:
        return False

    # infer_confidence_source returns 'alphafold' or 'xray' based on the
    # B-factor value profile (pLDDT values cluster near 60-100; X-ray
    # B-factors are lower and more spread).
    source = infer_confidence_source(bfactors, pdb_path=pdb_path)
    return source == 'alphafold'


def parse_pdb_with_bfactors(pdb_path):
    """Parse a PDB file and extract coordinates, sequence, and B-factors.

    Returns a dict keyed by chain ID, each containing:
    - seq: str
    - xyz: np.array [L, 14, 3]
    - mask: np.array [L, 14]
    - bfac: np.array [L, 14]
    """
    chains = defaultdict(lambda: defaultdict(dict))

    with open(pdb_path, 'r') as f:
        for line in f:
            if line[:6] in ('ATOM  ', 'HETATM'):
                resname = line[17:20].strip()
                if line[:6] == 'HETATM' and resname == 'MSE':
                    pass  # Accept selenomethionine
                elif line[:6] == 'HETATM':
                    continue

                if resname not in AA_3TO1:
                    continue

                chain_id = line[21]
                resseq = line[22:27].strip()
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                try:
                    bfactor = float(line[60:66])
                except ValueError:
                    bfactor = 0.0

                key = (chain_id, resseq, resname)
                if atom_name in ATOM14_NAMES:
                    idx = ATOM14_NAMES.index(atom_name)
                    chains[chain_id][(resseq, resname)][atom_name] = {
                        'xyz': np.array([x, y, z]),
                        'bfac': bfactor,
                        'idx': idx
                    }

    result = {}
    for chain_id, residues in chains.items():
        sorted_residues = sorted(residues.items(),
                                 key=lambda x: parse_resseq_sort_key(x[0][0]))
        seq = ''
        xyz = np.zeros((len(sorted_residues), 14, 3), dtype=np.float32)
        mask = np.zeros((len(sorted_residues), 14), dtype=np.float32)
        bfac = np.zeros((len(sorted_residues), 14), dtype=np.float32)

        for i, ((resseq, resname), atoms) in enumerate(sorted_residues):
            seq += AA_3TO1.get(resname, 'X')
            for atom_name, atom_data in atoms.items():
                j = atom_data['idx']
                xyz[i, j, :] = atom_data['xyz']
                mask[i, j] = 1.0
                bfac[i, j] = atom_data['bfac']

        # Only keep chains with at least backbone atoms for most residues
        backbone_coverage = mask[:, :4].sum() / (len(seq) * 4)
        if len(seq) >= 10 and backbone_coverage > 0.5:
            result[chain_id] = {
                'seq': seq,
                'xyz': torch.tensor(xyz),
                'mask': torch.tensor(mask),
                'bfac': torch.tensor(bfac),
            }

    return result


def process_pdb_files(pdb_dir, output_dir, val_fraction=0.1,
                      test_fraction=0.1, resolution_cutoff=3.5,
                      cath_list_path=None, include_alphafold=False):
    """Process all PDB files and create training data.

    Parameters
    ----------
    pdb_dir : str
        Directory tree containing PDB-like files.
    output_dir : str
        Destination for processed ``.pt`` files and split files.
    val_fraction, test_fraction : float
        Fractions of clusters reserved for validation / test.
    resolution_cutoff : float
        Ignored for CATH / AF2 files (kept for API compatibility).
    cath_list_path : str or None
        Path to a CATH S40 domain-list file.  When provided, domains are
        grouped into clusters by their CATH superfamily code (H-level).
    include_alphafold : bool
        When *False* (default) AlphaFold-derived entries detected via
        :func:`is_alphafold_pdb` are skipped.  Set to *True* to retain them;
        they will carry ``is_alphafold=True`` in their ``.pt`` metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_subdir = os.path.join(output_dir, 'pdb')
    os.makedirs(pdb_subdir, exist_ok=True)

    # ── Load optional CATH superfamily cluster assignments ──────────────────
    domain_to_superfamily = {}
    superfamily_to_cluster = {}
    if cath_list_path:
        print(f"Loading CATH S40 domain list: {cath_list_path}")
        domain_to_superfamily, superfamily_to_cluster = \
            parse_cath_list(cath_list_path)
        # Start cluster counter after the last CATH-assigned ID so that any
        # PDB files not present in the list still get a unique cluster.
        next_fallback_cluster = max(superfamily_to_cluster.values()) + 1 \
            if superfamily_to_cluster else 0
    else:
        next_fallback_cluster = 0

    pdb_files = []
    for root, _, files in os.walk(pdb_dir):
        for name in files:
            if name.startswith('.'):
                continue
            full_path = os.path.join(root, name)
            lower_name = name.lower()
            if lower_name.endswith('.pdb') or lower_name.endswith('.ent') or '.' not in name:
                pdb_files.append(full_path)
    pdb_files = sorted(pdb_files)
    print(f"Found {len(pdb_files)} PDB files")

    if not pdb_files:
        raise RuntimeError(
            f"No PDB-like files found under {pdb_dir}. "
            "Expected .pdb/.ent files or extensionless CATH domain files."
        )

    list_rows = []
    cluster_id = next_fallback_cluster
    processed = 0
    failed = 0
    skipped_af2 = 0

    for pdb_path in pdb_files:
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

        # ── AlphaFold detection ──────────────────────────────────────────────
        af2 = is_alphafold_pdb(pdb_path)
        if af2 and not include_alphafold:
            skipped_af2 += 1
            continue

        try:
            chains = parse_pdb_with_bfactors(pdb_path)
        except Exception as exc:
            print(f"  WARNING: failed to parse {pdb_path}: {exc}")
            failed += 1
            continue

        if not chains:
            failed += 1
            continue

        # Create subdirectory structure: pdb/XX/XXXX
        mid = pdb_id[1:3]
        chain_dir = os.path.join(pdb_subdir, mid)
        os.makedirs(chain_dir, exist_ok=True)

        chain_ids = list(chains.keys())

        # Save per-chain .pt files (add is_alphafold flag when relevant)
        for ch_id in chain_ids:
            chain_data = chains[ch_id]
            if af2:
                chain_data['is_alphafold'] = True
            chain_pt_path = os.path.join(chain_dir, f"{pdb_id}_{ch_id}.pt")
            torch.save(chain_data, chain_pt_path)

        # Save metadata .pt file
        meta = {
            'chains': chain_ids,
            'asmb_ids': ['1'] * len(chain_ids),
            'asmb_chains': [','.join(chain_ids)] * len(chain_ids),
            'asmb_xform0': torch.eye(4).unsqueeze(0),
            'tm': torch.ones(len(chain_ids), len(chain_ids), 2),
        }
        meta_path = os.path.join(chain_dir, f"{pdb_id}.pt")
        torch.save(meta, meta_path)

        # Add to list.csv
        # Determine cluster ID: prefer CATH superfamily assignment, fall back
        # to a sequential per-structure ID for files not in the list.
        if cath_list_path and pdb_id in domain_to_superfamily:
            entry_cluster = superfamily_to_cluster[domain_to_superfamily[pdb_id]]
        else:
            entry_cluster = cluster_id
            cluster_id += 1

        for ch_id in chain_ids:
            entry_id = f"{pdb_id}_{ch_id}"
            date = "2020-Jan-01"  # Placeholder
            resolution = 2.0  # Placeholder
            list_rows.append([entry_id, date, resolution, entry_id, entry_cluster])

        processed += 1

        if processed % 100 == 0:
            print(f"  Processed {processed} structures...")

    # Write list.csv
    list_csv_path = os.path.join(output_dir, 'list.csv')
    with open(list_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entry', 'date', 'resolution', 'chain', 'cluster'])
        writer.writerows(list_rows)

    # Create train/valid/test splits from the actual cluster IDs written
    all_clusters = sorted(set(row[4] for row in list_rows))
    random.shuffle(all_clusters)

    n_val = max(1, int(len(all_clusters) * val_fraction))
    n_test = max(1, int(len(all_clusters) * test_fraction))

    val_clusters = all_clusters[:n_val]
    test_clusters = all_clusters[n_val:n_val + n_test]

    val_path = os.path.join(output_dir, 'valid_clusters.txt')
    test_path = os.path.join(output_dir, 'test_clusters.txt')

    with open(val_path, 'w') as f:
        f.write('\n'.join(str(c) for c in val_clusters) + '\n')
    with open(test_path, 'w') as f:
        f.write('\n'.join(str(c) for c in test_clusters) + '\n')

    n_total_clusters = len(all_clusters)
    print(f"\n{'='*50}")
    print(f"  Processing complete!")
    print(f"  Processed:       {processed} structures")
    print(f"  Failed:          {failed}")
    if skipped_af2:
        print(f"  Skipped (AF2):   {skipped_af2}")
    print(f"  Clusters:        {n_total_clusters}")
    print(f"  Train:           {n_total_clusters - n_val - n_test}")
    print(f"  Validation:      {n_val}")
    print(f"  Test:            {n_test}")
    print(f"  Output:          {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_arg.add_argument('--pdb_dir', type=str, required=True,
                            help='Directory containing PDB files')
    parser_arg.add_argument('--output_dir', type=str, required=True,
                            help='Output directory for processed data')
    parser_arg.add_argument('--val_fraction', type=float, default=0.1)
    parser_arg.add_argument('--test_fraction', type=float, default=0.1)
    parser_arg.add_argument('--seed', type=int, default=42)
    parser_arg.add_argument(
        '--cath_list', type=str, default=None,
        help='Path to CATH S40 domain-list file (cath-dataset-nonredundant-S40.list). '
             'When provided, CATH superfamily codes are used as cluster IDs '
             'for train/val/test splitting.')
    parser_arg.add_argument(
        '--include_alphafold', action='store_true',
        help='Retain AlphaFold-derived entries (excluded by default). '
             'They will be stored with is_alphafold=True in their .pt files.')
    args = parser_arg.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    process_pdb_files(args.pdb_dir, args.output_dir,
                      args.val_fraction, args.test_fraction,
                      cath_list_path=args.cath_list,
                      include_alphafold=args.include_alphafold)
