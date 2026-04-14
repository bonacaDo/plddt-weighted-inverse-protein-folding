#!/usr/bin/env python3
"""
download_alphafold_plddt.py
============================
Download AlphaFold-predicted structures with pLDDT scores for evaluation.

This script uses the current AlphaFold DB prediction API to resolve the
up-to-date PDB download URL for each UniProt accession, then downloads the
corresponding structure and extracts per-residue pLDDT values from the
B-factor column of CA atoms.

Usage:
    python -m plddt_weighted.scripts.download_alphafold_plddt \
        --uniprot_ids_file data/eval_uniprot_ids.txt \
        --output_dir data/alphafold_eval
"""

import argparse
import json
import os
import sys

import numpy as np
import requests


if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


DEFAULT_UNIPROT_IDS = [
    "P00698",  # Lysozyme C (Gallus gallus)
    "P01308",  # Insulin (Homo sapiens)
    "P68871",  # Hemoglobin subunit beta (Homo sapiens)
    "P0A9Q1",  # Aerotaxis receptor (E. coli)
    "P04637",  # Cellular tumor antigen p53 (Homo sapiens)
    "Q9Y6K9",  # NF-kappa-B essential modulator (Homo sapiens)
    "P38398",  # BRCA1 (Homo sapiens)
    "Q13148",  # TARDBP / TDP-43 (Homo sapiens)
    "P10636",  # Microtubule-associated protein tau (Homo sapiens)
    "P09651",  # Heterogeneous nuclear ribonucleoprotein A1 (Homo sapiens)
]

ALPHAFOLD_API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"


def load_uniprot_ids(uniprot_ids_file, output_dir):
    if uniprot_ids_file and os.path.exists(uniprot_ids_file):
        with open(uniprot_ids_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    ids_file = os.path.join(output_dir, 'uniprot_ids.txt')
    with open(ids_file, 'w') as f:
        f.write('\n'.join(DEFAULT_UNIPROT_IDS) + '\n')
    return DEFAULT_UNIPROT_IDS


def fetch_prediction_metadata(uniprot_id, session):
    """Return AlphaFold prediction metadata for a UniProt accession."""
    api_url = ALPHAFOLD_API_TEMPLATE.format(uniprot_id=uniprot_id)
    response = session.get(api_url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"metadata request failed with HTTP {response.status_code}")

    payload = response.json()
    if not payload:
        raise RuntimeError("no AlphaFold predictions returned")

    if isinstance(payload, list):
        records = payload
    else:
        records = [payload]

    for record in records:
        if 'pdbUrl' in record and record.get('pdbUrl'):
            return record

    raise RuntimeError("prediction metadata missing pdbUrl")


def download_alphafold_structure(uniprot_id, output_dir, session):
    """Download the current AlphaFold PDB file for a UniProt accession."""
    output_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1.pdb")
    if os.path.exists(output_path):
        return output_path, None

    record = fetch_prediction_metadata(uniprot_id, session)
    pdb_url = record['pdbUrl']

    response = session.get(pdb_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"PDB download failed with HTTP {response.status_code}")

    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path, record


def extract_plddt_from_pdb(pdb_path):
    """Extract per-residue pLDDT scores from an AlphaFold PDB file."""
    plddt_scores = []
    residue_ids = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                resseq = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                plddt_scores.append(bfactor)
                residue_ids.append(resseq)

    return np.array(residue_ids), np.array(plddt_scores)


def summarise_structure(uniprot_id, pdb_path, metadata):
    res_ids, plddt = extract_plddt_from_pdb(pdb_path)
    mean_plddt = np.mean(plddt)
    low_conf = np.sum(plddt < 50) / len(plddt) * 100
    high_conf = np.sum(plddt > 90) / len(plddt) * 100

    summary = {
        'uniprot_id': uniprot_id,
        'length': len(plddt),
        'mean_plddt': float(mean_plddt),
        'pct_low_conf': float(low_conf),
        'pct_high_conf': float(high_conf),
        'pdb_path': pdb_path,
    }
    if metadata:
        summary['alphafold_db_version'] = metadata.get('latestVersion')
        summary['source_url'] = metadata.get('pdbUrl')
    return summary, res_ids, plddt


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    uniprot_ids = load_uniprot_ids(args.uniprot_ids_file, args.output_dir)

    print(f"Downloading {len(uniprot_ids)} AlphaFold structures...")
    print(f"Output: {args.output_dir}")
    print()

    session = requests.Session()
    session.headers.update({'User-Agent': 'plddt-weighted-proteinmpnn/1.0'})

    summary = []
    failures = []

    for uid in uniprot_ids:
        print(f"  {uid}...", end=" ")
        try:
            pdb_path, metadata = download_alphafold_structure(uid, args.output_dir, session)
            item, res_ids, plddt = summarise_structure(uid, pdb_path, metadata)
            print(f"L={item['length']}, mean_pLDDT={item['mean_plddt']:.1f}, "
                  f"low(<50)={item['pct_low_conf']:.1f}%, "
                  f"high(>90)={item['pct_high_conf']:.1f}%")

            np.savez(os.path.join(args.output_dir, f"{uid}_plddt.npz"),
                     residue_ids=res_ids, plddt=plddt)
            summary.append(item)
        except Exception as e:
            print(f"FAILED ({e})")
            failures.append({'uniprot_id': uid, 'error': str(e)})

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    failures_path = os.path.join(args.output_dir, 'failures.json')
    with open(failures_path, 'w') as f:
        json.dump(failures, f, indent=2)

    print(f"\nDownloaded {len(summary)}/{len(uniprot_ids)} structures")
    print(f"Summary saved to {summary_path}")
    if failures:
        print(f"Failures saved to {failures_path}")

    if len(summary) == 0:
        raise SystemExit("No AlphaFold structures were downloaded")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniprot_ids_file', type=str, default='',
                        help='File with UniProt IDs (one per line)')
    parser.add_argument('--output_dir', type=str,
                        default='./data/alphafold_eval',
                        help='Output directory')
    args = parser.parse_args()
    main(args)
