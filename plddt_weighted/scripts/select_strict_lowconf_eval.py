#!/usr/bin/env python3
"""
select_strict_lowconf_eval.py
=============================
Build a stricter low-confidence AlphaFold evaluation set from one or more
already-downloaded AlphaFold eval directories.

Selection is based on the summary metadata produced by
``download_alphafold_plddt.py``:
- ``mean_plddt`` must be <= ``max_mean_plddt``
- ``pct_low_conf`` must be >= ``min_pct_low_conf``

Matching PDB/NPZ files are copied into a fresh output directory so the
existing evaluation scripts can be pointed at the resulting folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load_candidates(input_dirs):
    seen = {}
    for raw_dir in input_dirs:
        input_dir = Path(raw_dir)
        summary_path = input_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in {input_dir}")

        with open(summary_path) as f:
            summary = json.load(f)

        for item in summary:
            uid = item["uniprot_id"]
            if uid not in seen:
                seen[uid] = (input_dir, item)
    return list(seen.values())


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more AlphaFold eval directories containing summary.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the strict low-confidence subset",
    )
    parser.add_argument(
        "--max_mean_plddt",
        type=float,
        default=75.0,
        help="Keep proteins whose mean pLDDT is at most this value",
    )
    parser.add_argument(
        "--min_pct_low_conf",
        type=float,
        default=20.0,
        help="Keep proteins whose pct_low_conf is at least this value",
    )
    parser.add_argument(
        "--min_structures",
        type=int,
        default=5,
        help="Fail if fewer than this many structures pass the filter",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.input_dirs)
    selected = []

    for input_dir, item in candidates:
        mean_plddt = float(item["mean_plddt"])
        pct_low_conf = float(item["pct_low_conf"])
        if mean_plddt <= args.max_mean_plddt and pct_low_conf >= args.min_pct_low_conf:
            selected.append((input_dir, item))

    selected.sort(key=lambda x: (x[1]["mean_plddt"], -x[1]["pct_low_conf"], x[1]["uniprot_id"]))

    if len(selected) < args.min_structures:
        raise SystemExit(
            f"Only {len(selected)} structures passed the strict filter; "
            f"need at least {args.min_structures}. "
            f"Try loosening thresholds."
        )

    selected_summary = []
    selected_ids = []

    for input_dir, item in selected:
        uid = item["uniprot_id"]
        pdb_name = f"AF-{uid}-F1.pdb"
        npz_name = f"{uid}_plddt.npz"
        copy_if_exists(input_dir / pdb_name, output_dir / pdb_name)
        copy_if_exists(input_dir / npz_name, output_dir / npz_name)
        selected_summary.append(item)
        selected_ids.append(uid)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(selected_summary, f, indent=2)
    with open(output_dir / "uniprot_ids.txt", "w") as f:
        f.write("\n".join(selected_ids) + "\n")

    print("=" * 60)
    print("  Strict low-confidence eval subset created")
    print("=" * 60)
    print(f"  Input dirs         : {', '.join(args.input_dirs)}")
    print(f"  Output dir         : {output_dir}")
    print(f"  max_mean_plddt     : {args.max_mean_plddt}")
    print(f"  min_pct_low_conf   : {args.min_pct_low_conf}")
    print(f"  Structures kept    : {len(selected_summary)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
