#!/usr/bin/env python3
"""Rank evaluated structures for a presentation demo.

This script is meant to run after plddt_weighted/evaluation/evaluate_models.py.
It reads evaluation_results.json, ranks structures where the pLDDT-weighted
model beats the vanilla model, and optionally copies the best PDB files into a
small folder that is easy to rsync back to a laptop.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


def extract_uniprot_id(name: str) -> str:
    """Extract a UniProt accession from names such as AF-P01308-F1.pdb."""

    match = re.search(r"AF-([A-Z0-9]+)-F\d+", name)
    if match:
        return match.group(1)
    return Path(name).stem


def weighted_bin_average(result: dict[str, Any], bins: list[str], metric: str) -> float | None:
    """Compute a residue-weighted average for selected pLDDT bins."""

    total = 0.0
    n_total = 0

    for bin_name in bins:
        bin_stats = result.get("bins", {}).get(bin_name)
        if not bin_stats:
            continue
        n_residues = int(bin_stats.get("n_residues", 0))
        if n_residues <= 0 or metric not in bin_stats:
            continue
        total += n_residues * float(bin_stats[metric])
        n_total += n_residues

    if n_total == 0:
        return None
    return total / n_total


def count_focus_residues(result: dict[str, Any], bins: list[str]) -> int:
    """Count residues in the selected low-confidence bins."""

    total = 0
    for bin_name in bins:
        bin_stats = result.get("bins", {}).get(bin_name)
        if bin_stats:
            total += int(bin_stats.get("n_residues", 0))
    return total


def locate_pdb(pdb_dir: Path | None, result_name: str, uniprot_id: str) -> Path | None:
    """Find the PDB file belonging to one evaluation result."""

    if pdb_dir is None:
        return None

    direct = pdb_dir / result_name
    if direct.exists():
        return direct

    matches = sorted(pdb_dir.glob(f"*{uniprot_id}*.pdb"))
    return matches[0] if matches else None


def build_row(result: dict[str, Any], focus_bins: list[str], pdb_dir: Path | None) -> dict[str, Any]:
    """Convert one evaluation result into a flat ranking row."""

    name = result["name"]
    uniprot_id = extract_uniprot_id(name)
    length = int(result["length"])
    focus_n = count_focus_residues(result, focus_bins)
    lowconf_fraction = focus_n / length if length else 0.0

    vanilla_focus = weighted_bin_average(result, focus_bins, "vanilla_recovery")
    plddt_focus = weighted_bin_average(result, focus_bins, "plddt_recovery")
    focus_delta = None
    if vanilla_focus is not None and plddt_focus is not None:
        focus_delta = plddt_focus - vanilla_focus

    overall_delta = float(result["plddt_recovery"]) - float(result["vanilla_recovery"])
    pdb_path = locate_pdb(pdb_dir, name, uniprot_id)

    # A presentation case should be a clear win, have meaningful low-confidence
    # content, and not be too huge. This score is only for sorting; the raw
    # columns are also written so we can sanity-check the choice.
    size_penalty = max(0, length - 500) / 5000
    presentation_score = (
        overall_delta
        + 0.50 * (focus_delta or 0.0)
        + 0.05 * lowconf_fraction
        - size_penalty
    )

    return {
        "name": name,
        "uniprot_id": uniprot_id,
        "length": length,
        "mean_plddt": float(result.get("mean_plddt", 0.0)),
        "focus_bins": "+".join(focus_bins),
        "focus_residues": focus_n,
        "lowconf_fraction": lowconf_fraction,
        "vanilla_recovery": float(result["vanilla_recovery"]),
        "plddt_recovery": float(result["plddt_recovery"]),
        "overall_delta": overall_delta,
        "vanilla_focus_recovery": vanilla_focus,
        "plddt_focus_recovery": plddt_focus,
        "focus_delta": focus_delta,
        "vanilla_perplexity": float(result.get("vanilla_perplexity", 0.0)),
        "plddt_perplexity": float(result.get("plddt_perplexity", 0.0)),
        "presentation_score": presentation_score,
        "pdb_path": "" if pdb_path is None else str(pdb_path),
    }


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Apply user-facing filters for demo quality."""

    if row["length"] < args.min_length or row["length"] > args.max_length:
        return False
    if row["lowconf_fraction"] < args.min_lowconf_fraction:
        return False
    if row["overall_delta"] < args.min_delta:
        return False
    if args.require_focus_win and (row["focus_delta"] is None or row["focus_delta"] < 0):
        return False
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ranking rows to a CSV file."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Write a tiny explanation next to the selected demo cases."""

    lines = [
        "Demo case selection",
        "===================",
        "",
        f"Source evaluation JSON: {args.evaluation_json}",
        f"Focus bins: {', '.join(args.focus_bins)}",
        f"Top N: {args.top_n}",
        "",
        "Columns to inspect:",
        "- overall_delta: pLDDT-weighted recovery minus vanilla recovery. Higher is better.",
        "- focus_delta: same comparison, but only in the selected low-confidence bins.",
        "- lowconf_fraction: fraction of the protein in those low-confidence bins.",
        "",
        "Selected cases:",
    ]

    for i, row in enumerate(rows, start=1):
        lines.append(
            f"{i}. {row['name']} ({row['uniprot_id']}): "
            f"delta={row['overall_delta']:+.4f}, "
            f"focus_delta={row['focus_delta'] if row['focus_delta'] is not None else 'n/a'}, "
            f"length={row['length']}, lowconf={row['lowconf_fraction']:.1%}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-json", required=True, type=Path)
    parser.add_argument("--pdb-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("plddt_weighted/results/demo_case_selection"))
    parser.add_argument("--focus-bins", nargs="+", default=["very_low", "low"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=700)
    parser.add_argument("--min-lowconf-fraction", type=float, default=0.05)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--require-focus-win", action="store_true")
    parser.add_argument("--copy-pdbs", action="store_true")
    args = parser.parse_args()

    results = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_row(result, args.focus_bins, args.pdb_dir) for result in results]
    rows.sort(
        key=lambda row: (
            row["presentation_score"],
            row["overall_delta"],
            row["lowconf_fraction"],
        ),
        reverse=True,
    )

    filtered = [row for row in rows if keep_row(row, args)]
    selected = filtered[: args.top_n]

    write_csv(args.output_dir / "all_ranked_cases.csv", rows)
    write_csv(args.output_dir / "selected_demo_cases.csv", selected)
    (args.output_dir / "selected_ids.txt").write_text(
        "\n".join(row["uniprot_id"] for row in selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )
    write_readme(args.output_dir / "README.txt", selected, args)

    if args.copy_pdbs:
        selected_pdb_dir = args.output_dir / "selected_pdbs"
        selected_pdb_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            pdb_path = Path(row["pdb_path"]) if row["pdb_path"] else None
            if pdb_path and pdb_path.exists():
                shutil.copy2(pdb_path, selected_pdb_dir / pdb_path.name)

    print(f"Wrote ranking to: {args.output_dir / 'selected_demo_cases.csv'}")
    print()
    for i, row in enumerate(selected, start=1):
        print(
            f"{i:2d}. {row['name']:22s} "
            f"delta={row['overall_delta']:+.4f} "
            f"focus={row['focus_delta'] if row['focus_delta'] is not None else 'n/a'} "
            f"lowconf={row['lowconf_fraction']:.1%} "
            f"L={row['length']}"
        )


if __name__ == "__main__":
    main()

