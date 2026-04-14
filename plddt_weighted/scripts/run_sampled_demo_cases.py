#!/usr/bin/env python3
"""Run app-style sampled comparisons on selected demo PDB files.

evaluate_models.py is fast and useful for ranking, but the Streamlit app shows
sampled FASTA outputs. This script runs the same command-line tools as the app
on a small selected set and writes a CSV summary of the sampled results.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_VANILLA_RUNNER = REPO_ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
DEFAULT_PLDDT_RUNNER = REPO_ROOT / "plddt_weighted" / "protein_mpnn_run_plddt.py"
DEFAULT_VANILLA_WEIGHTS = REPO_ROOT / "ProteinMPNN" / "vanilla_model_weights"
DEFAULT_PLDDT_WEIGHTS = (
    REPO_ROOT
    / "plddt_weighted"
    / "weights"
)


def parse_header_float(header: str, key: str) -> float | None:
    """Read score=... or seq_recovery=... from a FASTA header."""

    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}=([^,\s]+)", header)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_header_int(header: str, key: str) -> int | None:
    """Read sample=... from a FASTA header."""

    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}=([^,\s]+)", header)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def read_best_design(fasta_path: Path) -> dict[str, Any] | None:
    """Return the design with the best sequence recovery from a FASTA file."""

    best: dict[str, Any] | None = None
    header: str | None = None
    seq_parts: list[str] = []

    def flush() -> None:
        nonlocal best
        if header is None or not header.startswith("T="):
            return
        sequence = "".join(seq_parts)
        row = {
            "sample": parse_header_int(header, "sample"),
            "score": parse_header_float(header, "score"),
            "seq_recovery": parse_header_float(header, "seq_recovery"),
            "sequence": sequence,
            "header": header,
        }
        if row["seq_recovery"] is None:
            return
        if best is None or row["seq_recovery"] > best["seq_recovery"]:
            best = row

    with fasta_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    flush()
    return best


def first_fasta(output_dir: Path) -> Path | None:
    """Find the first FASTA output in a ProteinMPNN output folder."""

    seq_dir = output_dir / "seqs"
    if not seq_dir.exists():
        return None
    matches = sorted(seq_dir.glob("*.fa")) + sorted(seq_dir.glob("*.fasta"))
    return matches[0] if matches else None


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run one model command."""

    print("+", shlex.join(command), flush=True)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def build_vanilla_command(args: argparse.Namespace, pdb_path: Path, out_dir: Path) -> list[str]:
    """Build the vanilla ProteinMPNN command."""

    return [
        args.python,
        str(args.vanilla_runner),
        "--pdb_path",
        str(pdb_path),
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        str(args.num_seq),
        "--batch_size",
        "1",
        "--sampling_temp",
        args.temperature,
        "--path_to_model_weights",
        str(args.vanilla_weights_dir),
        "--model_name",
        args.vanilla_model_name,
        "--seed",
        str(args.seed),
        "--suppress_print",
        "1",
    ]


def build_plddt_command(args: argparse.Namespace, pdb_path: Path, out_dir: Path) -> list[str]:
    """Build the pLDDT-weighted command."""

    return [
        args.python,
        str(args.plddt_runner),
        "--pdb_path",
        str(pdb_path),
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        str(args.num_seq),
        "--batch_size",
        "1",
        "--sampling_temp",
        args.temperature,
        "--path_to_model_weights",
        str(args.plddt_weights_dir),
        "--model_name",
        args.plddt_model_name,
        "--data_source",
        args.data_source,
        "--seed",
        str(args.seed),
        "--suppress_print",
        "1",
    ]


def write_text(path: Path, text: str) -> None:
    """Write command output to a log file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def pdb_inputs(args: argparse.Namespace) -> list[Path]:
    """Resolve PDB inputs from a directory or explicit list file."""

    paths: list[Path] = []

    if args.pdb_list:
        for line in args.pdb_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                paths.append(Path(stripped))

    if args.pdb_dir:
        paths.extend(sorted(args.pdb_dir.glob("*.pdb")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)

    if args.limit:
        unique = unique[: args.limit]

    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", type=Path, default=None)
    parser.add_argument("--pdb-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("plddt_weighted/results/demo_sampling"))
    parser.add_argument("--num-seq", type=int, default=4)
    parser.add_argument("--temperature", default="0.1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-source", choices=["auto", "alphafold", "xray"], default="alphafold")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vanilla-runner", type=Path, default=DEFAULT_VANILLA_RUNNER)
    parser.add_argument("--plddt-runner", type=Path, default=DEFAULT_PLDDT_RUNNER)
    parser.add_argument("--vanilla-weights-dir", type=Path, default=DEFAULT_VANILLA_WEIGHTS)
    parser.add_argument("--plddt-weights-dir", type=Path, default=DEFAULT_PLDDT_WEIGHTS)
    parser.add_argument("--vanilla-model-name", default="v_48_020")
    parser.add_argument("--plddt-model-name", default="epoch_last")
    args = parser.parse_args()

    pdbs = pdb_inputs(args)
    if not pdbs:
        raise SystemExit("No PDB inputs found. Use --pdb-dir or --pdb-list.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pdb_path in pdbs:
        if not pdb_path.exists():
            print(f"Skipping missing PDB: {pdb_path}")
            continue

        case_dir = args.output_dir / pdb_path.stem
        vanilla_out = case_dir / "vanilla"
        plddt_out = case_dir / "plddt_weighted"

        print(f"\n=== {pdb_path.name} ===", flush=True)
        vanilla_cmd = build_vanilla_command(args, pdb_path, vanilla_out)
        plddt_cmd = build_plddt_command(args, pdb_path, plddt_out)

        vanilla_proc = run_command(vanilla_cmd, args.timeout)
        plddt_proc = run_command(plddt_cmd, args.timeout)

        write_text(case_dir / "vanilla_stdout.txt", vanilla_proc.stdout)
        write_text(case_dir / "vanilla_stderr.txt", vanilla_proc.stderr)
        write_text(case_dir / "plddt_stdout.txt", plddt_proc.stdout)
        write_text(case_dir / "plddt_stderr.txt", plddt_proc.stderr)

        vanilla_fasta = first_fasta(vanilla_out)
        plddt_fasta = first_fasta(plddt_out)
        vanilla_best = read_best_design(vanilla_fasta) if vanilla_fasta else None
        plddt_best = read_best_design(plddt_fasta) if plddt_fasta else None

        vanilla_rec = None if vanilla_best is None else vanilla_best["seq_recovery"]
        plddt_rec = None if plddt_best is None else plddt_best["seq_recovery"]
        delta = None
        if vanilla_rec is not None and plddt_rec is not None:
            delta = plddt_rec - vanilla_rec

        rows.append(
            {
                "name": pdb_path.name,
                "pdb_path": str(pdb_path),
                "vanilla_returncode": vanilla_proc.returncode,
                "plddt_returncode": plddt_proc.returncode,
                "vanilla_best_recovery": vanilla_rec,
                "plddt_best_recovery": plddt_rec,
                "delta": delta,
                "vanilla_best_score": None if vanilla_best is None else vanilla_best["score"],
                "plddt_best_score": None if plddt_best is None else plddt_best["score"],
                "vanilla_fasta": "" if vanilla_fasta is None else str(vanilla_fasta),
                "plddt_fasta": "" if plddt_fasta is None else str(plddt_fasta),
            }
        )

    rows.sort(
        key=lambda row: (
            -999.0 if row["delta"] is None else row["delta"],
            -999.0 if row["plddt_best_recovery"] is None else row["plddt_best_recovery"],
        ),
        reverse=True,
    )

    summary_path = args.output_dir / "sampled_demo_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote sampled summary to: {summary_path}")
    for i, row in enumerate(rows[:10], start=1):
        print(
            f"{i:2d}. {row['name']:22s} "
            f"delta={row['delta']} "
            f"weighted={row['plddt_best_recovery']} "
            f"vanilla={row['vanilla_best_recovery']}"
        )


if __name__ == "__main__":
    main()
