#!/usr/bin/env python3
"""
merge_datasets.py
=================
Merge two independently processed ProteinMPNN-style datasets (e.g. the
ProteinMPNN pdb_2021aug02 set and a CATH S40 set) into a single
training-ready directory.

No files are copied.  The merged ``list.csv`` gains a ``source_dir`` column
that tells ``loader_pdb()`` which tree to look in for each entry's ``.pt``
files.  The ``valid_clusters.txt`` / ``test_clusters.txt`` files are
regenerated from the merged cluster pool.

Usage
-----
    python merge_datasets.py \\
        --dir1 ./ProteinMPNN/training/pdb_2021aug02 \\
        --dir2 ./data/cath_s40/processed \\
        --output_dir ./data/merged \\
        --val_fraction 0.1 \\
        --test_fraction 0.1 \\
        --seed 42

The resulting ``output_dir`` can then be passed directly to training::

    python training.py --path_for_training_data ./data/merged ...
"""

import argparse
import csv
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _normalise_row(row, source_dir):
    """Map supported list.csv schemas to the canonical merged schema.

    Supported input formats:
    - pLDDT/CATH style: entry,date,resolution,chain,cluster[,source_dir]
    - Original ProteinMPNN style:
      CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE
    """
    if 'entry' in row and 'cluster' in row:
        normalised = {
            'entry': row['entry'],
            'date': row.get('date', ''),
            'resolution': row.get('resolution', ''),
            'chain': row.get('chain', row['entry']),
            'cluster': row['cluster'],
            'source_dir': row.get('source_dir', '') or source_dir,
        }
        return normalised

    if 'CHAINID' in row and 'CLUSTER' in row:
        chainid = row['CHAINID']
        normalised = {
            'entry': chainid,
            'date': row.get('DEPOSITION', ''),
            'resolution': row.get('RESOLUTION', ''),
            'chain': chainid,
            'cluster': row['CLUSTER'],
            'source_dir': source_dir,
        }
        return normalised

    raise KeyError(
        f"Unsupported list.csv schema in row from {source_dir}: "
        f"{sorted(row.keys())}"
    )


def read_list_csv(path, source_dir):
    """Read list.csv and return rows using the canonical merged schema."""
    source_dir = os.path.abspath(source_dir)
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_normalise_row(row, source_dir))
    return rows


def _cluster_key(row):
    """Return the raw cluster string from a row dict."""
    return row['cluster']


def renumber_clusters(rows, start_id):
    """Remap cluster IDs in *rows* so they begin at *start_id*.

    The relative grouping is preserved: entries that previously shared a
    cluster ID will still share one after renumbering.

    Returns (updated_rows, next_available_id).
    """
    old_to_new = {}
    next_id = start_id
    for row in rows:
        old = row['cluster']
        if old not in old_to_new:
            old_to_new[old] = next_id
            next_id += 1
        row['cluster'] = str(old_to_new[old])
    return rows, next_id


def deduplicate(rows1, rows2):
    """Merge two row lists, skipping entries whose ID already appears in rows1.

    Comparison is case-insensitive on the ``entry`` field.

    Returns (merged_rows, n_duplicates_skipped).
    """
    seen = {row['entry'].lower() for row in rows1}
    merged = list(rows1)
    n_dup = 0
    for row in rows2:
        if row['entry'].lower() in seen:
            n_dup += 1
        else:
            seen.add(row['entry'].lower())
            merged.append(row)
    return merged, n_dup


# ---------------------------------------------------------------------------
#  Core merge logic
# ---------------------------------------------------------------------------

def merge_datasets(dir1, dir2, output_dir, val_fraction, test_fraction, seed):
    """Merge the two processed dataset directories and write output files."""

    # ── Read both list.csv files ────────────────────────────────────────────
    list1 = os.path.join(dir1, 'list.csv')
    list2 = os.path.join(dir2, 'list.csv')
    for p in (list1, list2):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing list.csv: {p}")

    print(f"Reading dataset 1: {dir1}")
    rows1 = read_list_csv(list1, dir1)
    clusters1 = len(set(_cluster_key(r) for r in rows1))
    print(f"  {len(rows1):,} entries in {clusters1:,} clusters")

    print(f"Reading dataset 2: {dir2}")
    rows2 = read_list_csv(list2, dir2)
    clusters2 = len(set(_cluster_key(r) for r in rows2))
    print(f"  {len(rows2):,} entries in {clusters2:,} clusters")

    # ── Renumber dataset-2 clusters to avoid ID collisions ─────────────────
    if rows1:
        max_id1 = max(int(_cluster_key(r)) for r in rows1)
    else:
        max_id1 = -1
    rows2, _ = renumber_clusters(rows2, start_id=max_id1 + 1)

    # ── Deduplicate ─────────────────────────────────────────────────────────
    print("Deduplicating entries …")
    merged, n_dup = deduplicate(rows1, rows2)
    if n_dup:
        print(f"  Skipped {n_dup:,} duplicate entries from dataset 2")

    # ── Build train / val / test split ─────────────────────────────────────
    all_cluster_ids = sorted(set(int(_cluster_key(r)) for r in merged))
    rng = random.Random(seed)
    rng.shuffle(all_cluster_ids)

    n_val  = max(1, int(len(all_cluster_ids) * val_fraction))
    n_test = max(1, int(len(all_cluster_ids) * test_fraction))
    val_clusters  = set(all_cluster_ids[:n_val])
    test_clusters = set(all_cluster_ids[n_val:n_val + n_test])
    n_train = len(all_cluster_ids) - n_val - n_test

    # ── Write output files ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ['entry', 'date', 'resolution', 'chain', 'cluster', 'source_dir']
    list_out = os.path.join(output_dir, 'list.csv')
    with open(list_out, 'w', newline='') as f:
        # extrasaction='ignore': row dicts may carry extra keys from a
        # 6-column source list.csv; we only write the canonical columns above.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged)

    with open(os.path.join(output_dir, 'valid_clusters.txt'), 'w') as f:
        f.write('\n'.join(str(c) for c in sorted(val_clusters)) + '\n')

    with open(os.path.join(output_dir, 'test_clusters.txt'), 'w') as f:
        f.write('\n'.join(str(c) for c in sorted(test_clusters)) + '\n')

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Merge complete!")
    print(f"  Dataset 1 entries:   {len(rows1):>8,}")
    print(f"  Dataset 2 entries:   {len(rows2):>8,}  ({n_dup:,} duplicates removed)")
    print(f"  Merged entries:      {len(merged):>8,}")
    print(f"  Total clusters:      {len(all_cluster_ids):>8,}")
    print(f"    Train clusters:    {n_train:>8,}")
    print(f"    Valid clusters:    {n_val:>8,}")
    print(f"    Test  clusters:    {n_test:>8,}")
    print(f"  Output directory:    {output_dir}")
    print(f"{'='*52}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge two processed ProteinMPNN datasets into one",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dir1', required=True,
                        help='First processed dataset directory')
    parser.add_argument('--dir2', required=True,
                        help='Second processed dataset directory')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for merged list and split files')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of clusters for validation')
    parser.add_argument('--test_fraction', type=float, default=0.1,
                        help='Fraction of clusters for test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    args = parser.parse_args()

    merge_datasets(
        dir1=args.dir1,
        dir2=args.dir2,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
