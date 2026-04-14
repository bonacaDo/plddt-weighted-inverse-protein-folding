"""
pdb_utils.py
============
Shared PDB parsing and confidence-score normalisation utilities used by
every script in the pLDDT-weighted ProteinMPNN pipeline.

Keeping these in one place ensures that the same normalisation logic is
applied at training-data prep, inference, demo, and evaluation time.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
#  Amino-acid mapping
# ---------------------------------------------------------------------------

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',   # selenomethionine -> methionine
}


# ---------------------------------------------------------------------------
#  Confidence normalisation
# ---------------------------------------------------------------------------

def infer_confidence_source(bfactors, pdb_path=None):
    """Infer whether B-factor-like values come from AlphaFold or X-ray data.

    Heuristics, in order:
    1. AlphaFold DB files in this project are typically named
       ``AF-<UNIPROT>-F1...``; treat those as AlphaFold.
    2. Otherwise, inspect the value profile. AlphaFold pLDDT values are
       usually bounded in ``[0, 100]`` and concentrated toward the higher
       end, while crystallographic B-factors tend to be lower-confidence
       dispersion values that should be inverted.

    The overlap is not perfectly identifiable from the numbers alone, so
    explicit ``data_source`` remains the most reliable option when known.
    """
    if pdb_path:
        basename = os.path.basename(pdb_path).upper()
        if basename.startswith('AF-') and '-F' in basename:
            return 'alphafold'

    bfacs = np.asarray(bfactors, dtype=np.float64)
    bfacs = bfacs[np.isfinite(bfacs)]
    if len(bfacs) == 0:
        return 'xray'

    if np.all((bfacs >= 0.0) & (bfacs <= 100.0)):
        q1 = float(np.percentile(bfacs, 25))
        median = float(np.median(bfacs))
        if median >= 60.0 and q1 >= 45.0:
            return 'alphafold'

    return 'xray'


def normalise_confidence(bfactors, data_source):
    """Normalise per-residue B-factors / pLDDT values to a [0.05, 1.0]
    confidence score where **1.0 = most confident**.

    Parameters
    ----------
    bfactors : array-like, shape [L]
        Raw per-residue values (CA B-factor or pLDDT depending on source).
    data_source : {'auto', 'alphafold', 'xray'}
        * ``'auto'``       -- infer the source from the value profile.
        * ``'alphafold'`` -- values are pLDDT in [0, 100]; divide by 100
          (high pLDDT = high confidence, no inversion).
        * ``'xray'``      -- values are crystallographic B-factors; perform
          min-max normalisation *and* invert (low B = high confidence).

    Returns
    -------
    conf : np.ndarray, shape [L], dtype float32, values in [0.05, 1.0]
    """
    bfacs = np.asarray(bfactors, dtype=np.float64)
    if len(bfacs) == 0:
        return np.array([], dtype=np.float32)

    bmin, bmax = float(bfacs.min()), float(bfacs.max())

    # Uniform values -> full confidence (avoid division by zero)
    if bmax - bmin < 1e-6:
        return np.ones(len(bfacs), dtype=np.float32)

    if data_source == 'auto':
        data_source = infer_confidence_source(bfacs)

    if data_source == 'alphafold':
        conf = bfacs / 100.0
    elif data_source == 'xray':
        conf = 1.0 - (bfacs - bmin) / (bmax - bmin)
    else:
        raise ValueError("data_source must be 'auto', 'alphafold' or 'xray'")

    return np.clip(conf, 0.05, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
#  PDB backbone parser
# ---------------------------------------------------------------------------

def parse_resseq_sort_key(resseq_str):
    """Parse a PDB residue sequence number that may include an insertion code
    or a negative integer.

    Returns a ``(int_part, insertion_code)`` tuple suitable for sorting.
    """
    s = resseq_str.strip()
    # Match optional minus sign, digit run, optional trailing letter
    negative = s.startswith('-')
    rest = s[1:] if negative else s
    digits = ''
    inscode = ''
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            inscode += ch
    int_part = int(digits) if digits else 0
    if negative:
        int_part = -int_part
    return (int_part, inscode)


# Backward-compatible alias for older imports.
_parse_resseq = parse_resseq_sort_key


def parse_pdb_backbone(pdb_path, data_source='auto'):
    """Parse a PDB file and extract backbone coordinates plus confidence.

    Handles:
    * Standard ATOM records
    * HETATM records for selenomethionine (MSE -> MET)
    * Insertion codes and negative residue numbers
    * Missing backbone atoms (stored as NaN)

    Parameters
    ----------
    pdb_path : str
    data_source : {'auto', 'alphafold', 'xray'}
        Passed to :func:`normalise_confidence`. In ``auto`` mode the parser
        uses the PDB filename as a strong hint for AlphaFold DB entries and
        otherwise falls back to value-based inference.

    Returns
    -------
    pdb_dict : dict
        Keys:
        * ``name``                    -- PDB name (stem of the filename)
        * ``chain_ids``               -- list of chain IDs in file order
        * ``seq_chain_<C>``           -- single-letter sequence for chain C
        * ``coords_chain_<C>``        -- dict with 'N_chain_<C>', 'CA_chain_<C>',
                                         'C_chain_<C>', 'O_chain_<C>' (lists of [x,y,z])
        * ``confidence_chain_<C>``    -- list of float in [0.05, 1.0]
    """
    chains = {}  # chain_id -> per-chain accumulation dict

    with open(pdb_path, 'r') as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            resname = line[17:20].strip()
            if rec == 'HETATM' and resname != 'MSE':
                continue
            if resname not in AA_3TO1:
                continue

            resname_1letter = AA_3TO1[resname]  # MSE -> M via AA_3TO1

            chain_id = line[21]
            atom_name = line[12:16].strip()
            resseq = line[22:27].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfac = float(line[60:66])
            except ValueError:
                continue

            if chain_id not in chains:
                chains[chain_id] = {
                    'seq': [], 'N': [], 'CA': [], 'C': [], 'O': [],
                    'bfactors': [], 'last_resseq': None,
                }

            ch = chains[chain_id]
            if resseq != ch['last_resseq']:
                ch['last_resseq'] = resseq
                ch['seq'].append(resname_1letter)
                ch['bfactors'].append(bfac)
                ch['N'].append([float('nan')] * 3)
                ch['CA'].append([float('nan')] * 3)
                ch['C'].append([float('nan')] * 3)
                ch['O'].append([float('nan')] * 3)

            idx = len(ch['seq']) - 1
            if atom_name == 'N':
                ch['N'][idx] = [x, y, z]
            elif atom_name == 'CA':
                ch['CA'][idx] = [x, y, z]
                ch['bfactors'][idx] = bfac   # prefer CA B-factor for residue
            elif atom_name == 'C':
                ch['C'][idx] = [x, y, z]
            elif atom_name == 'O':
                ch['O'][idx] = [x, y, z]

    pdb_dict = {
        'name': os.path.splitext(os.path.basename(pdb_path))[0],
        'chain_ids': list(chains.keys()),
    }

    for cid, ch in chains.items():
        seq = ''.join(ch['seq'])
        pdb_dict[f'seq_chain_{cid}'] = seq
        pdb_dict[f'coords_chain_{cid}'] = {
            f'N_chain_{cid}':  ch['N'],
            f'CA_chain_{cid}': ch['CA'],
            f'C_chain_{cid}':  ch['C'],
            f'O_chain_{cid}':  ch['O'],
        }
        source = data_source
        if source == 'auto':
            source = infer_confidence_source(ch['bfactors'], pdb_path=pdb_path)
        conf = normalise_confidence(ch['bfactors'], data_source=source)
        pdb_dict[f'confidence_chain_{cid}'] = conf.tolist()

    return pdb_dict
