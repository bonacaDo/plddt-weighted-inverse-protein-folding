#!/usr/bin/env python3
"""
evaluate_models.py
==================
Compare vanilla ProteinMPNN vs pLDDT-weighted ProteinMPNN on AlphaFold
structures, stratified by pLDDT confidence bins.

Metrics computed:
- Sequence recovery rate (overall and per pLDDT bin)
- Perplexity (overall and per pLDDT bin)
- Per-residue accuracy vs pLDDT correlation

Output:
- CSV with per-protein results
- JSON with aggregate statistics
- Plots (if matplotlib available)

Usage:
    python evaluate_models.py \\
        --vanilla_weights path/to/vanilla.pt \\
        --plddt_weights path/to/plddt_weighted.pt \\
        --test_pdbs path/to/alphafold_eval \\
        --output_dir results/evaluation
"""

import argparse
import inspect
import os
import sys
import json
import csv
import glob
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ProteinMPNN'))

from pdb_utils import parse_pdb_backbone  # shared parser / normaliser

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

PLDDT_BINS = [
    (0, 50, "very_low"),
    (50, 70, "low"),
    (70, 90, "confident"),
    (90, 100, "very_high"),
]


def _forward_model(model, X, S, mask, chain_M, residue_idx, chain_encoding,
                   confidence=None):
    """Handle the different forward signatures used across model variants."""
    forward_params = inspect.signature(model.forward).parameters

    if confidence is not None:
        return model(X, S, mask, chain_M, residue_idx, chain_encoding, confidence)

    if 'randn' in forward_params:
        randn = torch.arange(
            chain_M.shape[-1], device=X.device, dtype=torch.float32
        ).unsqueeze(0).expand_as(chain_M)
        return model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)

    return model(X, S, mask, chain_M, residue_idx, chain_encoding)


def parse_pdb_for_eval(pdb_path):
    """Parse PDB file and extract coordinates, sequence, and B-factors/pLDDT.

    Delegates to the shared :func:`pdb_utils.parse_pdb_backbone` parser so
    that normalisation is consistent across all scripts.  AlphaFold structures
    are detected automatically from AF-style filenames or the B-factor /
    pLDDT value profile (``data_source='auto'``).
    """
    pdb_dict = parse_pdb_backbone(pdb_path, data_source='auto')

    # Flatten chains (eval treats the whole structure as one chain)
    chain_ids = pdb_dict['chain_ids']
    coords_N, coords_CA, coords_C, coords_O = [], [], [], []
    sequence = []
    plddt_list = []

    for cid in chain_ids:
        seq = pdb_dict[f'seq_chain_{cid}']
        coords = pdb_dict[f'coords_chain_{cid}']
        conf = pdb_dict[f'confidence_chain_{cid}']   # already in [0.05, 1.0]
        sequence.append(seq)
        coords_N.extend(coords[f'N_chain_{cid}'])
        coords_CA.extend(coords[f'CA_chain_{cid}'])
        coords_C.extend(coords[f'C_chain_{cid}'])
        coords_O.extend(coords[f'O_chain_{cid}'])
        # Convert confidence back to 0-100 pLDDT scale for binning / display
        plddt_list.extend((np.array(conf) * 100.0).tolist())

    full_seq = ''.join(sequence)
    X = np.stack([coords_N, coords_CA, coords_C, coords_O], axis=1)  # [L,4,3]

    return {
        'seq': full_seq,
        'X': torch.tensor(X, dtype=torch.float32),
        'plddt': np.array(plddt_list, dtype=np.float32),
    }


def score_sequence(model, X, S, mask, residue_idx, chain_encoding,
                   confidence=None, device='cpu'):
    """Score a sequence given a structure using the model."""
    model.eval()
    with torch.no_grad():
        X = X.unsqueeze(0).to(device)
        S = S.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        chain_M = torch.ones_like(mask)
        residue_idx = residue_idx.unsqueeze(0).to(device)
        chain_encoding = chain_encoding.unsqueeze(0).to(device)

        if confidence is not None:
            confidence = confidence.unsqueeze(0).to(device)
        log_probs = _forward_model(
            model, X, S, mask, chain_M, residue_idx, chain_encoding, confidence)

        # Per-residue metrics
        S_pred = torch.argmax(log_probs, dim=-1)  # [1, L]
        correct = (S_pred == S).float().squeeze(0)  # [L]

        # Per-residue NLL
        criterion = torch.nn.NLLLoss(reduction='none')
        nll = criterion(
            log_probs.squeeze(0),  # [L, 21]
            S.squeeze(0)  # [L]
        )

    return correct.cpu().numpy(), nll.cpu().numpy()


def evaluate_on_structure(pdb_path, vanilla_model, plddt_model, device):
    """Evaluate both models on a single structure."""
    data = parse_pdb_for_eval(pdb_path)
    seq = data['seq']
    X = data['X']
    plddt = data['plddt']
    L = len(seq)

    # Prepare tensors
    S = torch.tensor([ALPHABET.index(aa) if aa in ALPHABET else 20
                      for aa in seq], dtype=torch.long)
    mask = torch.isfinite(X.sum(dim=(1, 2))).float()
    residue_idx = torch.arange(L, dtype=torch.long)
    chain_encoding = torch.ones(L, dtype=torch.long)

    # Confidence for the weighted model: plddt is 0-100 display scale; clip
    # to [0.05, 1.0] to match the training-time range from normalise_confidence.
    confidence = torch.tensor(
        np.clip(plddt / 100.0, 0.05, 1.0), dtype=torch.float32)

    # Score with vanilla model
    vanilla_correct, vanilla_nll = score_sequence(
        vanilla_model, X, S, mask, residue_idx, chain_encoding,
        confidence=None, device=device)

    # Score with pLDDT-weighted model
    plddt_correct, plddt_nll = score_sequence(
        plddt_model, X, S, mask, residue_idx, chain_encoding,
        confidence=confidence, device=device)

    # Compute per-bin metrics
    results = {
        'name': os.path.basename(pdb_path),
        'length': L,
        'mean_plddt': float(np.mean(plddt)),
        'vanilla_recovery': float(np.mean(vanilla_correct)),
        'vanilla_perplexity': float(np.exp(np.mean(vanilla_nll))),
        'plddt_recovery': float(np.mean(plddt_correct)),
        'plddt_perplexity': float(np.exp(np.mean(plddt_nll))),
        'bins': {}
    }

    for low, high, name in PLDDT_BINS:
        bin_mask = (plddt >= low) & (plddt < high)
        if bin_mask.sum() > 0:
            results['bins'][name] = {
                'n_residues': int(bin_mask.sum()),
                'vanilla_recovery': float(np.mean(vanilla_correct[bin_mask])),
                'vanilla_perplexity': float(np.exp(np.mean(vanilla_nll[bin_mask]))),
                'plddt_recovery': float(np.mean(plddt_correct[bin_mask])),
                'plddt_perplexity': float(np.exp(np.mean(plddt_nll[bin_mask]))),
            }

    return results


def aggregate_focus_bins(all_results, focus_bins):
    """Aggregate only the selected confidence bins, weighted by residues."""
    focus = {
        'focus_bins': list(focus_bins),
        'focus_n_residues': 0,
        'focus_vanilla_recovery': None,
        'focus_plddt_recovery': None,
        'focus_improvement': None,
        'focus_per_bin': {},
    }

    total_vanilla_correct = 0.0
    total_plddt_correct = 0.0
    total_residues = 0

    for bin_name in focus_bins:
        bin_vanilla_correct = 0.0
        bin_plddt_correct = 0.0
        bin_residues = 0

        for result in all_results:
            if bin_name not in result['bins']:
                continue
            bin_stats = result['bins'][bin_name]
            n_res = bin_stats['n_residues']
            bin_residues += n_res
            bin_vanilla_correct += n_res * bin_stats['vanilla_recovery']
            bin_plddt_correct += n_res * bin_stats['plddt_recovery']

        if bin_residues > 0:
            vanilla_recovery = bin_vanilla_correct / bin_residues
            plddt_recovery = bin_plddt_correct / bin_residues
            focus['focus_per_bin'][bin_name] = {
                'n_residues': int(bin_residues),
                'vanilla_recovery': float(vanilla_recovery),
                'plddt_recovery': float(plddt_recovery),
                'improvement': float(plddt_recovery - vanilla_recovery),
            }
            total_residues += bin_residues
            total_vanilla_correct += bin_vanilla_correct
            total_plddt_correct += bin_plddt_correct

    if total_residues > 0:
        focus['focus_n_residues'] = int(total_residues)
        focus['focus_vanilla_recovery'] = float(
            total_vanilla_correct / total_residues
        )
        focus['focus_plddt_recovery'] = float(
            total_plddt_correct / total_residues
        )
        focus['focus_improvement'] = float(
            focus['focus_plddt_recovery'] - focus['focus_vanilla_recovery']
        )

    return focus


def main(args):
    if args.focus_only and not args.focus_bins:
        raise SystemExit("--focus_only requires --focus_bins")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading vanilla ProteinMPNN...")
    from protein_mpnn_utils import ProteinMPNN
    vanilla_ckpt = torch.load(args.vanilla_weights, map_location=device)
    vanilla_model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=vanilla_ckpt.get('num_edges', 48),
        augment_eps=0.0)
    vanilla_model.load_state_dict(vanilla_ckpt['model_state_dict'])
    vanilla_model.to(device)
    vanilla_model.eval()

    print("Loading pLDDT-weighted ProteinMPNN...")
    from model_utils import ProteinMPNN_pLDDT
    plddt_ckpt = torch.load(args.plddt_weights, map_location=device)
    plddt_model = ProteinMPNN_pLDDT(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=plddt_ckpt.get('num_edges', 48),
        augment_eps=0.0)
    plddt_model.load_state_dict(plddt_ckpt['model_state_dict'])
    plddt_model.to(device)
    plddt_model.eval()

    # Find test PDB files
    pdb_files = sorted(glob.glob(os.path.join(args.test_pdbs, '*.pdb')))
    print(f"Found {len(pdb_files)} test structures")

    all_results = []
    for pdb_path in pdb_files:
        print(f"  Evaluating {os.path.basename(pdb_path)}...", end=" ")
        try:
            result = evaluate_on_structure(pdb_path, vanilla_model,
                                           plddt_model, device)
            all_results.append(result)
            delta = result['plddt_recovery'] - result['vanilla_recovery']
            print(f"vanilla={result['vanilla_recovery']:.3f}, "
                  f"pLDDT={result['plddt_recovery']:.3f}, "
                  f"delta={delta:+.3f}")
        except Exception as e:
            print(f"FAILED: {e}")

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute aggregate statistics (residue-weighted)
    if all_results:
        total_residues = sum(int(r['length']) for r in all_results)
        if total_residues <= 0:
            raise SystemExit("No residues found in evaluation results.")

        vanilla_correct_sum = sum(
            int(r['length']) * float(r['vanilla_recovery'])
            for r in all_results
        )
        plddt_correct_sum = sum(
            int(r['length']) * float(r['plddt_recovery'])
            for r in all_results
        )
        vanilla_nll_sum = sum(
            int(r['length']) * float(np.log(r['vanilla_perplexity']))
            for r in all_results
        )
        plddt_nll_sum = sum(
            int(r['length']) * float(np.log(r['plddt_perplexity']))
            for r in all_results
        )

        agg = {
            'n_structures': len(all_results),
            'n_residues': int(total_residues),
            'vanilla_mean_recovery': float(vanilla_correct_sum / total_residues),
            'plddt_mean_recovery': float(plddt_correct_sum / total_residues),
            'vanilla_mean_perplexity': float(np.exp(vanilla_nll_sum / total_residues)),
            'plddt_mean_perplexity': float(np.exp(plddt_nll_sum / total_residues)),
        }

        # Per-bin aggregates (residue-weighted)
        for _, _, bin_name in PLDDT_BINS:
            bin_residues = 0
            bin_vanilla_correct = 0.0
            bin_plddt_correct = 0.0
            bin_vanilla_nll = 0.0
            bin_plddt_nll = 0.0

            for r in all_results:
                if bin_name in r['bins']:
                    bin_stats = r['bins'][bin_name]
                    n_res = int(bin_stats['n_residues'])
                    bin_residues += n_res
                    bin_vanilla_correct += n_res * float(bin_stats['vanilla_recovery'])
                    bin_plddt_correct += n_res * float(bin_stats['plddt_recovery'])
                    bin_vanilla_nll += n_res * float(np.log(bin_stats['vanilla_perplexity']))
                    bin_plddt_nll += n_res * float(np.log(bin_stats['plddt_perplexity']))

            if bin_residues > 0:
                vanilla_bin_recovery = bin_vanilla_correct / bin_residues
                plddt_bin_recovery = bin_plddt_correct / bin_residues
                agg[f'{bin_name}_n_residues'] = int(bin_residues)
                agg[f'{bin_name}_vanilla_recovery'] = float(vanilla_bin_recovery)
                agg[f'{bin_name}_plddt_recovery'] = float(plddt_bin_recovery)
                agg[f'{bin_name}_vanilla_perplexity'] = float(np.exp(bin_vanilla_nll / bin_residues))
                agg[f'{bin_name}_plddt_perplexity'] = float(np.exp(bin_plddt_nll / bin_residues))
                agg[f'{bin_name}_improvement'] = float(plddt_bin_recovery - vanilla_bin_recovery)

        focus = None
        if args.focus_bins:
            focus = aggregate_focus_bins(all_results, args.focus_bins)
            agg.update(focus)

        agg_path = os.path.join(args.output_dir, 'aggregate_results.json')
        with open(agg_path, 'w') as f:
            json.dump(agg, f, indent=2)

        if not args.focus_only:
            print("\n" + "=" * 60)
            print("  AGGREGATE RESULTS")
            print("=" * 60)
            print(f"  Structures evaluated: {agg['n_structures']}")
            print(f"  Residues evaluated:   {agg['n_residues']}")
            print(f"  Vanilla recovery:     {agg['vanilla_mean_recovery']:.4f}")
            print(f"  pLDDT recovery:       {agg['plddt_mean_recovery']:.4f}")
            print(f"  Improvement:          "
                  f"{agg['plddt_mean_recovery'] - agg['vanilla_mean_recovery']:+.4f}")
            print()
            for _, _, bin_name in PLDDT_BINS:
                if f'{bin_name}_improvement' in agg:
                    print(f"  {bin_name:12s}: "
                          f"improvement = {agg[f'{bin_name}_improvement']:+.4f}")
            print("=" * 60)

        if focus and focus['focus_n_residues'] > 0:
            label = " + ".join(focus['focus_bins'])
            print("\n" + "=" * 60)
            print("  FOCUSED LOW-CONFIDENCE RESULTS")
            print("=" * 60)
            print(f"  Structures evaluated: {agg['n_structures']}")
            print(f"  Focus bins:           {label}")
            print(f"  Focus residues:       {focus['focus_n_residues']}")
            print(f"  Vanilla recovery:     {focus['focus_vanilla_recovery']:.4f}")
            print(f"  pLDDT recovery:       {focus['focus_plddt_recovery']:.4f}")
            print(f"  Improvement:          {focus['focus_improvement']:+.4f}")
            print()
            for bin_name in focus['focus_bins']:
                if bin_name in focus['focus_per_bin']:
                    bin_stats = focus['focus_per_bin'][bin_name]
                    print(f"  {bin_name:12s}: "
                          f"improvement = {bin_stats['improvement']:+.4f} "
                          f"(n={bin_stats['n_residues']})")
            print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vanilla_weights', type=str, required=True)
    parser.add_argument('--plddt_weights', type=str, required=True)
    parser.add_argument('--test_pdbs', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/evaluation')
    parser.add_argument(
        '--focus_bins',
        type=str,
        nargs='+',
        choices=[name for _, _, name in PLDDT_BINS],
        default=None,
        help='Only highlight the selected confidence bins in an extra focused summary.',
    )
    parser.add_argument(
        '--focus_only',
        action='store_true',
        help='Suppress the full aggregate summary and print only the focused-bin summary.',
    )
    args = parser.parse_args()
    main(args)
