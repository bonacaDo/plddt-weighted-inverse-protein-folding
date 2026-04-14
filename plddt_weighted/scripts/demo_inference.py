#!/usr/bin/env python3
"""
demo_inference.py
=================
Quick demo script that shows how to run inference with the pLDDT-weighted
ProteinMPNN model on a single PDB file.

This is useful for:
- Live demos during the hackathon presentation
- Quick testing of the model
- Showing the difference between vanilla and pLDDT-weighted predictions

Usage:
    python demo_inference.py --pdb_file example.pdb --model_weights epoch_last.pt
    python demo_inference.py --pdb_file example.pdb --model_weights epoch_last.pt --temperature 0.1
    python demo_inference.py --pdb_file xray.pdb   --model_weights epoch_last.pt --data_source xray
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from pdb_utils import parse_pdb_backbone  # shared parser / normaliser

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def design_sequence(model, X, confidence, temperature=0.1, device='cpu'):
    """Design a sequence using the pLDDT-weighted model.

    Parameters
    ----------
    model      : ProteinMPNN_pLDDT
    X          : FloatTensor [L, 4, 3]  backbone coordinates (NaN for missing atoms)
    confidence : array-like [L]         normalised confidence in [0.05, 1.0]
    temperature: float                  sampling temperature
    device     : str or torch.device
    """
    model.eval()
    L = X.shape[0]

    X_batch = X.unsqueeze(0).to(device)
    mask = torch.isfinite(X_batch.sum(dim=(2, 3))).float()
    X_batch[torch.isnan(X_batch)] = 0.0
    chain_M = torch.ones_like(mask)
    residue_idx = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
    chain_encoding = torch.ones(1, L, dtype=torch.long, device=device)
    # confidence is already normalised to [0.05, 1.0] by parse_pdb_backbone
    conf = torch.tensor(confidence, dtype=torch.float32,
                        device=device).unsqueeze(0)

    # Use random initial sequence
    S = torch.randint(0, 20, (1, L), device=device)

    with torch.no_grad():
        log_probs = model(X_batch, S, mask, chain_M, residue_idx,
                          chain_encoding, conf)

        # Sample with temperature
        probs = F.softmax(log_probs / temperature, dim=-1)
        S_designed = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)

    designed_seq = ''.join([ALPHABET[i] for i in S_designed.cpu().numpy()])
    return designed_seq, log_probs.squeeze(0).cpu()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse PDB using the shared parser (handles both AF pLDDT and X-ray B-factors)
    print(f"\nParsing {args.pdb_file}  [data_source={args.data_source}]...")
    pdb_dict = parse_pdb_backbone(args.pdb_file, data_source=args.data_source)

    # Flatten all chains into a single sequence / coordinate / confidence array
    chain_ids = pdb_dict['chain_ids']
    all_coords_N, all_coords_CA, all_coords_C, all_coords_O = [], [], [], []
    all_seq = []
    all_conf = []
    for cid in chain_ids:
        seq = pdb_dict[f'seq_chain_{cid}']
        coords = pdb_dict[f'coords_chain_{cid}']
        conf = pdb_dict[f'confidence_chain_{cid}']
        all_seq.append(seq)
        all_coords_N.extend(coords[f'N_chain_{cid}'])
        all_coords_CA.extend(coords[f'CA_chain_{cid}'])
        all_coords_C.extend(coords[f'C_chain_{cid}'])
        all_coords_O.extend(coords[f'O_chain_{cid}'])
        all_conf.extend(conf)

    native_seq = ''.join(all_seq)
    L = len(native_seq)
    confidence = np.array(all_conf, dtype=np.float32)

    X = np.stack([all_coords_N, all_coords_CA, all_coords_C, all_coords_O],
                 axis=1)
    X = torch.tensor(X, dtype=torch.float32)

    print(f"  Length: {L} residues")
    print(f"  Mean confidence: {np.mean(confidence):.3f}")
    print(f"  Native: {native_seq[:60]}{'...' if L > 60 else ''}")

    # Load model
    print(f"\nLoading model from {args.model_weights}...")
    from model_utils import ProteinMPNN_pLDDT

    ckpt = torch.load(args.model_weights, map_location=device)
    model = ProteinMPNN_pLDDT(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        k_neighbors=ckpt.get('num_edges', 48), augment_eps=0.0)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    # Design sequences
    print(f"\nDesigning {args.num_sequences} sequences (T={args.temperature})...")
    results = []
    for i in range(args.num_sequences):
        designed_seq, log_probs = design_sequence(
            model, X, confidence, args.temperature, device)

        # Compute recovery
        recovery = sum(1 for a, b in zip(native_seq, designed_seq) if a == b) / L
        results.append((designed_seq, recovery))

        print(f"  [{i+1}] Recovery: {recovery:.3f}  "
              f"Seq: {designed_seq[:60]}{'...' if L > 60 else ''}")

    # Summary
    recoveries = [r[1] for r in results]
    print(f"\n{'='*60}")
    print(f"  Mean recovery: {np.mean(recoveries):.3f} "
          f"(+/- {np.std(recoveries):.3f})")
    print(f"  Best recovery: {max(recoveries):.3f}")
    print(f"{'='*60}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f">native\n{native_seq}\n")
            for i, (seq, rec) in enumerate(results):
                f.write(f">design_{i+1}_recovery_{rec:.3f}\n{seq}\n")
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num_sequences', type=int, default=5)
    parser.add_argument('--output', type=str, default='',
                        help='Output FASTA file')
    parser.add_argument('--data_source', type=str, default='auto',
                        choices=['auto', 'alphafold', 'xray'],
                        help="How to interpret the B-factor column: "
                             "'alphafold' (pLDDT /100), 'xray' (invert), "
                             "or 'auto' (infer from AF-style filenames or "
                             "the value profile).")
    args = parser.parse_args()
    main(args)
