"""
protein_mpnn_run_plddt.py
=========================
Inference entry-point for the **pLDDT-Weighted ProteinMPNN** model.

This script mirrors the interface of the original ``protein_mpnn_run.py``
but loads the pLDDT-weighted model (``ProteinMPNN_pLDDT``) and passes
per-residue confidence scores extracted from B-factor columns.

Supported modes
---------------
* **Design** (default): Generate new amino-acid sequences for a given
  backbone structure.
* **Score-only** (``--score_only 1``): Score an existing sequence against
  the structure.

Usage examples
--------------
Design 8 sequences for a single PDB::

    python protein_mpnn_run_plddt.py \\
        --pdb_path inputs/AF-P04637-F1.pdb \\
        --pdb_path_chains "A" \\
        --out_folder outputs/p53_design \\
        --num_seq_per_target 8 \\
        --sampling_temp "0.1" \\
        --path_to_model_weights weights \\
        --model_name epoch_last

Score the native sequence::

    python protein_mpnn_run_plddt.py \\
        --pdb_path inputs/AF-P04637-F1.pdb \\
        --out_folder outputs/p53_score \\
        --score_only 1 \\
        --path_to_model_weights weights \\
        --model_name epoch_last
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from pdb_utils import parse_pdb_backbone  # shared parser / normaliser


def main(args):
    import json
    import time
    import copy
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
    from model_utils import ProteinMPNN_pLDDT

    # ── Setup ────────────────────────────────────────────────────────────
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    np.random.seed(seed)

    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(ALPHABET, range(21)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_all = args.suppress_print == 0

    # ── Load model ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join(args.path_to_model_weights,
                                   f'{args.model_name}.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ProteinMPNN_pLDDT(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=checkpoint.get('num_edges', 48),
        augment_eps=args.backbone_noise
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if print_all:
        print(f"Loaded pLDDT-weighted model from {checkpoint_path}")
        print(f"  Noise level: {checkpoint.get('noise_level', 'N/A')}")
        print(f"  Num edges: {checkpoint.get('num_edges', 48)}")

    # ── Parse PDB ────────────────────────────────────────────────────────
    pdb_dict = parse_pdb_backbone(args.pdb_path, data_source=args.data_source)
    all_chains = pdb_dict['chain_ids']

    if args.pdb_path_chains:
        designed_chains = args.pdb_path_chains.split()
    else:
        designed_chains = all_chains
    fixed_chains = [c for c in all_chains if c not in designed_chains]

    if print_all:
        print(f"PDB: {pdb_dict['name']}")
        print(f"  Chains: {all_chains}")
        print(f"  Designed: {designed_chains}, Fixed: {fixed_chains}")
        for cid in all_chains:
            seq = pdb_dict[f'seq_chain_{cid}']
            conf = pdb_dict.get(f'confidence_chain_{cid}', [])
            mean_conf = np.mean(conf) if conf else 0
            print(f"  Chain {cid}: {len(seq)} residues, "
                  f"mean confidence: {mean_conf:.3f}")

    # ── Build tensors ────────────────────────────────────────────────────
    all_seqs = []
    all_coords_N, all_coords_CA, all_coords_C, all_coords_O = [], [], [], []
    all_conf = []
    all_chain_mask = []
    all_chain_encoding = []

    for ci, cid in enumerate(all_chains):
        seq = pdb_dict[f'seq_chain_{cid}']
        coords = pdb_dict[f'coords_chain_{cid}']
        conf = pdb_dict.get(f'confidence_chain_{cid}',
                            [1.0] * len(seq))
        L = len(seq)

        all_seqs.append(seq)
        all_coords_N.extend(coords[f'N_chain_{cid}'])
        all_coords_CA.extend(coords[f'CA_chain_{cid}'])
        all_coords_C.extend(coords[f'C_chain_{cid}'])
        all_coords_O.extend(coords[f'O_chain_{cid}'])
        all_conf.extend(conf)

        if cid in designed_chains:
            all_chain_mask.extend([1.0] * L)
        else:
            all_chain_mask.extend([0.0] * L)
        all_chain_encoding.extend([ci + 1] * L)

    full_seq = ''.join(all_seqs)
    L_total = len(full_seq)

    X = np.stack([all_coords_N, all_coords_CA, all_coords_C, all_coords_O],
                 axis=1)  # [L, 4, 3]
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    isnan = torch.isnan(X)
    mask = torch.isfinite(X.sum(dim=(2, 3))).float()
    X[isnan] = 0.0

    S = torch.tensor([alphabet_dict.get(aa, 20) for aa in full_seq],
                     dtype=torch.long).unsqueeze(0).to(device)
    chain_M = torch.tensor(all_chain_mask,
                           dtype=torch.float32).unsqueeze(0).to(device)
    residue_idx = torch.arange(L_total, dtype=torch.long,
                               device=device).unsqueeze(0)
    chain_encoding = torch.tensor(all_chain_encoding,
                                  dtype=torch.long).unsqueeze(0).to(device)
    confidence = torch.tensor(all_conf,
                              dtype=torch.float32).unsqueeze(0).to(device)

    # ── Output directory ─────────────────────────────────────────────────
    os.makedirs(args.out_folder, exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'seqs'), exist_ok=True)

    # ── Score only mode ──────────────────────────────────────────────────
    if args.score_only:
        if print_all:
            print(f"\nScoring native sequence...")
        with torch.no_grad():
            log_probs = model(X, S, mask, chain_M, residue_idx,
                              chain_encoding, confidence)
            mask_for_loss = mask * chain_M
            # NLL per residue
            criterion = torch.nn.NLLLoss(reduction='none')
            nll = criterion(log_probs.squeeze(0), S.squeeze(0))
            score = (nll * mask_for_loss.squeeze(0)).sum() / \
                    mask_for_loss.sum()
            perplexity = torch.exp(score)

        print(f"  Score (NLL): {score.item():.4f}")
        print(f"  Perplexity:  {perplexity.item():.4f}")

        score_file = os.path.join(args.out_folder, 'score_only',
                                  f'{pdb_dict["name"]}.npz')
        os.makedirs(os.path.dirname(score_file), exist_ok=True)
        np.savez(score_file, score=score.cpu().numpy(),
                 perplexity=perplexity.cpu().numpy())
        return

    # ── Design mode ──────────────────────────────────────────────────────
    temperatures = [float(t) for t in args.sampling_temp.split()]
    NUM_BATCHES = max(1, args.num_seq_per_target // args.batch_size)
    BATCH_COPIES = args.batch_size

    ali_file = os.path.join(args.out_folder, 'seqs',
                            f'{pdb_dict["name"]}.fa')

    if print_all:
        print(f"\nDesigning {args.num_seq_per_target} sequences...")

    def _S_to_seq(S_tensor, chain_mask_tensor):
        seq = ''
        for i in range(S_tensor.shape[0]):
            if chain_mask_tensor[i] > 0:
                seq += ALPHABET[S_tensor[i]]
        return seq

    t0 = time.time()
    with open(ali_file, 'w') as f:
        # Write native sequence header
        native_seq = _S_to_seq(S[0], chain_M[0])
        with torch.no_grad():
            log_probs_native = model(X, S, mask, chain_M, residue_idx,
                                     chain_encoding, confidence)
            criterion = torch.nn.NLLLoss(reduction='none')
            nll_native = criterion(log_probs_native.squeeze(0), S.squeeze(0))
            mask_for_loss = mask * chain_M
            native_score = (nll_native * mask_for_loss.squeeze(0)).sum() / \
                           mask_for_loss.sum()

        f.write(f'>{pdb_dict["name"]}, score={native_score.item():.4f}, '
                f'designed_chains={designed_chains}, '
                f'fixed_chains={fixed_chains}, '
                f'model=pLDDT-weighted, seed={seed}\n')
        f.write(f'{native_seq}\n')

        sample_count = 0
        for temp in temperatures:
            for j in range(NUM_BATCHES):
                with torch.no_grad():
                    # Expand for batch
                    X_b = X.expand(BATCH_COPIES, -1, -1, -1)
                    S_b = S.expand(BATCH_COPIES, -1)
                    mask_b = mask.expand(BATCH_COPIES, -1)
                    chain_M_b = chain_M.expand(BATCH_COPIES, -1)
                    ridx_b = residue_idx.expand(BATCH_COPIES, -1)
                    cenc_b = chain_encoding.expand(BATCH_COPIES, -1)
                    conf_b = confidence.expand(BATCH_COPIES, -1)

                    # Get log probs
                    log_probs = model(X_b, S_b, mask_b, chain_M_b, ridx_b,
                                      cenc_b, conf_b)

                    # Sample
                    probs = F.softmax(log_probs / temp, dim=-1)
                    S_sample = torch.multinomial(
                        probs.view(-1, 21), 1).view(BATCH_COPIES, L_total)

                    # Keep fixed positions
                    S_sample = (chain_M_b.long() * S_sample +
                                (1 - chain_M_b.long()) * S_b)

                    # Score samples
                    log_probs_sample = model(X_b, S_sample, mask_b,
                                             chain_M_b, ridx_b, cenc_b,
                                             conf_b)
                    nll_sample = criterion(
                        log_probs_sample.view(-1, 21),
                        S_sample.view(-1)
                    ).view(BATCH_COPIES, L_total)
                    scores = (nll_sample * mask_for_loss.expand(
                        BATCH_COPIES, -1)).sum(-1) / mask_for_loss.sum()

                    # Recovery
                    recovery = ((S_sample == S_b).float() *
                                mask_for_loss.expand(BATCH_COPIES, -1)
                                ).sum(-1) / mask_for_loss.sum()

                for b in range(BATCH_COPIES):
                    sample_count += 1
                    seq = _S_to_seq(S_sample[b], chain_M[0])
                    score_val = scores[b].item()
                    rec_val = recovery[b].item()
                    f.write(f'>T={temp}, sample={sample_count}, '
                            f'score={score_val:.4f}, '
                            f'seq_recovery={rec_val:.4f}\n')
                    f.write(f'{seq}\n')

    t1 = time.time()
    if print_all:
        print(f"Generated {sample_count} sequences in {t1-t0:.2f}s")
        print(f"Output: {ali_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="pLDDT-Weighted ProteinMPNN sequence design")

    argparser.add_argument("--suppress_print", type=int, default=0)
    argparser.add_argument("--path_to_model_weights", type=str, required=True,
                           help="Path to folder containing model .pt files")
    argparser.add_argument("--model_name", type=str, default="epoch_last",
                           help="Model checkpoint name (without .pt)")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--backbone_noise", type=float, default=0.0)
    argparser.add_argument("--num_seq_per_target", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--sampling_temp", type=str, default="0.1")
    argparser.add_argument("--out_folder", type=str, required=True)
    argparser.add_argument("--pdb_path", type=str, required=True)
    argparser.add_argument("--pdb_path_chains", type=str, default='')
    argparser.add_argument("--score_only", type=int, default=0)
    argparser.add_argument("--data_source", type=str, default="auto",
                           choices=["auto", "alphafold", "xray"],
                           help="How to interpret B-factor column: "
                                "'alphafold' divides by 100 (pLDDT scale); "
                                "'xray' inverts via min-max; "
                                "'auto' infers from AF-style filenames or "
                                "the value profile (default).")

    args = argparser.parse_args()
    main(args)
