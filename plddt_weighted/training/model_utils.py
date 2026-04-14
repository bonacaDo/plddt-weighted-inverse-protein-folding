"""
pLDDT-Weighted ProteinMPNN  –  model_utils.py
==============================================
Modified ProteinMPNN architecture that incorporates per-residue confidence
scores (B-factors during training, pLDDT at inference) into the GNN message
passing layers.

Key changes vs. vanilla ProteinMPNN
------------------------------------
1. **Directional confidence gating** – For every edge (i, j) in the k-NN
   graph, a scalar residual gate is computed from ``[conf_i, conf_j]`` plus
   a directional prior ``(conf_j - conf_i)``.  This encourages
   high-confidence senders to help low-confidence receivers while remaining
   close to vanilla message passing at initialisation.
2. **Confidence node embedding** – Per-node confidence is projected into a
   learnable embedding and added to the initial node features, scaled by a
   learnable mixing coefficient that starts near **zero** (no perturbation
   at init).
3. **Hybrid low-confidence loss** – Training keeps a vanilla-style global
   loss on all residues, then adds an auxiliary term that focuses
   specifically on low-confidence residues.  This targets the regime we care
   about instead of merely downweighting it.
4. **Neutral handling of missing confidence** – Structures without
   confidence/B-factor metadata receive a neutral confidence value instead of
   being treated as maximally trustworthy.
5. **`featurize` returns confidence** – The batching function now also
   returns a per-residue confidence tensor.

Everything else (decoder, positional encodings, RBF features, …) is kept
identical to the original training code so that the model can be initialised
from vanilla ProteinMPNN checkpoints (new parameters are initialised fresh).
"""

from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


# ---------------------------------------------------------------------------
#  Featurize  (extended with confidence / B-factor)
# ---------------------------------------------------------------------------

def featurize(batch, device):
    """Pack and pad batch into torch tensors.

    Compared to the vanilla version this additionally returns a
    ``confidence`` tensor of shape [B, L_max] with per-residue normalised
    B-factors (0-1 range, higher = more confident).
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    confidence = np.zeros([B, L_max], dtype=np.float32)  # NEW

    init_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet

    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains

        # duplicate-sequence detection (same as vanilla)
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)

        num_chains = b['num_of_chains']
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        chain_conf_list = []  # NEW
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = b[f'coords_chain_{letter}']

            if letter in visible_chains:
                chain_mask = np.zeros(chain_length)
            elif letter in masked_chains:
                chain_mask = np.ones(chain_length)
            else:
                continue

            x_chain = np.stack([chain_coords[c_key] for c_key in
                                [f'N_chain_{letter}', f'CA_chain_{letter}',
                                 f'C_chain_{letter}', f'O_chain_{letter}']], 1)
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(chain_length))

            # --- confidence / B-factor ---
            conf_key = f'confidence_chain_{letter}'
            if conf_key in b:
                chain_conf = np.array(b[conf_key], dtype=np.float32)
                # Clamp to [0, 1] – data scripts normalise B-factors to this range
                chain_conf = np.clip(chain_conf, 0.0, 1.0)
            else:
                # Missing confidence is treated as neutral instead of fully
                # reliable.  This matters for merged datasets where the
                # original ProteinMPNN training set does not carry pLDDT /
                # B-factor-derived confidence.
                chain_conf = np.full(chain_length, 0.5, dtype=np.float32)
            chain_conf_list.append(chain_conf)

            l1 += chain_length
            mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1

        x = np.concatenate(x_chain_list, 0)
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        conf = np.concatenate(chain_conf_list, 0)  # NEW

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant',
                        constant_values=(np.nan,))
        X[i, :, :, :] = x_pad
        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        chain_M[i, :] = m_pad
        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant',
                                    constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad
        conf_pad = np.pad(conf, [[0, L_max - l]], 'constant',
                          constant_values=(0.0,))
        confidence[i, :] = conf_pad

        indices = np.asarray([alphabet.index(a) for a in all_sequence],
                             dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device)
    confidence = torch.from_numpy(confidence).to(
        dtype=torch.float32, device=device)  # NEW

    return X, S, mask, lengths, chain_M, residue_idx, mask_self, \
        chain_encoding_all, confidence


# ---------------------------------------------------------------------------
#  Loss functions
# ---------------------------------------------------------------------------

def loss_nll(S, log_probs, mask):
    """Negative log probabilities."""
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """Negative log probabilities with label smoothing."""
    S_onehot = torch.nn.functional.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av


def loss_smoothed_weighted(S, log_probs, mask, confidence=None, weight=0.1,
                           conf_temperature=0.5, conf_floor=0.2):
    """Negative log probabilities with label smoothing, weighted by confidence.

    When *confidence* is provided the per-residue loss is scaled by a
    **gentle** function of the confidence score:

        w_i = floor + (1 - floor) * conf_i^(1/temperature)

    This ensures:
    - No residue is fully silenced (minimum weight = floor)
    - The weighting curve is softer than raw confidence
    - The denominator uses 2000.0 (same as vanilla) for training stability

    Parameters
    ----------
    S              : LongTensor   [B, L]  ground-truth amino-acid indices
    log_probs      : FloatTensor  [B, L, 21]
    mask           : FloatTensor  [B, L]  1 for valid / designed positions
    confidence     : FloatTensor  [B, L] or None  per-residue confidence [0,1]
    weight         : float   label-smoothing coefficient
    conf_temperature : float  temperature for softening the confidence curve
                       (< 1 = sharper, > 1 = flatter; 0.5 = sqrt)
    conf_floor     : float  minimum weight for any residue (0.2 = 20%)

    Returns
    -------
    loss    : FloatTensor [B, L]   per-residue loss (before masking)
    loss_av : scalar FloatTensor   batch-average loss
    """
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)

    if confidence is not None:
        # Gentle weighting: floor + (1-floor) * conf^(1/temp)
        conf_soft = confidence.clamp(0.0, 1.0).pow(1.0 / max(conf_temperature, 0.01))
        w = conf_floor + (1.0 - conf_floor) * conf_soft
        w = w * mask
        # Use fixed denominator like vanilla for stable training dynamics
        loss_av = torch.sum(loss * w) / 2000.0
    else:
        # Fall back to the fixed denominator used by the original loss_smoothed
        loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av


def loss_smoothed_lowconf_hybrid(
    S,
    log_probs,
    mask,
    confidence=None,
    weight=0.1,
    lowconf_threshold=0.7,
    lowconf_aux_weight=1.0,
    lowconf_power=1.0,
):
    """Vanilla loss plus an auxiliary low-confidence-focused term.

    The base loss keeps the strong ProteinMPNN training signal on *all*
    residues.  An extra auxiliary term then focuses optimisation on residues
    whose confidence falls below ``lowconf_threshold``.

    This is intentionally different from classic confidence weighting:
    instead of telling the model to ignore uncertain regions, we explicitly
    ask it to improve on them while preserving global performance.
    """
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)

    base_loss = torch.sum(loss * mask) / 2000.0

    if confidence is None:
        return loss, base_loss, torch.tensor(0.0, device=log_probs.device)

    # Low-confidence residues receive an auxiliary emphasis proportional to
    # how far below the threshold they are.
    low_strength = ((lowconf_threshold - confidence).clamp(min=0.0)
                    / max(lowconf_threshold, 1e-6))
    low_strength = low_strength.pow(lowconf_power) * mask
    low_denom = low_strength.sum()

    if low_denom.item() < 1e-6:
        aux_loss = torch.tensor(0.0, device=log_probs.device)
    else:
        aux_loss = torch.sum(loss * low_strength) / low_denom

    total_loss = base_loss + lowconf_aux_weight * aux_loss
    return loss, total_loss, aux_loss


# ---------------------------------------------------------------------------
#  Graph helper functions  (unchanged)
# ---------------------------------------------------------------------------

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


# ---------------------------------------------------------------------------
#  Confidence-weighted Encoder Layer  (IMPROVED)
# ---------------------------------------------------------------------------

class ConfidenceWeightedEncLayer(nn.Module):
    """Encoder layer with directional confidence-aware message weighting.

    Messages are modulated only on the node aggregation path.  The gate is
    residual and initialised near identity, so the model starts close to
    vanilla ProteinMPNN while still being able to learn asymmetric
    high-confidence -> low-confidence information flow.
    """

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None,
                conf_weights=None):
        """Forward pass with optional confidence weighting.

        Parameters
        ----------
        conf_weights : Tensor [B, N, K, 1] or None
            Pre-computed confidence gate values for each edge.
            If None, behaves identically to the vanilla EncLayer.
        """
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        # --- Apply confidence gate ONLY to node aggregation messages ---
        if conf_weights is not None:
            h_message = conf_weights * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Edge update: NO confidence gating here (gentler modification)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))

        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


# ---------------------------------------------------------------------------
#  Decoder Layer  (unchanged from vanilla)
# ---------------------------------------------------------------------------

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


# ---------------------------------------------------------------------------
#  Feed-forward & Positional Encodings  (unchanged)
# ---------------------------------------------------------------------------

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0,
                        2 * self.max_relative_feature) * mask + \
            (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(
            d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


# ---------------------------------------------------------------------------
#  Protein Features  (unchanged)
# ---------------------------------------------------------------------------

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust,
                                         np.minimum(self.top_k, X.shape[1]),
                                         dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(O, O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, O, E_idx))
        RBF_all.append(self._get_rbf(O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, O, E_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


# ---------------------------------------------------------------------------
#  pLDDT-Weighted ProteinMPNN  (MAIN MODEL – IMPROVED)
# ---------------------------------------------------------------------------

class ProteinMPNN_pLDDT(nn.Module):
    """ProteinMPNN with confidence-weighted message passing.

    Architecture is identical to vanilla ProteinMPNN except:
    - Encoder uses ``ConfidenceWeightedEncLayer`` instead of ``EncLayer``
    - A per-node confidence embedding is added to initial node features,
      scaled by a **learnable mixing coefficient** (starts at 0)
    - Edge confidence gates are computed once and shared across layers,
      **initialised near 1.0** so the model starts like vanilla

    IMPROVEMENTS over previous versions:
    1. Residual edge gates stay close to identity at init
    2. Directional prior favours high-conf senders helping low-conf receivers
    3. Node embedding is scaled by a learnable alpha (init=0)
    4. Edge gate only modulates node aggregation, not edge updates
    5. Training can use a low-confidence-focused hybrid loss
    """

    def __init__(self, num_letters=21, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features,
                                        top_k=k_neighbors,
                                        augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # --- Confidence node embedding (IMPROVED) ---
        # Project scalar confidence to hidden_dim
        # Scaled by learnable alpha that starts at 0 (no perturbation at init)
        self.conf_node_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        # Learnable mixing coefficient, initialised to 0
        self.conf_node_alpha = nn.Parameter(torch.zeros(1))

        # --- Confidence edge gate (IMPROVED) ---
        # Residual gate centred around identity.  The MLP learns an edge-wise
        # signal, while the directional prior (conf_j - conf_i) pushes the
        # gate to favour high-confidence neighbours for low-confidence nodes.
        self.conf_edge_gate = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Residual strength is bounded to a gentle range so the gate cannot
        # destroy the vanilla message path.  Initial alpha ~= 0.03.
        self.conf_gate_alpha_raw = nn.Parameter(torch.tensor(-2.0))

        # Encoder layers (confidence-weighted)
        self.encoder_layers = nn.ModuleList([
            ConfidenceWeightedEncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers (unchanged)
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Xavier init for all weight matrices
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Re-initialise confidence-specific scalars after xavier.
        with torch.no_grad():
            self.conf_gate_alpha_raw.fill_(-2.0)
            self.conf_node_alpha.fill_(0.0)

    def _compute_conf_weights(self, confidence, E_idx):
        """Compute per-edge confidence gate values.

        Parameters
        ----------
        confidence : Tensor [B, N]  – per-residue confidence in [0, 1]
        E_idx      : Tensor [B, N, K] – neighbor indices

        Returns
        -------
        conf_weights : Tensor [B, N, K, 1]
        """
        B, N = confidence.shape
        K = E_idx.shape[-1]

        conf_i = confidence.unsqueeze(-1).expand(-1, -1, K)  # [B, N, K]
        # Gather neighbor confidences
        conf_j = torch.gather(confidence.unsqueeze(-1).expand(-1, -1, K), 1, E_idx)  # [B, N, K]

        conf_pair = torch.stack([conf_i, conf_j], dim=-1)  # [B, N, K, 2]
        gate_logit = self.conf_edge_gate(conf_pair)  # [B, N, K, 1]

        # Directional prior:
        #   positive when sender is more confident than receiver
        #   negative when a low-confidence sender talks to a high-confidence
        #   receiver
        directional_prior = (conf_j - conf_i).unsqueeze(-1)

        # Residual strength in (0, 0.25); starts very small for a safe
        # vanilla-like warm-start but still lets gradients flow.
        alpha = 0.25 * torch.sigmoid(self.conf_gate_alpha_raw)
        gate_delta = torch.tanh(gate_logit + directional_prior)
        conf_weights = 1.0 + alpha * gate_delta
        conf_weights = conf_weights.clamp(0.75, 1.25)
        return conf_weights

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                confidence=None):
        """Graph-conditioned sequence model with confidence weighting."""
        device = X.device

        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)

        # Initial node features: zero + scaled confidence embedding
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        if confidence is not None:
            conf_embed = self.conf_node_embed(confidence.unsqueeze(-1))  # [B, N, H]
            # Scale by learnable alpha (starts at 0, so no perturbation initially)
            h_V = h_V + self.conf_node_alpha * conf_embed

        h_E = self.W_e(E)

        # Compute confidence edge weights
        conf_weights = None
        if confidence is not None:
            conf_weights = self._compute_conf_weights(confidence, E_idx)

        # Encoder with confidence-weighted message passing
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend, conf_weights,
                use_reentrant=False
            )

        # Autoregressive decoder (unchanged)
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * torch.abs(torch.randn(chain_M.shape, device=device)))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_ESV, mask, use_reentrant=False)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


# ---------------------------------------------------------------------------
#  Optimizer  (unchanged)
# ---------------------------------------------------------------------------

class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)
