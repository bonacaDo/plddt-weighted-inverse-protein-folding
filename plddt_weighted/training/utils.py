"""
pLDDT-Weighted ProteinMPNN  –  utils.py
========================================
Data loading utilities extended to propagate per-residue B-factor
confidence through the training pipeline.

Key change vs. vanilla:
- ``get_pdbs`` and ``loader_pdb`` now extract normalised B-factors from
  the chain ``.pt`` files and include them as ``confidence_chain_X`` in
  the training dictionaries.
"""

import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os


# ---------------------------------------------------------------------------
#  Dataset / Loader helpers  (unchanged)
# ---------------------------------------------------------------------------

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None,
                 max_length=100, alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {'bad_chars': 0, 'too_long': 0, 'bad_seq_length': 0}
        self.data = []
        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1
            if truncate is not None and len(self.data) == truncate:
                return
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()



# ---------------------------------------------------------------------------
#  B-factor normalisation helper  (NEW)
# ---------------------------------------------------------------------------

def normalise_bfactors(bfac_tensor):
    """Convert raw B-factors to a 0-1 confidence score per residue.

    Strategy:
    1. Take per-residue mean B-factor (average over atoms), ignoring NaN
       and zero values (missing atoms).
    2. Residues where ALL atoms are NaN/missing get a neutral confidence
       of 0.5 instead of being silently assigned max confidence.
    3. Invert: conf = 1 - bfac_norm  (low B-factor = high confidence).
    4. Clamp to [0.05, 1.0] so no residue is fully zeroed out.

    This mirrors how pLDDT works: high value = high confidence.
    """
    if bfac_tensor is None or bfac_tensor.numel() == 0:
        return None

    # bfac_tensor shape: [L, 14] (per-atom B-factors)
    # Mask out NaN values and zero values (missing atoms)
    bfac = bfac_tensor.float()
    mask = (torch.isfinite(bfac) & (bfac > 0)).float()
    bfac_finite = torch.where(torch.isfinite(bfac) & (bfac > 0), bfac, torch.zeros_like(bfac))
    bfac_sum = bfac_finite.sum(dim=-1)
    bfac_count = mask.sum(dim=-1)

    # Residues with no valid atoms: mark separately, assign neutral score later
    has_valid = bfac_count > 0
    bfac_mean = torch.where(has_valid, bfac_sum / bfac_count.clamp(min=1), torch.zeros_like(bfac_sum))  # [L]

    # Normalise to [0, 1] via min-max within the chain (using only valid residues)
    valid_vals = bfac_mean[has_valid]
    if valid_vals.numel() == 0 or (valid_vals.max() - valid_vals.min()) < 1e-6:
        # All uniform or all missing -> full confidence for valid, neutral for missing
        confidence = torch.where(has_valid, torch.ones_like(bfac_mean), torch.full_like(bfac_mean, 0.5))
        return confidence.clamp(0.05, 1.0)

    bmin = valid_vals.min()
    bmax = valid_vals.max()
    bfac_norm = (bfac_mean - bmin) / (bmax - bmin)  # 0 = lowest B, 1 = highest B
    confidence = 1.0 - bfac_norm                     # Invert: low B = high confidence

    # Residues with all-NaN atoms get neutral confidence (0.5) instead of a
    # spurious max-confidence value that the old mask=(bfac>0) logic produced.
    confidence = torch.where(has_valid, confidence, torch.full_like(confidence, 0.5))

    return confidence.clamp(0.05, 1.0)


# ---------------------------------------------------------------------------
#  get_pdbs  (MODIFIED – now includes confidence)
# ---------------------------------------------------------------------------

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step, t in enumerate(data_loader):
            t = {k: v[0] for k, v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx'] == idx)
                        initial_sequence = "".join(
                            list(np.array(list(t['seq']))[res][0, ]))

                        # Strip His-tags
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:, 6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:, :-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:, :-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:, :-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:, 7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:, 8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:, 9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:, 10:]

                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_' + letter] = "".join(
                                list(np.array(list(t['seq']))[res][0, ]))
                            concat_seq += my_dict['seq_chain_' + letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res, ])[0, ]  # [L, 14, 3]
                            coords_dict_chain['N_chain_' + letter] = \
                                all_atoms[:, 0, :].tolist()
                            coords_dict_chain['CA_chain_' + letter] = \
                                all_atoms[:, 1, :].tolist()
                            coords_dict_chain['C_chain_' + letter] = \
                                all_atoms[:, 2, :].tolist()
                            coords_dict_chain['O_chain_' + letter] = \
                                all_atoms[:, 3, :].tolist()
                            my_dict['coords_chain_' + letter] = coords_dict_chain

                            # --- B-factor confidence (NEW) ---
                            if 'bfac' in t:
                                bfac_chain = t['bfac'][res][0, ]  # [L, 14]
                                if isinstance(bfac_chain, np.ndarray):
                                    bfac_chain = torch.from_numpy(bfac_chain)
                                conf = normalise_bfactors(bfac_chain)
                                if conf is not None:
                                    my_dict['confidence_chain_' + letter] = \
                                        conf.tolist()

                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list


# ---------------------------------------------------------------------------
#  PDB_dataset  (unchanged)
# ---------------------------------------------------------------------------

class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out


# ---------------------------------------------------------------------------
#  loader_pdb  (MODIFIED – now loads B-factors)
# ---------------------------------------------------------------------------

def loader_pdb(item, params):
    pdbid, chid = item[0].split('_')
    # item[2] carries the per-entry source directory when the list.csv has a
    # ``source_dir`` column; fall back to the global DIR otherwise.
    source_dir = item[2] if len(item) > 2 else params['DIR']
    PREFIX = "%s/pdb/%s/%s" % (source_dir, pdbid[1:3], pdbid)

    if not os.path.isfile(PREFIX + ".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX + ".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    asmb_candidates = set([a for a, b in zip(asmb_ids, asmb_chains)
                           if chid in b.split(',')])

    if len(asmb_candidates) < 1:
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        L = len(chain['seq'])
        out = {'seq': chain['seq'],
               'xyz': chain['xyz'],
               'idx': torch.zeros(L).int(),
               'masked': torch.Tensor([0]).int(),
               'label': item[0]}
        # Include B-factors if available
        if 'bfac' in chain:
            out['bfac'] = chain['bfac']
        return out

    asmb_i = random.sample(list(asmb_candidates), 1)
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    chains = {c: torch.load("%s_%s.pt" % (PREFIX, c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    asmb = {}
    asmb_bfac = {}  # NEW
    for k in idx:
        xform = meta['asmb_xform%d' % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
                # Store B-factors for each copy (NEW)
                if 'bfac' in chains[c]:
                    for i in range(xyz_ru.shape[0]):
                        asmb_bfac[(c, k, i)] = chains[c]['bfac']
            except KeyError:
                return {'seq': np.zeros(5)}

    seqid = meta['tm'][chids == chid][0, :, 1]
    homo = set([ch_j for seqid_j, ch_j in zip(seqid, chids)
                if seqid_j > params['HOMO']])

    seq, xyz, idx_list, masked = "", [], [], []
    bfac_list = []  # NEW
    for counter, (k, v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        xyz.append(v)
        idx_list.append(torch.full((v.shape[0],), counter))
        if k[0] in homo:
            masked.append(counter)
        # Collect B-factors (NEW)
        if k in asmb_bfac:
            bfac_list.append(asmb_bfac[k])

    out = {'seq': seq,
           'xyz': torch.cat(xyz, dim=0),
           'idx': torch.cat(idx_list, dim=0),
           'masked': torch.Tensor(masked).int(),
           'label': item[0]}

    if bfac_list:
        try:
            out['bfac'] = torch.cat(bfac_list, dim=0)
        except Exception:
            pass  # If shapes don't match, skip B-factors for this entry

    return out


# ---------------------------------------------------------------------------
#  build_training_clusters  (unchanged)
# ---------------------------------------------------------------------------

def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])

    if debug:
        val_ids = []
        test_ids = []

    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = []
        for r in reader:
            if float(r[2]) > params['RESCUT']:
                continue
            if parser.parse(r[1]) > parser.parse(params['DATCUT']):
                continue
            # Optional 6th column: per-entry source directory.
            # Falls back to the global DIR when absent.
            source_dir = (r[5].strip() if (len(r) > 5 and r[5].strip())
                          else params['DIR'])
            rows.append([r[0], r[3], int(r[4]), source_dir])

    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2] + [r[3]])
            else:
                valid[r[2]] = [r[:2] + [r[3]]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2] + [r[3]])
            else:
                test[r[2]] = [r[:2] + [r[3]]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2] + [r[3]])
            else:
                train[r[2]] = [r[:2] + [r[3]]]
    if debug:
        valid = train
    return train, valid, test
