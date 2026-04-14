"""
pLDDT-Weighted ProteinMPNN  –  training.py
============================================
Training script for the confidence-weighted ProteinMPNN variant.

Usage is identical to the vanilla training script, with one additional
optional flag:

    --from_vanilla_checkpoint  path/to/vanilla_epoch_last.pt

which initialises the model from a vanilla ProteinMPNN checkpoint and
trains only the new confidence-related parameters for a warm-up period
before unfreezing the full model.
"""

import argparse
import os.path


def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor

    from utils import (worker_init_fn, get_pdbs, loader_pdb,
                       build_training_clusters, PDB_dataset,
                       StructureDataset, StructureLoader)
    from model_utils import (featurize, loss_smoothed,
                             loss_smoothed_lowconf_hybrid, loss_nll,
                             get_std_opt, ProteinMPNN_pLDDT)

    use_amp = torch.cuda.is_available() and args.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST":    f"{data_path}/list.csv",
        "VAL":     f"{data_path}/valid_clusters.txt",
        "TEST":    f"{data_path}/test_clusters.txt",
        "DIR":     f"{data_path}",
        "DATCUT":  "2030-Jan-01",
        "RESCUT":  args.rescut,
        "HOMO":    0.70
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 4}

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(
        train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    # --- Instantiate pLDDT-weighted model ---
    model = ProteinMPNN_pLDDT(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise
    )
    model.to(device)

    # --- Optional: initialise from vanilla checkpoint ---
    if args.from_vanilla_checkpoint and os.path.isfile(args.from_vanilla_checkpoint):
        print(f"Loading vanilla checkpoint: {args.from_vanilla_checkpoint}")
        vanilla_ckpt = torch.load(args.from_vanilla_checkpoint, map_location=device)
        vanilla_state = vanilla_ckpt['model_state_dict']

        # Map vanilla EncLayer weights to ConfidenceWeightedEncLayer
        # (they share the same base parameters)
        model_state = model.state_dict()
        loaded = 0
        for k, v in vanilla_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"Loaded {loaded}/{len(vanilla_state)} parameters from vanilla checkpoint")
        print("New confidence parameters will be trained from scratch.")

    if PATH:
        checkpoint = torch.load(PATH, map_location=device)
        total_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Optional: freeze base parameters for warm-up ---
    warmup_epochs = args.warmup_new_params
    if args.from_vanilla_checkpoint and warmup_epochs > 0:
        # Freeze everything except confidence-related parameters
        conf_param_names = {
            'conf_node_embed',
            'conf_edge_gate',
            'conf_gate_alpha_raw',
            'conf_node_alpha',
        }
        for name, param in model.named_parameters():
            if not any(cp in name for cp in conf_param_names):
                param.requires_grad = False
        print(f"Frozen base parameters for {warmup_epochs} warm-up epochs")

    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(
                get_pdbs, train_loader, 1,
                args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(
                get_pdbs, valid_loader, 1,
                args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()

        dataset_train = StructureDataset(pdb_dict_train, truncate=None,
                                         max_length=args.max_protein_length)
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None,
                                         max_length=args.max_protein_length)

        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)

        reload_c = 0
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e

            # Unfreeze after warm-up
            if args.from_vanilla_checkpoint and warmup_epochs > 0 and \
                    e == epoch + warmup_epochs:
                for param in model.parameters():
                    param.requires_grad = True
                print(f"Epoch {e+1}: Unfreezing all parameters")

            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(
                        pdb_dict_train, truncate=None,
                        max_length=args.max_protein_length)
                    loader_train = StructureLoader(
                        dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(
                        pdb_dict_valid, truncate=None,
                        max_length=args.max_protein_length)
                    loader_valid = StructureLoader(
                        dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(
                        get_pdbs, train_loader, 1,
                        args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(
                        get_pdbs, valid_loader, 1,
                        args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1

            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                # Extended featurize returns confidence as 9th element
                X, S, mask, lengths, chain_M, residue_idx, mask_self, \
                    chain_encoding_all, confidence = featurize(batch, device)

                optimizer.zero_grad()
                mask_for_loss = mask * chain_M

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        log_probs = model(X, S, mask, chain_M, residue_idx,
                                          chain_encoding_all, confidence)
                        _, loss_av_smoothed, lowconf_aux_loss = loss_smoothed_lowconf_hybrid(
                            S, log_probs, mask_for_loss, confidence,
                            lowconf_threshold=args.lowconf_threshold,
                            lowconf_aux_weight=args.lowconf_aux_weight,
                            lowconf_power=args.lowconf_power)
                    scaler.scale(loss_av_smoothed).backward()
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx,
                                      chain_encoding_all, confidence)
                    _, loss_av_smoothed, lowconf_aux_loss = loss_smoothed_lowconf_hybrid(
                        S, log_probs, mask_for_loss, confidence,
                        lowconf_threshold=args.lowconf_threshold,
                        lowconf_aux_weight=args.lowconf_aux_weight,
                        lowconf_power=args.lowconf_power)
                    loss_av_smoothed.backward()
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)
                    optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, \
                        chain_encoding_all, confidence = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx,
                                      chain_encoding_all, confidence)
                    mask_for_loss = mask * chain_M
                    loss, loss_av, true_false = loss_nll(
                        S, log_probs, mask_for_loss)
                    validation_sum += torch.sum(
                        loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(
                        true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(
                        mask_for_loss).cpu().data.numpy()

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            train_perplexity_ = np.format_float_positional(
                np.float32(train_perplexity), unique=False, precision=3)
            validation_perplexity_ = np.format_float_positional(
                np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(
                np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(
                np.float32(t1 - t0), unique=False, precision=1)
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, '
                        f'train: {train_perplexity_}, '
                        f'valid: {validation_perplexity_}, '
                        f'train_acc: {train_accuracy_}, '
                        f'valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, '
                  f'train: {train_perplexity_}, '
                  f'valid: {validation_perplexity_}, '
                  f'train_acc: {train_accuracy_}, '
                  f'valid_acc: {validation_accuracy_}')

            checkpoint_filename_last = \
                base_folder + 'model_weights/epoch_last.pt'
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_type': 'plddt_weighted',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
            }, checkpoint_filename_last)

            if (e + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = \
                    base_folder + f'model_weights/epoch{e+1}_step{total_step}.pt'
                torch.save({
                    'epoch': e + 1,
                    'step': total_step,
                    'num_edges': args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_type': 'plddt_weighted',
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str,
                           default="my_path/pdb_2021aug02",
                           help="path for loading training data")
    argparser.add_argument("--path_for_outputs", type=str,
                           default="./exp_plddt",
                           help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="",
                           help="path for previous pLDDT-weighted model checkpoint")
    argparser.add_argument("--from_vanilla_checkpoint", type=str, default="",
                           help="path to vanilla ProteinMPNN checkpoint for init")
    argparser.add_argument("--warmup_new_params", type=int, default=5,
                           help="epochs to train only new confidence params")
    argparser.add_argument("--num_epochs", type=int, default=200)
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10)
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2)
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--max_protein_length", type=int, default=10000)
    argparser.add_argument("--hidden_dim", type=int, default=128)
    argparser.add_argument("--num_encoder_layers", type=int, default=3)
    argparser.add_argument("--num_decoder_layers", type=int, default=3)
    argparser.add_argument("--num_neighbors", type=int, default=48)
    argparser.add_argument("--dropout", type=float, default=0.1)
    argparser.add_argument("--backbone_noise", type=float, default=0.2)
    argparser.add_argument("--rescut", type=float, default=3.5)
    argparser.add_argument("--debug", type=bool, default=False)
    argparser.add_argument("--gradient_norm", type=float, default=-1.0)
    argparser.add_argument("--mixed_precision", type=bool, default=True)
    argparser.add_argument("--lowconf_threshold", type=float, default=0.7,
                           help="residues below this confidence receive auxiliary emphasis")
    argparser.add_argument("--lowconf_aux_weight", type=float, default=0.75,
                           help="strength of the low-confidence auxiliary loss")
    argparser.add_argument("--lowconf_power", type=float, default=1.0,
                           help="shape exponent for the low-confidence emphasis curve")

    args = argparser.parse_args()
    main(args)
