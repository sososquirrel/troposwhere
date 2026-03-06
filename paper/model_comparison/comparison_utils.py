"""
comparison_utils.py — helpers for loading and comparing trained VAE-HMM experiments.
"""

import os
import re
import sys
import pickle

import numpy as np
import torch

import workflow_paper as wfp  # wfp sets up the latent-markov-convection path on import
from model import VAE_HMM

RUNS_DIR = wfp._RUNS

_EXP_PATTERN = re.compile(r'^exp_(\d{8})_\d{4}_[a-f0-9]+$')


def list_experiments(runs_dir: str = RUNS_DIR, from_date: str = None) -> list:
    """
    Return sorted list of experiment directories.
    from_date : optional 'YYYYMMDD' string — only return experiments on or after this date.
    """
    dirs = []
    for name in sorted(os.listdir(runs_dir)):
        m = _EXP_PATTERN.match(name)
        if m:
            if from_date is None or m.group(1) >= from_date.replace('-', ''):
                dirs.append(os.path.join(runs_dir, name))
    return dirs


def read_config(exp_dir: str) -> dict:
    """Read train_config.pkl from an experiment directory (fast, no model loading)."""
    path = os.path.join(exp_dir, 'train_config.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def process_experiment(
    exp_dir: str,
    loader=None,
    input_dim: int = None,
) -> dict:
    """
    Load a trained experiment and run inference.

    Parameters
    ----------
    exp_dir   : path to the experiment directory (must contain best_model.pt + train_config.pkl)
    loader    : optional pre-built DataLoader (avoids re-reading data from disk)
    input_dim : required if loader is provided

    Returns
    -------
    dict with keys: num_states, latent_dim, embeddings, states, emb_pca, pca,
                    state_keep_indices, model
    """
    config = read_config(exp_dir)
    num_states = config['num_states']
    latent_dim = config['latent_dim']
    hidden_dim = config.get('hidden_dim', 512)

    if loader is None or input_dim is None:
        loader, input_dim = wfp.load_dataloader()

    device     = wfp.DEVICE
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pt'),
                            map_location=device, weights_only=True)
    model = VAE_HMM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_states=num_states,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    out        = wfp.extract_latents(model, loader)
    embeddings = out['embeddings']
    states     = out['states']
    emb_pca, pca = wfp.run_pca(embeddings)
    _, state_keep_indices = wfp.keep_closest_latent_samples(
        embeddings, states, num_states=num_states)

    return dict(
        num_states         = num_states,
        latent_dim         = latent_dim,
        embeddings         = embeddings,
        states             = states,
        emb_pca            = emb_pca,
        pca                = pca,
        state_keep_indices = state_keep_indices,
        model              = model,
    )
