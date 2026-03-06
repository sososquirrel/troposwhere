#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from toolbox import straight_through_one_hot_from_probs



# ============================================================
# VAE-HMM model & loss (copied and slightly tidied from your snippet)
class VAE_HMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_states=16, dropout_prob=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_states = num_states

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # State predictor (from z)
        self.state_predictor = nn.Sequential(
            nn.Linear(latent_dim, num_states),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, num_states)
        )

        # Decoder (symmetrical)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_dim)
        )

        self.trans_logits = nn.Parameter(torch.randn(num_states, num_states) * 0.1)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h).clamp(-5, 5)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        logits_state = self.state_predictor(z)
        s_probs = F.softmax(logits_state, dim=-1)
        s_onehot_st, s_argmax = straight_through_one_hot_from_probs(s_probs)

        trans_mat = F.softmax(self.trans_logits, dim=-1)
        recon_x = self.decoder(z)

        return {
            "input_x": x,
            "recon_x": recon_x,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "s_probs": s_probs,
            "s_onehot": s_onehot_st,
            "s_argmax": s_argmax,
            "trans_mat": trans_mat,
            "logits_state": logits_state,
        }