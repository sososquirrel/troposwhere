
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
from toolbox import inv_log_signed, create_image_from_flat_tensor_torch

# ============================================================
# Dice utilities (torch, vectorized)
def make_three_masks_torch(imgs: torch.Tensor):
    """
    imgs: B x H x W or B x 1 x H x W
    returns masks: B x H x W (int64) with values {0,1,2}
    thresholds identical to your numpy version: < -1.5 ->0, -1.5..1.5->1, >1.5->2
    """
    if imgs.dim() == 4:
        imgs = imgs[:, 0]
    mask = torch.zeros_like(imgs, dtype=torch.int64)
    mask[imgs < -1.5] = 0
    mask[(imgs >= -1.5) & (imgs <= 1.5)] = 1
    mask[imgs > 1.5] = 2
    return mask

def vectorized_macro_dice_from_masks(maskA: torch.Tensor, maskB: torch.Tensor, eps=1e-8):
    """
    maskA, maskB: B x H x W integer masks with classes {0,1,2}
    returns dice_per_example: B tensor representing mean dice over classes.
    """
    B = maskA.shape[0]
    dices = []
    
    # Pre-view to avoid doing it inside the loop
    maskA_flat = maskA.view(B, -1)
    maskB_flat = maskB.view(B, -1)
    
    for c in (0, 1, 2):
        # Create binary masks for class c
        A = (maskA_flat == c).float()
        Bm = (maskB_flat == c).float()
        
        inter = (A * Bm).sum(dim=1)
        A_sum = A.sum(dim=1)
        B_sum = Bm.sum(dim=1)
        denom = A_sum + B_sum
        
        # FIX: Robust check for empty unions
        # If denom < eps, both masks are empty for this class -> Dice = 1.0
        # Otherwise -> 2*inter / denom
        dice_c = torch.where(
            denom > eps, 
            (2.0 * inter) / (denom + eps), 
            torch.ones_like(denom)
        )
        dices.append(dice_c)
        
    dices = torch.stack(dices, dim=1)  # B x 3
    return dices.mean(dim=1)  # B


    
def dice_contrastive_soft(
    x_flat: torch.Tensor,
    s_probs: torch.Tensor,
    num_pairs: int = 128,
    thr_coh: float = 0.3,
    thr_sep: float = 0.5,
    width: float = 0.05,
    mid_strength: float = 0.2,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:

    B = x_flat.shape[0]
    if B < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    # ----------------------------------------------------------
    # PREP
    # ----------------------------------------------------------
    imgs = create_image_from_flat_tensor_torch(inv_log_signed(x_flat))
    masks = make_three_masks_torch(imgs)

    idx_i = torch.randint(0, B, (num_pairs,), device=device)
    idx_j = torch.randint(0, B, (num_pairs,), device=device)

    dice_vals = vectorized_macro_dice_from_masks(masks[idx_i], masks[idx_j])
    dice_dist = 1.0 - dice_vals

    p_same = (s_probs[idx_i] * s_probs[idx_j]).sum(dim=1)
    p_diff = 1.0 - p_same

    # ----------------------------------------------------------
    # LINEAR WINDOWS (no sigmoid)
    # ----------------------------------------------------------
    # Cohesion window: high when dice_dist < thr_coh
    w_coh = torch.clamp((thr_coh - dice_dist) / width, 0.0, 1.0)

    # Separation window: high when dice_dist > thr_sep
    w_sep = torch.clamp((dice_dist - thr_sep) / width, 0.0, 1.0)

    # Mid window: triangular bump between thr_coh and thr_sep
    left = torch.clamp((dice_dist - thr_coh) / width, 0.0, 1.0)
    right = torch.clamp((thr_sep - dice_dist) / width, 0.0, 1.0)
    w_mid = left * right  # simple triangular band-pass

    # ----------------------------------------------------------
    # PENALTIES
    # ----------------------------------------------------------

    cohesion_penalty = w_coh * (1.0 - p_same)            # low dist but diff states
    separation_penalty = w_sep * p_same                  # high dist but same state
    mid_penalty = mid_strength * w_mid * (p_same * p_diff)  # uncertainty penalty

    # ----------------------------------------------------------
    # FINAL LOSS
    # ----------------------------------------------------------
    return (cohesion_penalty + separation_penalty + mid_penalty).mean()


# Loss helpers
def recon_loss_fn(recon_x, x):
    return F.mse_loss(recon_x, x, reduction="mean")

def kl_continuous_z(mu, logvar):
    sigma2 = torch.exp(logvar)
    kl_per_dim = sigma2 + mu.pow(2) - 1.0 - logvar
    kl = 0.5 * torch.sum(kl_per_dim, dim=1)
    return kl.mean()

def hmm_transition_loss(s_probs_t, s_probs_tp1, trans_mat):
    log_trans = torch.log(trans_mat + 1e-10)
    expected_logprob = (s_probs_t.unsqueeze(2) * s_probs_tp1.unsqueeze(1) * log_trans).sum(dim=(1,2))
    return -expected_logprob.mean()

def entropy_regularization(s_probs):
    p = s_probs.mean(dim=0)
    entropy = -(p * (p + 1e-12).log()).sum()
    return entropy

def compute_hmm_vae_loss(
        x_t, x_tp1, out_t, out_tp1,
        beta_kl=1.0, gamma_hmm=1.0,
        lambda_entropy=0.01):
    recon_t = out_t["recon_x"];  mu_t = out_t["mu"];     logvar_t = out_t["logvar"]
    recon_tp1 = out_tp1["recon_x"]; mu_tp1 = out_tp1["mu"]; logvar_tp1 = out_tp1["logvar"]
    s_probs_t = out_t["s_probs"]
    s_probs_tp1 = out_tp1["s_probs"]
    trans_mat = out_t["trans_mat"]

    recon_total = 0.5*(recon_loss_fn(recon_t, x_t) + recon_loss_fn(recon_tp1, x_tp1))
    kl_total    = 0.5*(kl_continuous_z(mu_t, logvar_t) + kl_continuous_z(mu_tp1, logvar_tp1))
    hmm_unscaled = hmm_transition_loss(s_probs_t, s_probs_tp1, trans_mat)
    entropy_term = 0.5*entropy_regularization(s_probs_t)

    total = (recon_total
             + beta_kl*kl_total
             + gamma_hmm*hmm_unscaled
             - lambda_entropy*entropy_term)

    return {
        "total": total,
        "recon": recon_total,
        "kl_scaled": beta_kl*kl_total,
        "hmm_scaled": gamma_hmm*hmm_unscaled,
        "entropy": entropy_term,
        "entropy_scaled": lambda_entropy*entropy_term,
        "entropy_unscaled": entropy_term,
        "kl_unscaled": kl_total,
        "hmm_unscaled": hmm_unscaled,
        "trans_mat": trans_mat.detach(),
    }

# ============================================================
# Metrics utilities (updated to use random-pairs SSIM/Dice evaluation, vectorized)
def compute_metrics_epoch_level(
    accumulators,
    num_samples_seen
):
    """
    Produce epoch-level metrics from accumulators to avoid last-batch bias.
    accumulators is a dict collecting sums/counts per metric across batches.
    """
    out = {}
    # metrics we expect: total_sum, total_count etc.
    # For simple scalar losses we kept sum and count; for lists like state_counts we kept sums
    for k, v in accumulators.items():
        if isinstance(v, dict) and "sum" in v and "count" in v:
            out[k] = v["sum"] / max(1e-12, v["count"])
        else:
            out[k] = v
    # attach sample count
    out["num_samples"] = num_samples_seen
    return out