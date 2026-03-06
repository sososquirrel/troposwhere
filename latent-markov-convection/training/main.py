# ============================================================
# main.py — Single training run for VAE-HMM
# ============================================================

import os
import sys
import random
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import uuid
from datetime import datetime

# Make models/, training/, and toolbox importable regardless of working directory
_PROJECT_ROOT = '/Users/sophieabramian/Documents/troposwhere/latent-markov-convection'
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'models'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'training'))

from model import VAE_HMM
from losses import dice_contrastive_soft, compute_hmm_vae_loss
from toolbox import (
    inv_log_signed,
    create_image_from_flat_tensor_torch,
    make_three_masks_torch,
)

# ============================================================
# Configuration (AUTHORITATIVE)
# ============================================================

def make_run_name(prefix="exp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    uid = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{uid}"

CONFIG = dict(
    # ---- experiment ----
    exp_name=make_run_name(prefix="exp"),
    seed=42,

    # ---- data ----
    #data_folder="/Volumes/LaCie/000_POSTDOC_2025/long_high_res",
    data_folder="/Users/sophieabramian/Documents/troposwhere/data/ml_data",
    data_file="rho_w_centered_smoothed_log.npy",
    image_size=48,
    data_range=10.0,
    n_time_future=5,

    # ---- model ----
    latent_dim=8,
    hidden_dim=512,
    num_states=7,

    # ---- optimization ----
    optimizer="AdamW",
    lr=5e-5,
    batch_size=256,
    num_epochs=512,
    clip_grad=1.0,

    # ---- loss weights ----
    beta_kl=0.008,
    gamma_hmm=0.3,
    gamma_entropy=0.0, #0.1,
    lambda_dice=10.0, #10.0,

    # ---- dice regularization ----
    dice_thr_coh=0.25,
    dice_thr_sep=0.4,
    num_pairs_triplet=256,
)

BASE_RUN_DIR = '/Users/sophieabramian/Documents/troposwhere/data/runs'

# ============================================================
# Reproducibility & device
# ============================================================

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# Dataset
# ============================================================

class NextStepDataset(Dataset):
    def __init__(self, data, indices, n_time_future):
        self.data = data
        self.indices = indices
        self.n_time_future = n_time_future

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if idx + self.n_time_future >= len(self.data):
            idx = len(self.data) - (self.n_time_future + 1)
        return self.data[idx], self.data[idx + self.n_time_future]


# ============================================================
# Logging utilities
# ============================================================

class EpochAccumulator:
    """Accumulates epoch-level averages without last-batch bias."""

    def __init__(self):
        self.scalars = {}
        self.other = {}

    def add_scalar(self, name, value, n):
        if name not in self.scalars:
            self.scalars[name] = {"sum": 0.0, "count": 0}
        self.scalars[name]["sum"] += float(value) * n
        self.scalars[name]["count"] += n

    def set_other(self, name, value):
        self.other[name] = value

    def summarize(self):
        out = {
            k: v["sum"] / max(1e-12, v["count"])
            for k, v in self.scalars.items()
        }
        out.update(self.other)
        return out


def print_epoch_metrics(split, metrics):
    keys = ["total", "recon", "kl_scaled", "hmm_scaled",
            "dice_loss_scaled", "entropy_unscaled"]
    msg = f"[{split}] "
    for k in keys:
        if k in metrics:
            msg += f"{k}:{metrics[k]:.4f} | "
    if "trans_acc" in metrics:
        msg += f"TRANS_ACC:{metrics['trans_acc']:.4f} | "
    print(msg)


# ============================================================
# Data loading
# ============================================================

data_path = os.path.join(CONFIG["data_folder"], CONFIG["data_file"])
data = np.load(data_path)
full_tensor = torch.tensor(data, dtype=torch.float32)

# Flatten spatial dims: (N, 48, 48) -> (N, 2304)
if full_tensor.dim() > 2:
    full_tensor = full_tensor.flatten(start_dim=1)

input_dim = full_tensor.shape[1]
indices = np.random.permutation(len(full_tensor))

n_train = int(0.90 * len(indices))
n_val = int(0.999 * len(indices))

train_idx = indices[:n_train]
val_idx = indices[n_train:n_val]

train_loader = DataLoader(
    NextStepDataset(full_tensor, train_idx, CONFIG["n_time_future"]),
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    pin_memory=True,
)

val_loader = DataLoader(
    NextStepDataset(full_tensor, val_idx, CONFIG["n_time_future"]),
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    pin_memory=True,
)

# ============================================================
# Model & optimizer
# ============================================================

model = VAE_HMM(
    input_dim=input_dim,
    hidden_dim=CONFIG["hidden_dim"],
    latent_dim=CONFIG["latent_dim"],
    num_states=CONFIG["num_states"],
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])

# ============================================================
# Output directory
# ============================================================

OUT_DIR = os.path.join(BASE_RUN_DIR, CONFIG["exp_name"])
os.makedirs(OUT_DIR, exist_ok=True)

best_ckpt_path = os.path.join(OUT_DIR, "best_model.pt")
best_val = float("inf")

loss_history = {"train": [], "val": []}

# ============================================================
# Training loop
# ============================================================

for epoch in range(CONFIG["num_epochs"]):
    # ---------------- TRAIN ----------------
    model.train()
    train_acc = EpochAccumulator()

    for x_t, x_tp1 in train_loader:
        x_t = x_t.to(DEVICE, non_blocking=True)
        x_tp1 = x_tp1.to(DEVICE, non_blocking=True)
        batch_size = x_t.size(0)

        optimizer.zero_grad()

        out_t = model(x_t)
        out_tp1 = model(x_tp1)

        loss = compute_hmm_vae_loss(
            x_t, x_tp1, out_t, out_tp1,
            beta_kl=CONFIG["beta_kl"],
            gamma_hmm=CONFIG["gamma_hmm"],
            lambda_entropy=CONFIG["gamma_entropy"],
        )

        dice_pen = dice_contrastive_soft(
            x_flat=out_t["input_x"],
            s_probs=out_t["s_probs"],
            num_pairs=CONFIG["num_pairs_triplet"],
            thr_coh=CONFIG["dice_thr_coh"],
            thr_sep=CONFIG["dice_thr_sep"],
            device=DEVICE,
        )

        loss["dice_loss_scaled"] = CONFIG["lambda_dice"] * dice_pen
        loss["total"] = loss["total"] + loss["dice_loss_scaled"]

        loss["total"].backward()
        clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
        optimizer.step()

        with torch.no_grad():
            s_t = out_t["s_argmax"]
            s_tp1_pred = out_t["trans_mat"][s_t].argmax(dim=1)
            trans_acc = (s_tp1_pred == out_tp1["s_argmax"]).float().mean()

            train_acc.add_scalar("total", loss["total"], batch_size)
            train_acc.add_scalar("recon", loss["recon"], batch_size)
            train_acc.add_scalar("kl_scaled", loss["kl_scaled"], batch_size)
            train_acc.add_scalar("hmm_scaled", loss["hmm_scaled"], batch_size)
            train_acc.add_scalar("dice_loss_scaled", loss["dice_loss_scaled"], batch_size)
            train_acc.add_scalar("entropy_unscaled", loss["entropy_unscaled"], batch_size)
            train_acc.add_scalar("trans_acc", trans_acc, batch_size)

    train_metrics = train_acc.summarize()
    loss_history["train"].append(train_metrics)
    print_epoch_metrics("train", train_metrics)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_acc = EpochAccumulator()

    with torch.no_grad():
        for x_t, x_tp1 in val_loader:
            x_t = x_t.to(DEVICE, non_blocking=True)
            x_tp1 = x_tp1.to(DEVICE, non_blocking=True)
            batch_size = x_t.size(0)

            out_t = model(x_t)
            out_tp1 = model(x_tp1)

            loss = compute_hmm_vae_loss(
                x_t, x_tp1, out_t, out_tp1,
                beta_kl=CONFIG["beta_kl"],
                gamma_hmm=CONFIG["gamma_hmm"],
                lambda_entropy=CONFIG["gamma_entropy"],
            )

            dice_pen = dice_contrastive_soft(
                x_flat=out_t["input_x"],
                s_probs=out_t["s_probs"],
                num_pairs=CONFIG["num_pairs_triplet"],
                thr_coh=CONFIG["dice_thr_coh"],
                thr_sep=CONFIG["dice_thr_sep"],
                device=DEVICE,
            )

            loss["dice_loss_scaled"] = CONFIG["lambda_dice"] * dice_pen
            loss["total"] = loss["total"] + loss["dice_loss_scaled"]

            s_t = out_t["s_argmax"]
            s_tp1_pred = out_t["trans_mat"][s_t].argmax(dim=1)
            trans_acc = (s_tp1_pred == out_tp1["s_argmax"]).float().mean()

            val_acc.add_scalar("total", loss["total"], batch_size)
            val_acc.add_scalar("recon", loss["recon"], batch_size)
            val_acc.add_scalar("kl_scaled", loss["kl_scaled"], batch_size)
            val_acc.add_scalar("hmm_scaled", loss["hmm_scaled"], batch_size)
            val_acc.add_scalar("dice_loss_scaled", loss["dice_loss_scaled"], batch_size)
            val_acc.add_scalar("entropy_unscaled", loss["entropy_unscaled"], batch_size)
            val_acc.add_scalar("trans_acc", trans_acc, batch_size)

    val_metrics = val_acc.summarize()
    loss_history["val"].append(val_metrics)
    print_epoch_metrics("val", val_metrics)

    if val_metrics["total"] < best_val:
        best_val = val_metrics["total"]
        torch.save({"state_dict": model.state_dict()}, best_ckpt_path)
        print(f"✓ Saved best model (val={best_val:.6f})")

# ============================================================
# Save artifacts
# ============================================================

with open(os.path.join(OUT_DIR, "loss_history.pkl"), "wb") as f:
    pickle.dump(loss_history, f)

with open(os.path.join(OUT_DIR, "train_config.pkl"), "wb") as f:
    pickle.dump(CONFIG, f)

print("Training complete.")