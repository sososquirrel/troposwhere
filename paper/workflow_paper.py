"""
workflow_paper.py — Shared utilities for paper figure notebooks.

Provides:
- Model loading and latent extraction
- PCA and state analysis
- Image reconstruction helpers
- Shared visualization constants (colors, cmaps)
- Shared smoothing helpers
- Centralised path configuration
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

# Make the VAE-HMM model importable (latent-markov-convection can't be a proper package due to dashes)
_LMC = '/Users/sophieabramian/Documents/troposwhere/latent-markov-convection'
sys.path.insert(0, _LMC)
sys.path.insert(0, _LMC + '/models')

from model import VAE_HMM


# ============================================================
# Paths  (edit here — nowhere else)
# ============================================================

_ROOT      = '/Users/sophieabramian/Documents/troposwhere'
_DATA      = os.path.join(_ROOT, 'data')
_RUNS      = os.path.join(_DATA, 'runs')

PATHS = dict(
    #data_log  = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/smoothed_masked_log.npy',
    data_log      = os.path.join(_DATA, 'ml_data', 'rho_w_centered_smoothed_log.npy'),
    data_raw      = os.path.join(_DATA, 'ml_data', 'rho_w_dataset.npy'),
    z_array       = os.path.join(_DATA, 'squall_lines', 'z_array.txt'),
    pw            = os.path.join(_DATA, 'diagnostics', 'var_PW.npy'),
    prec          = os.path.join(_DATA, 'diagnostics', 'mean_Prec.npy'),
    valid_indices = os.path.join(_DATA, 'ml_data', 'valid_indices.npy'),
    cpi           = os.path.join(_DATA, 'diagnostics', 'CPI_norm.npy'),
)

# Specific trained model used in the paper
MODEL_PATH = os.path.join(_RUNS, 'exp_20260220_1809_7a8b88_paper', 'best_model.pt')

# Figure output directories
FIGURES_DIR    = os.path.join(_ROOT, 'paper', 'figures')
FIGURES_SI_DIR = os.path.join(_ROOT, 'paper', 'figures', 'si')


# ============================================================
# Model / hardware configuration
# ============================================================

BATCH_SIZE = 256
IMAGE_SIZE = 48
KEEP_FRAC  = 0.01

LATENT_DIM  = 8
HIDDEN_DIM  = 512
NUM_STATES  = 7

DEVICE = torch.device(
    'mps'  if torch.backends.mps.is_available()  else
    'cuda' if torch.cuda.is_available()           else
    'cpu'
)


# ============================================================
# Shared visualisation constants
# ============================================================

cluster_colors = {
    0: '#bcbcdd',
    1: '#ff9895',
    2: '#ffe066',
    3: '#1e78b3',
    4: '#9be7c4',
    5: '#5ec962',
    6: '#e02222',
}

state_colors  = np.array([cluster_colors[s] for s in range(NUM_STATES)])
cluster_cmap  = ListedColormap(state_colors)
norm_state    = Normalize(vmin=0, vmax=NUM_STATES - 1)


# ============================================================
# Smoothing helpers  (shared across notebooks)
# ============================================================

def smooth_1d(x, win=10):
    """Box-car smooth a 1-D array."""
    if win <= 1:
        return x
    pad  = win // 2
    xpad = np.pad(x, pad, mode='edge')
    return np.convolve(xpad, np.ones(win) / win, mode='valid')


def smooth_traj(traj, win=5):
    """Box-car smooth a 2-D trajectory (T × 2)."""
    if win <= 1 or len(traj) < win:
        return traj
    pad      = win // 2
    traj_pad = np.pad(traj, ((pad, pad), (0, 0)), mode='edge')
    kernel   = np.ones(win) / win
    xs = np.convolve(traj_pad[:, 0], kernel, mode='valid')
    ys = np.convolve(traj_pad[:, 1], kernel, mode='valid')
    return np.column_stack([xs, ys])


# ============================================================
# Dataset
# ============================================================

class FullDataset(Dataset):
    """Simple dataset wrapper for full time series (no pairs needed)."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor) - 1

    def __getitem__(self, idx):
        return self.tensor[idx]


def load_dataloader(
    data_path: str = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, int]:
    """
    Load dataset from .npy file and return (DataLoader, input_dim).
    Automatically flattens spatial dims: (N, H, W) → (N, H*W).
    """
    if data_path is None:
        data_path = PATHS['data_log']

    data = np.load(data_path)
    if data.ndim > 2:
        data = data.reshape(len(data), -1)

    tensor = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(FullDataset(tensor), batch_size=batch_size, shuffle=False)
    return loader, data.shape[1]


# ============================================================
# Model loading & inference
# ============================================================

def load_model(
    model_path: str = None,
    input_dim: int = None,
    device: torch.device = DEVICE,
) -> VAE_HMM:
    """Load VAE-HMM from a checkpoint (state_dict format)."""
    if model_path is None:
        model_path = MODEL_PATH

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = VAE_HMM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_states=NUM_STATES,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_latents(
    model: VAE_HMM,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> dict:
    """
    Run model on full dataset. Returns:
      embeddings  : (N, latent_dim)  — posterior means
      states      : (N,)             — argmax state
      trans_mat   : (S, S)           — average transition matrix
    """
    all_z, all_states, all_trans = [], [], []

    for x in loader:
        x   = x.to(device)
        out = model(x)
        all_z.append(out['mu'].cpu().numpy())
        all_states.append(out['s_argmax'].cpu().numpy())
        all_trans.append(out['trans_mat'].cpu().numpy())

    return dict(
        embeddings = np.concatenate(all_z,      axis=0),
        states     = np.concatenate(all_states, axis=0),
        trans_mat  = np.mean(np.stack(all_trans), axis=0),
    )


# ============================================================
# PCA
# ============================================================

def run_pca(embeddings: np.ndarray, n_components: int = 2):
    """Run PCA on latent embeddings. Returns (emb_pca, pca)."""
    pca     = PCA(n_components=n_components)
    emb_pca = pca.fit_transform(embeddings)
    return emb_pca, pca


# ============================================================
# State analysis
# ============================================================

def compute_state_centroids_pca(emb_pca, states, num_states=NUM_STATES):
    """Barycenters per state in PCA space."""
    return {
        s: emb_pca[states == s].mean(axis=0)
        for s in range(num_states)
        if (states == s).any()
    }


def representative_indices_pca(emb_pca, states, centroids):
    """Index of the closest point to each centroid in PCA space."""
    reps = {}
    for s, centroid in centroids.items():
        idx  = np.where(states == s)[0]
        d    = np.linalg.norm(emb_pca[idx] - centroid, axis=1)
        reps[s] = idx[np.argmin(d)]
    return reps


def keep_closest_latent_samples(embeddings, states, num_states=NUM_STATES, keep_frac=KEEP_FRAC):
    """Keep the closest `keep_frac` fraction of latent points per state."""
    centroids    = {}
    keep_indices = {}

    for s in range(num_states):
        idx = np.where(states == s)[0]
        if idx.size == 0:
            continue

        z        = embeddings[idx]
        centroid = z.mean(axis=0)
        d        = np.linalg.norm(z - centroid, axis=1)
        k        = max(1, int(np.ceil(keep_frac * idx.size)))

        centroids[s]    = centroid
        keep_indices[s] = idx[np.argsort(d)[:k]]

    return centroids, keep_indices


# ============================================================
# Image reconstruction helpers
# ============================================================

def _load_valid_indices():
    """Load valid pixel indices (lazy, cached on first call)."""
    if not hasattr(_load_valid_indices, '_cache'):
        vi = np.load(PATHS['valid_indices'])
        _load_valid_indices._cache = (vi, torch.from_numpy(vi).long())
    return _load_valid_indices._cache


def create_image_from_flat_tensor_np(x_flat: np.ndarray):
    """Flat NumPy array → (B, H, W) image."""
    valid_indices, _ = _load_valid_indices()
    if x_flat.ndim == 1:
        x_flat = x_flat[None, :]
    out = np.zeros((x_flat.shape[0], IMAGE_SIZE * IMAGE_SIZE))
    out[:, valid_indices] = x_flat
    return out.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)


def create_image_from_flat_tensor_torch(x_flat: torch.Tensor):
    """Flat Torch tensor → (B, 1, H, W) image."""
    _, valid_idx_torch = _load_valid_indices()
    valid_idx_torch    = valid_idx_torch.to(x_flat.device)
    B   = x_flat.shape[0]
    out = torch.zeros(B, IMAGE_SIZE * IMAGE_SIZE, dtype=x_flat.dtype, device=x_flat.device)
    out[:, valid_idx_torch] = x_flat
    return out.view(B, 1, IMAGE_SIZE, IMAGE_SIZE)


# ============================================================
# Physical grid helpers
# ============================================================

def load_physical_grids():
    """Return (S, Z) meshgrid: MSE bins × height."""
    s_array = np.linspace(320, 350, IMAGE_SIZE)
    z_array = np.loadtxt(PATHS['z_array'])[:IMAGE_SIZE]
    return np.meshgrid(s_array, z_array)


def load_z_array():
    """Return height grid (m), length IMAGE_SIZE."""
    return np.loadtxt(PATHS['z_array'])[:IMAGE_SIZE]


# ============================================================
# Misc
# ============================================================

def inv_log_signed_np(x: np.ndarray) -> np.ndarray:
    """Inverse log-signed transform (NumPy)."""
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


# ============================================================
# Transition matrix & stationary distribution
# ============================================================

def compute_transition_matrix(
    states: np.ndarray,
    num_states: int = NUM_STATES,
    future: int = 5,
) -> np.ndarray:
    """Empirical transition matrix with lag `future`."""
    Tmat = np.zeros((num_states, num_states))
    for i in range(len(states) - future):
        Tmat[states[i], states[i + future]] += 1
    row_sums = Tmat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return Tmat / row_sums


def stationary_distribution(Tmat: np.ndarray) -> np.ndarray:
    """Stationary distribution from left eigenvector (eigenvalue = 1)."""
    eigvals, eigvecs = np.linalg.eig(Tmat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


# ============================================================
# Markov transition arrow drawing (Bezier, avoids FancyArrowPatch clipping bug)
# ============================================================

def _bezier_quadratic(p0, ctrl, p1, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1


def _normalized_outgoing_weights(T, i, eps=1e-12, min_prob=0.01):
    row = T[i].copy()
    row[i] = 0.0
    row[row < min_prob] = 0
    denom = row.sum()
    if denom < eps:
        return {}
    return {j: row[j] / denom for j in range(len(row)) if row[j] > 0}


def draw_markov_transitions(
    ax,
    T: np.ndarray,
    centers: dict,
    lw_min: float = 1.2,
    lw_max: float = 6.0,
    min_prob: float = 0.01,
    min_weight: float = 0.05,
    rad: float = 0.18,
    zorder: int = 60,
    exclude_states: frozenset = frozenset({0, 4}),
):
    """Draw Markov transition arrows as Bezier curves."""
    N = T.shape[0]
    T_modif = T.copy()
    for i in exclude_states:
        for j in range(N):
            T_modif[i, j] = 0
            T_modif[j, i] = 0

    for i in range(N):
        if i not in centers:
            continue
        weights = _normalized_outgoing_weights(T_modif, i, min_prob=min_prob)
        for j, w in weights.items():
            if j not in centers or w < min_weight:
                continue
            p0 = np.array(centers[i], float)
            p1 = np.array(centers[j], float)
            d = p1 - p0
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            perp = np.array([-d[1], d[0]]) / dist
            ctrl = (p0 + p1) / 2 + rad * dist * perp
            curve = _bezier_quadratic(p0, ctrl, p1)
            lw = lw_min + 0.1 * w * (lw_max - lw_min)
            ax.plot(curve[:, 0], curve[:, 1], color="black", lw=lw,
                    alpha=0.85, zorder=zorder, solid_capstyle="round")
            ax.annotate("", xy=curve[-1], xytext=curve[-4],
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        fc="black", lw=0, mutation_scale=12),
                        zorder=zorder + 1)


# ============================================================
# Autocorrelation helpers
# ============================================================

def indicator_autocorrelation_normalized(
    states: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Per-state indicator autocorrelation (normalised). Returns (K, max_lag+1)."""
    K = int(states.max()) + 1
    C = np.zeros((K, max_lag + 1))
    for i in range(K):
        Xi = (states == i).astype(float)
        Xi -= Xi.mean()
        var = Xi.var()
        if var == 0:
            continue
        for k in range(max_lag + 1):
            C[i, k] = np.mean(Xi[:-k or None] * Xi[k:]) / var
    return C


def markov_autocorrelation(
    Tmat: np.ndarray,
    pi: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Theoretical autocorrelation predicted by a Markov chain."""
    K = Tmat.shape[0]
    x = np.arange(K)
    mu = np.sum(pi * x)
    var = np.sum(pi * (x - mu) ** 2)
    C = np.zeros(max_lag + 1)
    C[0] = 1.0
    Tk = np.eye(K)
    for lag in range(1, max_lag + 1):
        Tk = Tk @ Tmat
        cov = np.sum(pi * (x - mu) * (Tk @ x - mu))
        C[lag] = cov / var
    return C


def indicator_autocorrelation_state6_cp(
    states: np.ndarray,
    cp: np.ndarray,
    cp_bins: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """CP-conditioned autocorrelation for state 6. Returns (n_bins, max_lag+1)."""
    C = np.zeros((len(cp_bins) - 1, max_lag + 1))
    for b in range(len(cp_bins) - 1):
        I = ((states == 6) & (cp >= cp_bins[b]) & (cp < cp_bins[b + 1])).astype(float)
        I -= I.mean()
        var = I.var()
        if var == 0:
            continue
        for k in range(max_lag + 1):
            C[b, k] = np.mean(I[:-k or None] * I[k:]) / var
    return C


def smooth_gaussian(C: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing along last axis."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(C, sigma=sigma, mode="nearest")
