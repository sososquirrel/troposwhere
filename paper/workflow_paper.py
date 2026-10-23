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

# Figure output directories — notebooks write directly into the manuscript folder
# (single source of truth). LaTeX's \graphicspath in main.tex / supplementary.tex
# already lists figures/ relative to the manuscript root.
FIGURES_DIR    = os.path.join(_ROOT, 'paper', 'manuscript', 'figures')
FIGURES_SI_DIR = os.path.join(_ROOT, 'paper', 'manuscript', 'figures')


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

# Display offset: data is 0-indexed; figures/manuscript use 1-indexed labels.
STATE_DISPLAY_OFFSET = 1

def state_label(s):
    """Display label for a state index. Data stays 0-indexed; figures show 1..NUM_STATES."""
    return str(int(s) + STATE_DISPLAY_OFFSET)


def panel_label(ax, label, x=0.02, y=0.97, fontsize=14, top=False):
    """Add a bold panel label (a, b, c, ...) to a subplot.

    By default the label sits in the top-left inside the axes. With
    ``top=True`` it is placed just above the axes frame (top-left), which
    reads as a panel header on top of the figure.
    """
    if top:
        ax.text(-0.01, 1.03, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold', va='bottom', ha='left')
    else:
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold', va='top', ha='left')


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
    # The Perron eigenvector is sign-arbitrary; flip so it is mostly positive
    # before clamping (otherwise an all-negative eigenvector is zeroed → NaN).
    if pi.sum() < 0:
        pi = -pi
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


def eigendecomposition(Tmat: np.ndarray):
    """Right eigendecomposition of a transition matrix, sorted by descending |λ|.

    Returns:
        eigvals: complex (K,)  — eigenvalues of Tmat.
        eigvecs: complex (K, K) — right eigenvectors as columns, same ordering.
    """
    eigvals, eigvecs = np.linalg.eig(Tmat)
    order = np.argsort(-np.abs(eigvals))
    return eigvals[order], eigvecs[:, order]


def relaxation_times(eigvals: np.ndarray, dt_minutes: float) -> np.ndarray:
    """Exponential relaxation time τ_k = -dt / log(|λ_k|).

    Stationary mode (|λ| = 1) returns inf. Use with eigvals from eigendecomposition().
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = -dt_minutes / np.log(np.abs(eigvals))
    return tau


def survival_conditional_on_cp(
    states: np.ndarray,
    cp: np.ndarray,
    target_state: int,
    max_lag: int,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> dict:
    """Empirical survival probability inside `target_state`, conditioned on CP at start.

    For each lag k = 0, 1, ..., max_lag, computes
        S_bin(k) = P( s_{t+k} = target_state | s_t = target_state, CP_t in bin )
    for two bins of CP at time t: 'low' (<= q-th quantile of CP at state-t timesteps)
    and 'high' (>= q-th quantile). Also computes the unconditional curve.

    The CP condition applies only at the starting time t, so no "chopping" of the
    indicator series — the survival is well-defined throughout the lag window.

    Returns dict with keys 'low', 'high', 'unc', each a tuple (S, n):
        S: np.ndarray of length max_lag+1, with S[0] = 1
        n: number of starting points used
    """
    states = np.asarray(states)
    cp     = np.asarray(cp, dtype=float)
    L      = min(len(states), len(cp))
    states, cp = states[:L], cp[:L]

    valid_starts = np.where(states == target_state)[0]
    valid_starts = valid_starts[valid_starts + max_lag < L]

    if len(valid_starts) == 0:
        z = np.zeros(max_lag + 1)
        return {'low': (z, 0), 'high': (z, 0), 'unc': (z, 0)}

    cp_at_starts = cp[valid_starts]
    low_thr      = np.quantile(cp_at_starts, low_q)
    high_thr     = np.quantile(cp_at_starts, high_q)

    starts_low  = valid_starts[cp_at_starts <= low_thr]
    starts_high = valid_starts[cp_at_starts >= high_thr]

    def _survival(idx):
        n = len(idx)
        if n == 0:
            return np.zeros(max_lag + 1), 0
        S = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            S[k] = np.mean(states[idx + k] == target_state)
        return S, n

    return {
        'low':  _survival(starts_low),
        'high': _survival(starts_high),
        'unc':  _survival(valid_starts),
    }


def markov_survival(Tmat: np.ndarray, target_state: int, max_lag: int) -> np.ndarray:
    """Markov-predicted survival probability P(s_{t+k}=i | s_t=i) = [T^k]_{ii}.

    Note: this includes trajectories that left state i and came back, which is
    fine for comparing with the empirical estimator above (which does the same).
    """
    T = np.asarray(Tmat)
    n_states = T.shape[0]
    S = np.empty(max_lag + 1)
    Tk = np.eye(n_states)
    for k in range(max_lag + 1):
        S[k] = Tk[target_state, target_state]
        Tk = Tk @ T
    return S


def load_cp_lifecycle_indices(
    timeseries_csv: str,
    summary_csv: str,
    young_age_threshold: int = 5,
    n_timesteps: int = None,
) -> dict:
    """Build age-stratified cold-pool indices from the per-CP lifecycle CSVs.

    `timeseries_csv` provides one row per (CP, timestep) with persistent label_id.
    `summary_csv` provides one row per CP with start_time, duration, etc.
    We join the two and compute age = current_time - start_time for each entry.

    Returns dict of numpy arrays, all of length `n_timesteps` (or max global_time+1):
        'area'            : total cold-pool area  (matches the original CPI definition)
        'intensity'       : sum of |Δqv| across CPs
        'area_x_int'      : sum of area * |Δqv| across CPs (intensity-weighted area)
        'young'           : sum of area * |Δqv| only for CPs with age < young_age_threshold
                            -- proxy for "active / fresh" cold pools (still building up)
        'old'             : sum of area * |Δqv| for CPs with age >= young_age_threshold
                            -- proxy for "decaying / mature" cold pools
        'specific_intensity': sum of |Δqv| / sqrt(area)
                            -- proxy for "small AND intense": high if CPs are physically
                            compact but strongly desiccated; low if they're spread out
                            and weak
        'mean_age'        : intensity-weighted mean age of CPs alive at each timestep
                            -- low when convection is bursting new CPs, high when CPs
                            are old and spreading
    """
    import pandas as pd
    ts = pd.read_csv(timeseries_csv)
    sm = pd.read_csv(summary_csv)
    ts = ts.merge(sm[['simulation', 'label_id', 'start_time']],
                  on=['simulation', 'label_id'])
    ts['age'] = ts['time'] - ts['start_time']

    # Always allocate enough room to hold the maximum global_time index.
    needed = int(ts['global_time'].max()) + 1
    if n_timesteps is None or n_timesteps < needed:
        n_timesteps = needed

    gt    = ts['global_time'].values.astype(int)
    sz    = ts['size'].values.astype(float)
    aqv   = ts['anomaly_qv'].abs().values.astype(float)
    age_a = ts['age'].values.astype(float)
    out = {}

    def _sum_by_time(values, mask=None):
        a = np.zeros(n_timesteps, dtype=float)
        if mask is None:
            np.add.at(a, gt, values)
        else:
            np.add.at(a, gt[mask], values[mask])
        return a

    out['area']       = _sum_by_time(sz)
    out['intensity']  = _sum_by_time(aqv)
    out['area_x_int'] = _sum_by_time(sz * aqv)
    mask_young = age_a < young_age_threshold
    mask_old   = age_a >= young_age_threshold
    out['young']      = _sum_by_time(sz * aqv, mask=mask_young)
    out['old']        = _sum_by_time(sz * aqv, mask=mask_old)
    out['specific_intensity'] = _sum_by_time(aqv / np.sqrt(sz + 1e-9))

    # Population-level indices (probe cold-pool collisions / count, not aggregate intensity)
    ones = np.ones_like(sz)
    out['count']        = _sum_by_time(ones)                    # # of CPs alive at t
    out['count_mature'] = _sum_by_time(ones, mask=mask_old)     # only mature (collision-ready)
    out['count_births'] = _sum_by_time(ones, mask=(age_a == 0)) # # CPs born at t (fresh activity)
    # mean size = total area / count, zero if no CPs
    out['mean_size']    = np.divide(out['area'], out['count'],
                                     out=np.zeros_like(out['area']),
                                     where=out['count'] > 0)

    # Dominant-CP indices: probes "is the leading CP still strong?"
    # max_intensity:        across all CPs alive at t, the largest |Δq_v|
    # intensity_of_largest: |Δq_v| of the largest-area CP at t
    # area_of_largest:      area of the largest-area CP at t
    out['max_intensity']        = np.zeros(n_timesteps, dtype=float)
    out['intensity_of_largest'] = np.zeros(n_timesteps, dtype=float)
    out['area_of_largest']      = np.zeros(n_timesteps, dtype=float)
    out['age_of_largest']       = np.zeros(n_timesteps, dtype=float)
    # Group by global_time using sorted approach (faster than pandas groupby)
    order = np.argsort(gt)
    gt_s, sz_s, aqv_s, age_s = gt[order], sz[order], aqv[order], age_a[order]
    starts = np.searchsorted(gt_s, np.arange(n_timesteps), side="left")
    ends   = np.searchsorted(gt_s, np.arange(n_timesteps), side="right")
    for t in range(n_timesteps):
        i0, i1 = starts[t], ends[t]
        if i1 > i0:
            sz_t  = sz_s[i0:i1]
            aqv_t = aqv_s[i0:i1]
            age_t = age_s[i0:i1]
            out['max_intensity'][t]        = aqv_t.max()
            j_max = int(np.argmax(sz_t))
            out['intensity_of_largest'][t] = aqv_t[j_max]
            out['area_of_largest'][t]      = sz_t[j_max]
            out['age_of_largest'][t]       = age_t[j_max]

    # Intensity-weighted mean age per timestep
    weight = sz * aqv
    age_weighted_sum = _sum_by_time(weight * age_a)
    total_weight = _sum_by_time(weight)
    out['mean_age'] = np.divide(age_weighted_sum, total_weight,
                                 out=np.zeros_like(age_weighted_sum),
                                 where=total_weight > 0)
    return out


def load_cp_index_variants(pickle_path: str) -> dict:
    """Build alternative cold-pool index variants from the per-CP pickle.

    The pickle is a list of dicts, each describing one cold pool at one time
    step, with keys: 'ts', 'sim', 'area', 'anomaly_qv', 'max_anomaly_qv'.

    Returns a dict of numpy arrays, all of length max_ts + 1:
        'area'         — sum of cold-pool areas at each ts (= current CPI definition)
        'intensity'    — sum of |anomaly_qv| at each ts (intensity-only, area-agnostic)
        'area_x_int'   — sum of area * |anomaly_qv| (intensity-weighted area)
        'max_int'      — sum of max_anomaly_qv (peak-intensity proxy)
        'count'        — number of cold pools alive at each ts
        'mean_int'     — mean |anomaly_qv| across cold pools at each ts (0 if none)
    """
    import pickle as _pickle
    with open(pickle_path, 'rb') as f:
        data = _pickle.load(f)
    if not data:
        return {}
    max_ts = max(cp['ts'] for cp in data)
    n = max_ts + 1
    out = {k: np.zeros(n, dtype=float) for k in
           ['area', 'intensity', 'area_x_int', 'max_int', 'count']}
    for cp in data:
        t = cp['ts']
        a = float(cp['area'])
        i = abs(float(cp['anomaly_qv']))
        m = abs(float(cp['max_anomaly_qv']))
        out['area'][t]       += a
        out['intensity'][t]  += i
        out['area_x_int'][t] += a * i
        out['max_int'][t]    += m
        out['count'][t]      += 1
    out['mean_int'] = np.divide(out['intensity'], out['count'],
                                 out=np.zeros_like(out['intensity']),
                                 where=out['count'] > 0)
    return out


def composite_around_events(
    event_times: np.ndarray,
    signal: np.ndarray,
    window: int,
) -> tuple:
    """Lagged composite of `signal` aligned on arbitrary event times.

    Same logic as composite_around_entries but for a generic event-time array.
    Returns (lags, mean, sem, n).
    """
    signal = np.asarray(signal, dtype=float)
    event_times = np.asarray(event_times, dtype=int)
    valid = (event_times >= window) & (event_times + window + 1 <= len(signal))
    event_times = event_times[valid]
    n = len(event_times)
    lags = np.arange(-window, window + 1)
    if n == 0:
        return lags, np.zeros_like(lags, dtype=float), np.zeros_like(lags, dtype=float), 0
    comps = np.stack([signal[t - window : t + window + 1] for t in event_times])
    mean = comps.mean(axis=0)
    sem  = comps.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return lags, mean, sem, n


def state_entry_times(state_series: np.ndarray, target_state: int) -> np.ndarray:
    """Time indices t where state_series[t] == target_state and state_series[t-1] != target_state."""
    s = np.asarray(state_series)
    return np.where((s[1:] == target_state) & (s[:-1] != target_state))[0] + 1


def state_exit_times(state_series: np.ndarray, target_state: int) -> np.ndarray:
    """Time indices t where state_series[t] != target_state and state_series[t-1] == target_state."""
    s = np.asarray(state_series)
    return np.where((s[1:] != target_state) & (s[:-1] == target_state))[0] + 1


def state_presence_times(state_series: np.ndarray, target_state: int) -> np.ndarray:
    """All time indices t where state_series[t] == target_state."""
    return np.where(np.asarray(state_series) == target_state)[0]


def composite_around_entries(
    state_series: np.ndarray,
    signal: np.ndarray,
    target_state: int,
    window: int,
) -> tuple:
    """Lagged composite of `signal` aligned on entries into `target_state`.

    An "entry" is a time step t with state_series[t]=target_state and
    state_series[t-1]!=target_state. For each entry with a full window
    available, we extract signal[t-window : t+window+1] and average.

    Returns:
        lags:     np.arange(-window, window+1)
        mean:     mean composite (length 2*window+1)
        sem:      standard error of the mean per lag (length 2*window+1)
        n:        number of entry events used
    """
    state_series = np.asarray(state_series)
    signal       = np.asarray(signal, dtype=float)
    # Entry times: first index of each run of target_state
    entries = np.where(
        (state_series[1:] == target_state) & (state_series[:-1] != target_state)
    )[0] + 1
    valid   = (entries >= window) & (entries + window + 1 <= len(signal))
    entries = entries[valid]
    n       = len(entries)
    lags    = np.arange(-window, window + 1)

    if n == 0:
        return lags, np.zeros_like(lags, dtype=float), np.zeros_like(lags, dtype=float), 0

    comps = np.stack([signal[t - window : t + window + 1] for t in entries])
    mean  = comps.mean(axis=0)
    sem   = comps.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return lags, mean, sem, n


def autocorrelation_continuous(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of a (zero-mean-centred) 1-D time series.
    Returns C with C[0]=1, length max_lag+1.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return np.zeros(max_lag + 1)
    C = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        C[k] = np.mean(x[:-k or None] * x[k:]) / var
    return C


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple:
    """Symmetric cross-correlation of two 1-D time series.

    Computes Pearson correlation of (x_t, y_{t+tau}) for tau in [-max_lag, +max_lag].
    Positive tau means y leads x (i.e., y at time t+tau correlates with x at time t).

    Returns:
        lags: np.ndarray of length 2*max_lag+1
        C:    np.ndarray of cross-correlation values
    """
    x = np.asarray(x, dtype=float); x = x - x.mean()
    y = np.asarray(y, dtype=float); y = y - y.mean()
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.arange(-max_lag, max_lag + 1), np.zeros(2 * max_lag + 1)
    lags = np.arange(-max_lag, max_lag + 1)
    C = np.empty(lags.size)
    for i, tau in enumerate(lags):
        if tau >= 0:
            C[i] = np.mean(x[:len(x) - tau] * y[tau:]) / (sx * sy)
        else:
            C[i] = np.mean(x[-tau:] * y[:len(y) + tau]) / (sx * sy)
    return lags, C


def markov_autocorrelation_indicator(
    Tmat: np.ndarray,
    pi: np.ndarray,
    state_idx: int,
    max_lag: int,
) -> np.ndarray:
    """Theoretical autocorrelation of the indicator I_i(t) = 1[X_t = i]
    under a stationary Markov chain with transition T and stationary distribution pi.

    C_i(τ) = (T^τ[i,i] - π_i) / (1 - π_i)
    """
    pi_i = pi[state_idx]
    if pi_i >= 1.0 - 1e-12:
        return np.ones(max_lag + 1)
    C = np.zeros(max_lag + 1)
    Tk = np.eye(Tmat.shape[0])           # T^0
    for k in range(max_lag + 1):
        C[k] = (Tk[state_idx, state_idx] - pi_i) / (1.0 - pi_i)
        Tk = Tk @ Tmat                    # now T^(k+1) for next iter
    return C


def compute_conditional_transition_matrix(
    states: np.ndarray,
    cp: np.ndarray,
    cp_bins: np.ndarray,
    num_states: int,
    future: int = 1,
    min_window_len: int = 20,
) -> tuple:
    """Per-bin transition matrix built only from in-window transitions.

    For each CP bin, find contiguous time spans where cp[t] stays in the bin
    for at least `min_window_len` consecutive steps, collect (s_t, s_{t+future})
    pairs *inside* those spans, and build a row-stochastic transition matrix.

    Returns:
        T_list:        list of (num_states, num_states) row-stochastic matrices,
                       one per CP bin.
        pi_list:       list of (num_states,) stationary distributions of those T's.
        n_pairs_list:  list of ints, number of transition pairs used per bin.
    """
    n_bins        = len(cp_bins) - 1
    T_list        = []
    pi_list       = []
    n_pairs_list  = []

    for b in range(n_bins):
        in_bin = (cp >= cp_bins[b]) & (cp < cp_bins[b + 1])
        runs   = [(s, e) for (s, e) in _contiguous_runs(in_bin) if e - s >= min_window_len]

        T = np.zeros((num_states, num_states))
        n_pairs = 0
        for s, e in runs:
            for t in range(s, e - future):
                T[states[t], states[t + future]] += 1
                n_pairs += 1

        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        T_norm = T / row_sums

        try:
            pi_cp = stationary_distribution(T_norm)
        except Exception:
            counts = T.sum(axis=0)
            pi_cp = counts / counts.sum() if counts.sum() > 0 else np.ones(num_states) / num_states

        T_list.append(T_norm)
        pi_list.append(pi_cp)
        n_pairs_list.append(n_pairs)

    return T_list, pi_list, n_pairs_list


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


# ============================================================
# State-7 split by cold-pool area tendency
# (test: is cold-pool maturity the slow variable the 7-state
#  partition is missing? Split State 7 into growing/decaying
#  cold-pool sub-states, recompute T, recompute autocorrelation.)
# ============================================================

def cold_pool_area_tendency(
    area: np.ndarray,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Per-timestep cold-pool area tendency dA/dt (same length as `area`).

    `area` is a per-timestep cold-pool area series (e.g.
    ``cp_lifecycle['area']`` for total area, or ``'area_of_largest'`` for the
    dominant cold pool). Optionally Gaussian-smoothed before differencing to
    suppress single-step tracking noise. Sign convention:
        > 0  cold pool(s) growing  -> "young / aims to last"
        < 0  cold pool(s) decaying -> "old / about to exit"
    """
    a = np.asarray(area, dtype=float)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        a = gaussian_filter1d(a, sigma=smooth_sigma, mode="nearest")
    return np.gradient(a)


def split_state_by_area_tendency(
    states: np.ndarray,
    area_tendency: np.ndarray,
    target_state: int = 6,
    new_label: int = 7,
) -> np.ndarray:
    """Split `target_state` into two sub-states by cold-pool area-tendency sign.

    Growing/flat (tendency >= 0) keeps `target_state` ("young, aims to last");
    decaying (tendency < 0) is relabelled `new_label` ("old, about to exit").
    All other states are untouched. `states` and `area_tendency` are aligned to
    a common length (the shorter of the two) before splitting.
    """
    n = min(len(states), len(area_tendency))
    s8 = np.asarray(states[:n]).copy()
    tend = np.asarray(area_tendency[:n])
    decaying = (s8 == target_state) & (tend < 0)
    s8[decaying] = new_label
    return s8


def markov_autocorrelation_group_indicator(
    Tmat: np.ndarray,
    pi: np.ndarray,
    group,
    max_lag: int,
) -> np.ndarray:
    """Theoretical autocorrelation of the *grouped* indicator Y(t)=1[X_t in G].

    For a stationary Markov chain (T, pi) and group G:
        C_G(tau) = ( sum_{i in G} pi_i sum_{j in G} (T^tau)_{ij} - P_G^2 )
                   / ( P_G - P_G^2 ),     P_G = sum_{i in G} pi_i.

    With G a single state this reduces to markov_autocorrelation_indicator.
    Used to compare the 8-state (split) chain's prediction for the *physical*
    State 7 (= union of the two sub-states) against the empirical curve.
    """
    g = np.asarray(list(group), dtype=int)
    P = float(pi[g].sum())
    if P <= 1e-12 or P >= 1.0 - 1e-12:
        return np.ones(max_lag + 1)
    C = np.zeros(max_lag + 1)
    Tk = np.eye(Tmat.shape[0])              # T^0
    for k in range(max_lag + 1):
        joint = float(pi[g] @ Tk[np.ix_(g, g)].sum(axis=1))
        C[k] = (joint - P * P) / (P - P * P)
        Tk = Tk @ Tmat                      # T^(k+1) for next iter
    return C


def state7_split_markovianity(
    states: np.ndarray,
    area: np.ndarray,
    target_state: int = 6,
    new_label: int = 7,
    future: int = 5,
    subsample: int = 5,
    max_lag: int = 50,
    smooth_sigma: float = 0.0,
) -> dict:
    """Test whether splitting State 7 by cold-pool area tendency restores Markovianity.

    Builds the 7-state and 8-state chains the same way the main analysis does
    (full-resolution `states`, lag `future`; autocorrelation on the
    `subsample`-decimated series). Returns the empirical State-7 autocorrelation
    (unchanged by the split), the 7-state Markov prediction (which undershoots —
    the Fig.~9 gap), and the 8-state grouped prediction, plus how much of the
    gap the split closes.

    Returns dict with keys:
        lags, emp, pred7, pred8,
        states8, T7, pi7, T8, pi8,
        frac_young, rmse7, rmse8, gap_closed_frac
    """
    states = np.asarray(states)
    tend = cold_pool_area_tendency(area, smooth_sigma=smooth_sigma)
    states8 = split_state_by_area_tendency(states, tend, target_state, new_label)
    n = len(states8)
    states_al = states[:n]

    K7 = int(states_al.max()) + 1
    T7 = compute_transition_matrix(states_al, K7, future=future)
    pi7 = stationary_distribution(T7)
    T8 = compute_transition_matrix(states8, new_label + 1, future=future)
    pi8 = stationary_distribution(T8)

    emp = indicator_autocorrelation_normalized(
        states_al[::subsample], max_lag=max_lag,
    )[target_state]
    pred7 = markov_autocorrelation_indicator(T7, pi7, target_state, max_lag)
    pred8 = markov_autocorrelation_group_indicator(
        T8, pi8, (target_state, new_label), max_lag,
    )

    s7_mask = states8 == target_state
    n_phys7 = int((states_al == target_state).sum())
    frac_young = float(s7_mask.sum()) / n_phys7 if n_phys7 else float("nan")

    rmse7 = float(np.sqrt(np.mean((emp - pred7) ** 2)))
    rmse8 = float(np.sqrt(np.mean((emp - pred8) ** 2)))
    gap_closed = (rmse7 - rmse8) / rmse7 if rmse7 > 0 else float("nan")

    return dict(
        lags=np.arange(max_lag + 1), emp=emp, pred7=pred7, pred8=pred8,
        states8=states8, T7=T7, pi7=pi7, T8=T8, pi8=pi8,
        frac_young=frac_young, rmse7=rmse7, rmse8=rmse8,
        gap_closed_frac=gap_closed,
    )


def indicator_autocorrelation_state6_cp(
    states: np.ndarray,
    cp: np.ndarray,
    cp_bins: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """CP-conditioned autocorrelation for state 6 (original chopped-series version).

    Builds the indicator (state==6 AND cp in bin) over the full series and computes
    its autocorrelation. Note: drops to zero whenever cp leaves the bin, which
    artificially shortens persistence. See indicator_autocorrelation_state6_cp_windowed
    for the reviewer-suggested alternative.
    """
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


def _contiguous_runs(mask: np.ndarray):
    """List of (start, end_exclusive) tuples of True runs in a 1D boolean array."""
    runs, in_run, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def indicator_autocorrelation_state6_cp_windowed(
    states: np.ndarray,
    cp: np.ndarray,
    cp_bins: np.ndarray,
    max_lag: int,
    min_window_len: int = 20,
) -> tuple:
    """Cold-pool conditioned state-6 autocorrelation, computed inside contiguous CP windows.

    For each CP bin, find contiguous time windows where cp[t] stays in the bin for at
    least `min_window_len` consecutive steps. Compute the state-6 indicator autocorrelation
    *within each window*, then average across windows weighted by the number of valid
    lag-k pairs (window_len - k). This addresses the reviewer's concern that the original
    chopped-series version artificially reduces autocorrelation when cp leaves the bin.

    Returns:
        C:        (n_bins, max_lag+1) — weighted-average autocorrelation per bin.
        n_windows: (n_bins,) — number of qualifying windows per bin.
    """
    n_bins = len(cp_bins) - 1
    C = np.zeros((n_bins, max_lag + 1))
    n_windows = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        in_bin = (cp >= cp_bins[b]) & (cp < cp_bins[b + 1])
        runs   = [(s, e) for (s, e) in _contiguous_runs(in_bin) if e - s >= min_window_len]
        n_windows[b] = len(runs)
        if not runs:
            continue

        # Pre-center each window's state-6 indicator
        windows = []
        for s, e in runs:
            I   = (states[s:e] == 6).astype(float)
            I  -= I.mean()
            var = I.var()
            if var > 0:
                windows.append((e - s, I, var))

        for k in range(max_lag + 1):
            num, denom = 0.0, 0.0
            for window_len, I, var in windows:
                if window_len <= k:
                    continue
                ac_k = np.mean(I[:-k or None] * I[k:]) / var
                w    = window_len - k
                num   += w * ac_k
                denom += w
            if denom > 0:
                C[b, k] = num / denom

    return C, n_windows


def bootstrap_profile_band(
    samples: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.90,
    seed: int = 0,
) -> tuple:
    """Bootstrap a (lower, upper) confidence band for a sample mean along axis 0.

    samples: (n_samples, ...) — observations to average.
    Returns (mean, lower, upper) each shaped like samples.mean(axis=0).
    """
    rng = np.random.default_rng(seed)
    n   = samples.shape[0]
    if n == 0:
        return None, None, None
    boot_means = np.empty((n_boot,) + samples.shape[1:], dtype=samples.dtype)
    for b in range(n_boot):
        idx           = rng.integers(0, n, size=n)
        boot_means[b] = samples[idx].mean(axis=0)
    alpha = (1.0 - ci) / 2.0
    lo    = np.quantile(boot_means, alpha,       axis=0)
    hi    = np.quantile(boot_means, 1.0 - alpha, axis=0)
    return samples.mean(axis=0), lo, hi


def smooth_gaussian(C: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing along last axis."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(C, sigma=sigma, mode="nearest")
