"""Manuscript figure generation for the K=5 variant.

Self-contained: imports the shared paper utilities from
[paper/workflow_paper.py](../../paper/workflow_paper.py) but overrides
NUM_STATES, MODEL_PATH, and FIGURES_DIR so all output lands in
paper_5clusters/manuscript/figures/. The original paper/manuscript/figures/
directory is never touched.

Style mirrors the K=7 paper notebook (paper/notebooks/figures_paper.ipynb):
joint plots with KDE marginals, bootstrap CI bands on profiles, discrete
cluster colorbar at bottom, panel labels a/b/c/d.

Run from a fresh shell:

    cd /Users/sophieabramian/Documents/troposwhere
    conda activate samenv
    KMP_DUPLICATE_LIB_OK=TRUE python paper_5clusters/notebooks/figures_paper_5c.py

or invoke specific figures via the WHICH list at the bottom.
"""

import os
import sys

# Force all paths to be absolute and explicit.
ROOT = "/Users/sophieabramian/Documents/troposwhere"
PAPER_K7 = os.path.join(ROOT, "paper")
PAPER_K5 = os.path.join(ROOT, "paper_5clusters")
FIGURES_OUT = os.path.join(PAPER_K5, "manuscript", "figures")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, PAPER_K7)  # shared helpers (workflow_paper.py)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d

# Bump legend/axis/tick fonts to roughly caption size (~11–13pt) so the
# figures read well at the size they print in the manuscript.
plt.rcParams.update({
    "axes.labelsize":      13,
    "axes.titlesize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "legend.title_fontsize": 11,
    "figure.titlesize":    14,
})

import workflow_paper as wfp


# ---------------------------------------------------------------------------
# K=5 reconfiguration of shared module
# ---------------------------------------------------------------------------

NUM_STATES = 5
MODEL_PATH_K5 = os.path.join(
    ROOT, "data", "runs", "exp_20260223_1140_5c129e", "best_model.pt"
)

# 5-colour palette (same as figures_5clusters.ipynb).
PALETTE = ["#1e78b3", "#e02222", "#bcbcdd", "#ffe066", "#5ec962"]
STATE_COLORS = np.array(PALETTE)
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = Normalize(vmin=0, vmax=NUM_STATES - 1)

wfp.NUM_STATES = NUM_STATES
wfp.MODEL_PATH = MODEL_PATH_K5
wfp.state_colors = STATE_COLORS
wfp.cluster_cmap = STATE_CMAP
wfp.norm_state = STATE_NORM
wfp.FIGURES_DIR = FIGURES_OUT
wfp.FIGURES_SI_DIR = FIGURES_OUT

assert FIGURES_OUT.startswith(PAPER_K5), "Figure output must stay inside paper_5clusters/"
os.makedirs(FIGURES_OUT, exist_ok=True)

# Shared style constants (mirror K=7 notebook).
INTERVALS = [(10000, 10300), (5000, 5300)]  # same time-index trajectories as K=7
TRAJ_COLORS = ["k", "k"]
TRAJ_STYLES = ["--", ":"]
TRAJ_SMOOTH_WIN = 7


# ---------------------------------------------------------------------------
# Shared setup — cached lazily
# ---------------------------------------------------------------------------

_cache = {}


def setup():
    if "states" in _cache:
        return _cache

    print("[setup] loading dataloader + K=5 model...")
    loader, input_dim = wfp.load_dataloader()
    model = wfp.load_model(input_dim=input_dim)
    out = wfp.extract_latents(model, loader)
    embeddings = out["embeddings"]
    states = out["states"]
    emb_pca, pca = wfp.run_pca(embeddings)
    _, state_keep_indices = wfp.keep_closest_latent_samples(embeddings, states)

    print("[setup] loading diagnostics...")
    mass_flux = np.load(wfp.PATHS["data_raw"])[:-1]
    z_array = wfp.load_z_array()
    s_array = np.linspace(320, 350, mass_flux.shape[-1])
    pw = np.load(wfp.PATHS["pw"])[: len(states)]
    prec = np.load(wfp.PATHS["prec"])[: len(states)]

    n = min(len(mass_flux), len(states))
    mass_flux = mass_flux[:n]
    state_keep_indices_mf = {
        s: idxs[idxs < n]
        for s, idxs in state_keep_indices.items()
        if (idxs < n).any()
    }

    # Normalisation for the diagnostic colorings (mirrors K=7 notebook).
    pw_norm = Normalize(vmin=1, vmax=17)
    prec_norm = Normalize(vmin=0, vmax=6)

    # Transition matrix + stationary distribution.
    Tmat = wfp.compute_transition_matrix(states, NUM_STATES, future=5)
    pi = wfp.stationary_distribution(Tmat)

    print(f"[setup] N={len(states)}  mass_flux shape={mass_flux.shape}")
    print(f"[setup] cluster counts: "
          f"{dict(zip(*np.unique(states, return_counts=True)))}")
    print(f"[setup] pi = {[f'{p:.3f}' for p in pi]}")

    _cache.update(dict(
        loader=loader, input_dim=input_dim, model=model,
        embeddings=embeddings, states=states,
        emb_pca=emb_pca, pca=pca,
        state_keep_indices=state_keep_indices,
        state_keep_indices_mf=state_keep_indices_mf,
        mass_flux=mass_flux, z_array=z_array, s_array=s_array,
        pw=pw, prec=prec,
        pw_norm=pw_norm, prec_norm=prec_norm,
        Tmat=Tmat, pi=pi,
    ))
    return _cache


# ---------------------------------------------------------------------------
# Figure 3 — clusters in latent space (joint plot, KDE marginals, trajectories)
# ---------------------------------------------------------------------------

def generate_fig3_clusters():
    d = setup()
    states = d["states"]
    emb_pca = d["emb_pca"]
    evr = d["pca"].explained_variance_ratio_

    fig = plt.figure(figsize=(9, 8))
    ax_main = plt.axes([0.12, 0.12, 0.65, 0.65])
    ax_top = plt.axes([0.12, 0.80, 0.65, 0.15], sharex=ax_main)
    ax_right = plt.axes([0.80, 0.12, 0.15, 0.65], sharey=ax_main)

    unique, counts = np.unique(states, return_counts=True)
    count_dict = dict(zip(unique.astype(int), counts.astype(int)))

    for s in range(NUM_STATES):
        pts = emb_pca[states == s]
        ax_main.scatter(
            pts[:, 0], pts[:, 1],
            s=28, alpha=0.5, linewidth=0.25,
            facecolors=STATE_COLORS[s],
            label=f"State {wfp.state_label(s)} (n={count_dict.get(s, 0)})",
        )
        hist, bins = np.histogram(pts[:, 0], bins=40, density=True)
        cx = (bins[:-1] + bins[1:]) / 2
        ax_top.fill_between(cx, gaussian_filter1d(hist, 1), alpha=0.25, color=STATE_COLORS[s])
        ax_top.plot(cx, gaussian_filter1d(hist, 1), color=STATE_COLORS[s], lw=2)
        hist2, bins2 = np.histogram(pts[:, 1], bins=40, density=True)
        cy = (bins2[:-1] + bins2[1:]) / 2
        ax_right.fill_betweenx(cy, gaussian_filter1d(hist2, 1), alpha=0.25, color=STATE_COLORS[s])
        ax_right.plot(gaussian_filter1d(hist2, 1), cy, color=STATE_COLORS[s], lw=2)

    # Centroid labels with repulsion (same as K=7).
    barycenters = {s: emb_pca[states == s].mean(axis=0)
                   for s in range(NUM_STATES) if (states == s).any()}
    shifted = np.array([barycenters[s] for s in range(NUM_STATES)])
    for _ in range(200):
        for i in range(NUM_STATES):
            for j in range(i + 1, NUM_STATES):
                diff = shifted[i] - shifted[j]
                dist = np.linalg.norm(diff)
                if dist < 0.12:
                    if dist < 1e-6:
                        diff = np.random.randn(2)
                        dist = np.linalg.norm(diff)
                    push = 0.003 * diff / dist
                    shifted[i] += push
                    shifted[j] -= push
    for s in range(NUM_STATES):
        ax_main.text(
            *shifted[s], wfp.state_label(s),
            fontsize=13, fontweight="bold", color=STATE_COLORS[s],
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
        )

    # Example trajectories.
    for (t0, t1), ls in zip(INTERVALS, TRAJ_STYLES):
        traj = wfp.smooth_traj(emb_pca[t0:t1], win=TRAJ_SMOOTH_WIN)
        ax_main.plot(traj[:, 0], traj[:, 1], color="black",
                     lw=2.6, linestyle=ls, alpha=0.9, zorder=30)

    ax_main.set_xlabel(f"PCA 1  (intensity, {100*evr[0]:.0f}%)")
    ax_main.set_ylabel(f"PCA 2  (organization, {100*evr[1]:.0f}%)")
    for a in (ax_main, ax_top, ax_right):
        a.grid(True, linestyle="--", alpha=0.35)
    ax_top.set_ylabel("Density")
    ax_right.set_xlabel("Density")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)

    sm_cb = ScalarMappable(cmap=STATE_CMAP, norm=STATE_NORM)
    sm_cb.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.03, 0.65, 0.01])
    cbar = fig.colorbar(sm_cb, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("State Cluster")
    cbar.set_ticks(np.arange(NUM_STATES))
    cbar.set_ticklabels([wfp.state_label(i) for i in range(NUM_STATES)])

    traj_handles = [
        Line2D([0], [0], color="black", lw=2.6, linestyle="--", label="Strongly organized"),
        Line2D([0], [0], color="black", lw=2.6, linestyle=":", label="Weakly organized"),
    ]
    ax_main.legend(handles=traj_handles, loc="lower left",
                   frameon=True, framealpha=0.9, fontsize=11)

    out_path = os.path.join(FIGURES_OUT, "fig3_clusters.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig3] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — latent axes vs diagnostics (4 panels: a/b/c/d)
# ---------------------------------------------------------------------------

def generate_fig4_latent_axis():
    d = setup()
    states = d["states"]
    emb_pca = d["emb_pca"]
    pw = d["pw"]
    prec = d["prec"]
    pw_norm = d["pw_norm"]
    prec_norm = d["prec_norm"]
    evr = d["pca"].explained_variance_ratio_
    xlbl = f"PCA 1  (intensity, {100*evr[0]:.0f}%)"
    ylbl = f"PCA 2  (organization, {100*evr[1]:.0f}%)"

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 11, height_ratios=[1, 0.02, 1, 0.02],
                           hspace=0.35, wspace=0.25)

    ax_tl = fig.add_subplot(gs[0, :5])
    ax_tr = fig.add_subplot(gs[0, 6:])
    ax_bl = fig.add_subplot(gs[2, :5])
    ax_br = fig.add_subplot(gs[2, 6:])
    cax_tl = fig.add_subplot(gs[1, :5])
    cax_tr = fig.add_subplot(gs[1, 6:])
    cax_bottom = fig.add_subplot(gs[3, 2:9])

    sc_tl = ax_tl.scatter(emb_pca[:, 0], emb_pca[:, 1], c=prec,
                          cmap="Blues", norm=prec_norm, s=28, alpha=0.6)
    sc_tr = ax_tr.scatter(emb_pca[:, 0], emb_pca[:, 1], c=pw,
                          cmap="rainbow", norm=pw_norm, s=28, alpha=0.6)
    sc_bl = ax_bl.scatter(emb_pca[:, 0], prec, c=states,
                          cmap=STATE_CMAP, norm=STATE_NORM, s=26, alpha=0.6)
    ax_br.scatter(emb_pca[:, 1], pw, c=states,
                  cmap=STATE_CMAP, norm=STATE_NORM, s=26, alpha=0.6)

    ax_tl.set(xlabel=xlbl, ylabel=ylbl)
    ax_tr.set(xlabel=xlbl, ylabel=ylbl)
    ax_bl.set(xlabel=xlbl, ylabel="Mean precipitation")
    ax_br.set(xlabel=ylbl, ylabel="PW index")
    for ax in (ax_tl, ax_tr, ax_bl, ax_br):
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax, lbl in zip((ax_tl, ax_tr, ax_bl, ax_br), ("a", "b", "c", "d")):
        wfp.panel_label(ax, lbl)

    fig.colorbar(sc_tl, cax=cax_tl, orientation="horizontal").set_label("Precipitation")
    fig.colorbar(sc_tr, cax=cax_tr, orientation="horizontal").set_label("PW index")
    cb_bottom = fig.colorbar(sc_bl, cax=cax_bottom, orientation="horizontal", aspect=110)
    cb_bottom.set_label("Cluster state")
    cb_bottom.set_ticks(np.arange(NUM_STATES))
    cb_bottom.set_ticklabels([wfp.state_label(i) for i in range(NUM_STATES)])

    out_path = os.path.join(FIGURES_OUT, "fig4_latent_axis.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig4] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Combined 3-panel PCA figure (replaces fig3 + fig4 in the PRL main text):
#   (a) states   (b) precipitation (intensity)   (c) PW index (organization)
# each panel with a horizontal colorbar below.  No KDE marginals, no trajectories.
# ---------------------------------------------------------------------------

def generate_fig_pca_panels():
    d = setup()
    states = d["states"]
    emb = d["emb_pca"]
    pw = d["pw"]
    prec = d["prec"]
    pw_norm = d["pw_norm"]
    prec_norm = d["prec_norm"]
    evr = d["pca"].explained_variance_ratio_
    xlbl = f"PCA 1  (intensity, {100*evr[0]:.0f}%)"
    ylbl = f"PCA 2  (organization, {100*evr[1]:.0f}%)"

    fig = plt.figure(figsize=(13.5, 5.2))
    # row 1 is an empty spacer so the (thin) colorbars sit well below the panels
    gs = gridspec.GridSpec(
        3, 3, height_ratios=[1, 0.12, 0.028], hspace=0.05, wspace=0.18,
        left=0.055, right=0.985, top=0.94, bottom=0.12,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caxes = [fig.add_subplot(gs[2, i]) for i in range(3)]

    # (a) states / clusters
    sc_a = axes[0].scatter(emb[:, 0], emb[:, 1], c=states,
                           cmap=STATE_CMAP, norm=STATE_NORM,
                           s=20, alpha=0.55, linewidth=0)
    for s in range(NUM_STATES):
        m = states == s
        if m.any():
            axes[0].text(emb[m, 0].mean(), emb[m, 1].mean(), wfp.state_label(s),
                         fontsize=12, fontweight="bold", color=STATE_COLORS[s],
                         ha="center", va="center",
                         path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    # example trajectories (strongly / weakly organized), panel (a) only
    for (t0, t1), ls in zip(INTERVALS, TRAJ_STYLES):
        traj = wfp.smooth_traj(emb[t0:t1], win=TRAJ_SMOOTH_WIN)
        axes[0].plot(traj[:, 0], traj[:, 1], color="black",
                     lw=2.2, linestyle=ls, alpha=0.9, zorder=30)
    axes[0].legend(handles=[
        Line2D([0], [0], color="black", lw=2.2, linestyle="--", label="Strongly organized"),
        Line2D([0], [0], color="black", lw=2.2, linestyle=":", label="Weakly organized"),
    ], loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    cb_a = fig.colorbar(sc_a, cax=caxes[0], orientation="horizontal")
    cb_a.set_label("State")
    cb_a.set_ticks(np.arange(NUM_STATES))
    cb_a.set_ticklabels([wfp.state_label(i) for i in range(NUM_STATES)])

    # (b) precipitation -> intensity
    sc_b = axes[1].scatter(emb[:, 0], emb[:, 1], c=prec, cmap="Blues",
                           norm=prec_norm, s=20, alpha=0.6, linewidth=0)
    fig.colorbar(sc_b, cax=caxes[1], orientation="horizontal").set_label(
        "Precipitation (intensity)")

    # (c) PW index -> organization
    sc_c = axes[2].scatter(emb[:, 0], emb[:, 1], c=pw, cmap="rainbow",
                           norm=pw_norm, s=20, alpha=0.6, linewidth=0)
    fig.colorbar(sc_c, cax=caxes[2], orientation="horizontal").set_label(
        "PW index (organization)")

    for ax in axes:
        ax.set_xlabel(xlbl)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(ylbl)
    for ax, lbl in zip(axes, ("a", "b", "c")):
        wfp.panel_label(ax, lbl)

    out_path = os.path.join(FIGURES_OUT, "fig_pca_panels.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig_pca_panels] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — vertical mass-flux profiles (positive / negative / net)
# ---------------------------------------------------------------------------

def generate_fig5_vertical_profiles():
    d = setup()
    mass_flux = d["mass_flux"]
    z_array = d["z_array"]
    state_keep_indices_mf = d["state_keep_indices_mf"]

    win = 5
    mf_pos = np.where(mass_flux > 0, mass_flux, 0.0)
    mf_neg = np.where(mass_flux < 0, mass_flux, 0.0)
    mf_list = [
        (mf_pos, "Positive mass flux anomaly"),
        (mf_neg, "Negative mass flux anomaly"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    for ax, (mf, title), lbl in zip(axes, mf_list, ("a", "b")):
        mf_z = np.sum(mf, axis=2)
        mean_z = mf_z.mean(axis=0)
        ax.axvline(0.0, color="k", lw=1.2, alpha=0.6)
        for s in range(NUM_STATES):
            idxs = state_keep_indices_mf.get(s)
            if idxs is None or len(idxs) == 0:
                continue
            mean_p, lo_p, hi_p = wfp.bootstrap_profile_band(
                mf_z[idxs] - mean_z[None, :], n_boot=400, ci=0.90, seed=s,
            )
            mean_s = wfp.smooth_1d(mean_p, win=win)
            lo_s = wfp.smooth_1d(lo_p, win=win)
            hi_s = wfp.smooth_1d(hi_p, win=win)
            ax.fill_betweenx(z_array, lo_s, hi_s,
                             color=STATE_COLORS[s], alpha=0.18, linewidth=0)
            ax.plot(mean_s, z_array, color=STATE_COLORS[s], lw=2)
        ax.set_title(title)
        ax.set_xlabel("Mass flux anomaly")
        ax.grid(True, linestyle="--", alpha=0.3)
        wfp.panel_label(ax, lbl)
    axes[0].set_ylabel("Height (m)")

    legend_handles = [
        Line2D([0], [0], color=STATE_COLORS[s], lw=3, label=f"State {wfp.state_label(s)}")
        for s in range(NUM_STATES)
    ]
    leg = fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=NUM_STATES, frameon=True, fancybox=True, framealpha=0.9,
        title="State / Cluster (shaded: 90% bootstrap CI)",
    )
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(1.2)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    out_path = os.path.join(FIGURES_OUT, "fig5_vertical_profiles.pdf")
    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"[fig5] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 6 — mass-flux profiles along the MSE / isentropic axis
# ---------------------------------------------------------------------------

def generate_fig6_mse_profiles():
    d = setup()
    mass_flux = d["mass_flux"]
    s_array = d["s_array"]
    state_keep_indices_mf = d["state_keep_indices_mf"]

    win = 5
    mf_pos = np.clip(mass_flux, 0.0, None)
    mf_neg = np.clip(mass_flux, None, 0.0)
    mf_fields = [
        (mf_pos, "Positive mass flux anomaly"),
        (mf_neg, "Negative mass flux anomaly"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, (mf, title), lbl in zip(axes, mf_fields, ("a", "b")):
        mf_sum = mf.sum(axis=1)
        mean_profile = mf_sum.mean(axis=0)
        ax.axhline(0.0, color="k", lw=1.2, alpha=0.6)
        for s in range(NUM_STATES):
            idxs = state_keep_indices_mf.get(s)
            if idxs is None or len(idxs) == 0:
                continue
            mean_p, lo_p, hi_p = wfp.bootstrap_profile_band(
                mf_sum[idxs] - mean_profile[None, :], n_boot=400, ci=0.90, seed=s,
            )
            mean_s = wfp.smooth_1d(mean_p, win=win)
            lo_s = wfp.smooth_1d(lo_p, win=win)
            hi_s = wfp.smooth_1d(hi_p, win=win)
            ax.fill_between(s_array, lo_s, hi_s,
                            color=STATE_COLORS[s], alpha=0.18, linewidth=0)
            ax.plot(s_array, mean_s, color=STATE_COLORS[s], lw=2)
        ax.set_title(title)
        ax.set_ylabel("Mass flux anomaly")
        ax.grid(True, linestyle="--", alpha=0.3)
        wfp.panel_label(ax, lbl)
    axes[-1].set_xlabel("MSE")

    legend_handles = [
        Line2D([0], [0], color=STATE_COLORS[s], lw=3, label=f"State {wfp.state_label(s)}")
        for s in range(NUM_STATES)
    ]
    leg = fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=NUM_STATES, frameon=True, fancybox=True, framealpha=0.9,
        title="State / Cluster (shaded: 90% bootstrap CI)",
        bbox_to_anchor=(0.5, -0.06), borderpad=0.8, columnspacing=1.8, handlelength=2.8,
    )
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(1.2)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(FIGURES_OUT, "fig6_mse_profiles.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 7 — Markov transitions in latent space + transition matrix
# ---------------------------------------------------------------------------

def generate_fig7_markov_matrix():
    d = setup()
    states = d["states"]
    emb_pca = d["emb_pca"]
    Tmat = d["Tmat"]
    pi = d["pi"]
    evr = d["pca"].explained_variance_ratio_

    FIGSIZE = (14, 7)
    BUBBLE_SCALE = 5000
    TMIN, TMAX = 0.01, 0.8
    EDGE_PROB_THRESHOLD = 0.01  # draw transitions with P >= 1% (shows the weak 1<->2 link)

    bubble_size = BUBBLE_SCALE * pi

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(100, 100, hspace=0.30, wspace=0.25)
    ax_pca = fig.add_subplot(gs[10:85, 6:51])
    ax_mat = fig.add_subplot(gs[10:85, 60:94])
    cax_pca = fig.add_subplot(gs[98:100, 10:47])
    cax_mat = fig.add_subplot(gs[98:100, 64:90])

    sc = ax_pca.scatter(emb_pca[:, 0], emb_pca[:, 1], c=states,
                        cmap=STATE_CMAP, norm=STATE_NORM,
                        s=28, alpha=0.55, linewidth=0.25)
    ax_pca.set(
        xlabel=f"PCA 1  (intensity, {100*evr[0]:.0f}%)",
        ylabel=f"PCA 2  (organization, {100*evr[1]:.0f}%)",
    )
    ax_pca.grid(True, linestyle="--", alpha=0.35)
    wfp.panel_label(ax_pca, "a")

    bary = np.array([
        emb_pca[states == s].mean(axis=0) if (states == s).any() else np.zeros(2)
        for s in range(NUM_STATES)
    ])

    for ((t0, t1), col, style) in zip(INTERVALS, TRAJ_COLORS, TRAJ_STYLES):
        traj = wfp.smooth_traj(emb_pca[t0:t1], win=TRAJ_SMOOTH_WIN)
        if len(traj) == 0:
            continue
        ax_pca.plot(traj[:, 0], traj[:, 1], color=col, linestyle=style,
                    lw=2.4, zorder=20,
                    path_effects=[pe.withStroke(linewidth=4.5, foreground="white")])

    cbar_pca = fig.colorbar(sc, cax=cax_pca, orientation="horizontal", aspect=60, shrink=0.8)
    cbar_pca.set_label("Cluster state")
    cbar_pca.set_ticks(np.arange(NUM_STATES))
    cbar_pca.set_ticklabels([wfp.state_label(i) for i in range(NUM_STATES)])

    Tplot = np.ma.masked_less(Tmat, TMIN)
    norm_mat = mcolors.LogNorm(vmin=TMIN, vmax=TMAX)
    im = ax_mat.imshow(Tplot, cmap="Blues", norm=norm_mat, origin="upper")
    ax_mat.set(xlabel="Next state j", ylabel="Current state i")
    ax_mat.set_xticks(np.arange(NUM_STATES))
    ax_mat.set_yticks(np.arange(NUM_STATES))
    ax_mat.set_xticklabels([wfp.state_label(i) for i in range(NUM_STATES)])
    ax_mat.set_yticklabels([wfp.state_label(i) for i in range(NUM_STATES)])
    wfp.panel_label(ax_mat, "b")
    for i in range(Tmat.shape[0]):
        for j in range(Tmat.shape[1]):
            ax_mat.text(j, i, f"{Tmat[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if Tmat[i, j] > 0.5 * Tmat.max() else "black")
    fig.colorbar(im, cax=cax_mat, orientation="horizontal", aspect=60, shrink=0.8).set_label("P(i → j)")

    # Chain edges with probability labels. Each curved Bezier connects
    # barycenter i -> j; line width and arrowhead encode P(i->j); a small
    # numeric badge sits next to the curve so it does not interrupt the
    # arrow flow (critical for reading the 4<->5 oscillation).
    LW_MIN, LW_MAX = 1.5, 8.0
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            if i == j:
                continue
            p_ij = float(Tmat[i, j])
            if p_ij < EDGE_PROB_THRESHOLD:
                continue
            p0 = bary[i]
            p1 = bary[j]
            d_vec = p1 - p0
            dist = np.linalg.norm(d_vec)
            if dist < 1e-9:
                continue
            perp = np.array([-d_vec[1], d_vec[0]]) / dist
            rad = 0.22  # slightly more bowed so reciprocal pairs separate
            ctrl = (p0 + p1) / 2 + rad * dist * perp
            t_samples = np.linspace(0.0, 1.0, 80)[:, None]
            curve = ((1 - t_samples) ** 2 * p0
                     + 2 * (1 - t_samples) * t_samples * ctrl
                     + t_samples ** 2 * p1)
            # Trim curve so it lives in the space between the two bubbles.
            # Bubble radius scales with pi; pick a conservative shrink distance
            # that clears the largest bubble (C3).
            trim_source = 0.20 + 1.2 * np.sqrt(pi[i])
            trim_target = 0.20 + 1.2 * np.sqrt(pi[j])
            d_from_p0 = np.linalg.norm(curve - p0, axis=1)
            d_from_p1 = np.linalg.norm(curve - p1, axis=1)
            keep = (d_from_p0 > trim_source) & (d_from_p1 > trim_target)
            if keep.sum() < 4:  # too short to be useful
                continue
            curve = curve[keep]
            # Map p_ij in [EDGE_PROB_THRESHOLD, 0.20] -> [LW_MIN, LW_MAX].
            t = np.clip((p_ij - EDGE_PROB_THRESHOLD) / 0.18, 0, 1)
            lw = LW_MIN + t * (LW_MAX - LW_MIN)
            ax_pca.plot(curve[:, 0], curve[:, 1], color="black", lw=lw,
                        alpha=0.9, zorder=60, solid_capstyle="round")
            # Filled arrowhead near the end of the curve. The head is sized
            # generously so the direction is unambiguous at the printed size;
            # taking a longer base (curve[-8]) gives it enough length to
            # render as a clear triangle rather than a sliver.
            head_anchor = max(8, int(0.12 * len(curve)))
            ax_pca.annotate(
                "", xy=tuple(curve[-1]), xytext=tuple(curve[-head_anchor]),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.55,head_width=0.32",
                    color="black", fc="black", lw=0,
                    mutation_scale=22 + 18 * t,
                ),
                zorder=61,
            )
            # Place label closer to the source (t=0.30 along the curve) and
            # pushed off the curve perpendicularly. Anchoring near the source
            # rather than the midpoint separates labels belonging to edges
            # that share a destination — critical near C3 where 1->3, 2->3,
            # and 5->3 would otherwise stack.
            anchor_t = 0.30
            anchor_idx = int(anchor_t * (len(curve) - 1))
            anchor = curve[anchor_idx]
            label_pos = anchor + 0.45 * perp
            ax_pca.text(
                label_pos[0], label_pos[1], f"{p_ij:.2f}",
                fontsize=10, color="black", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec="0.6", lw=0.6, alpha=0.92),
                zorder=62,
            )

    for s in range(NUM_STATES):
        ax_pca.scatter(*bary[s], s=bubble_size[s], color=STATE_COLORS[s],
                       edgecolors="black", linewidths=1.2, alpha=0.85, zorder=999)
        ax_pca.text(*bary[s], wfp.state_label(s),
                    fontsize=13, fontweight="bold", color=STATE_COLORS[s],
                    ha="center", va="center", zorder=1000,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    out_path = os.path.join(FIGURES_OUT, "fig7_markov_matrix.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[fig7] saved -> {out_path}")


# ---------------------------------------------------------------------------
# fig7 blank — PCA scatter + cluster bubbles only (no arrows, no labels).
# Designed as a starter for hand-annotation (e.g. in PowerPoint / Illustrator).
# Saves to fig7_markov_matrix_blank.pdf so the manuscript figure is untouched.

def generate_fig7_blank():
    d = setup()
    states = d["states"]
    emb_pca = d["emb_pca"]
    Tmat = d["Tmat"]
    pi = d["pi"]
    evr = d["pca"].explained_variance_ratio_

    FIGSIZE = (14, 7)
    BUBBLE_SCALE = 5000
    TMIN, TMAX = 0.01, 0.8
    bubble_size = BUBBLE_SCALE * pi

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = gridspec.GridSpec(100, 100, hspace=0.30, wspace=0.25)
    ax_pca = fig.add_subplot(gs[10:85, 6:51])
    ax_mat = fig.add_subplot(gs[10:85, 60:94])
    cax_pca = fig.add_subplot(gs[98:100, 10:47])
    cax_mat = fig.add_subplot(gs[98:100, 64:90])

    # Panel A: PCA scatter + cluster bubbles (no chain edges, no labels).
    sc = ax_pca.scatter(emb_pca[:, 0], emb_pca[:, 1], c=states,
                        cmap=STATE_CMAP, norm=STATE_NORM,
                        s=28, alpha=0.55, linewidth=0.25)
    ax_pca.set(
        xlabel=f"PCA 1  (intensity, {100*evr[0]:.0f}%)",
        ylabel=f"PCA 2  (organization, {100*evr[1]:.0f}%)",
    )
    ax_pca.grid(True, linestyle="--", alpha=0.35)

    bary = np.array([
        emb_pca[states == s].mean(axis=0) if (states == s).any() else np.zeros(2)
        for s in range(NUM_STATES)
    ])
    for s in range(NUM_STATES):
        ax_pca.scatter(*bary[s], s=bubble_size[s], color=STATE_COLORS[s],
                       edgecolors="black", linewidths=1.2, alpha=0.85, zorder=999)
        ax_pca.text(*bary[s], wfp.state_label(s),
                    fontsize=13, fontweight="bold", color=STATE_COLORS[s],
                    ha="center", va="center", zorder=1000,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    cbar_pca = fig.colorbar(sc, cax=cax_pca, orientation="horizontal", aspect=60, shrink=0.8)
    cbar_pca.set_label("Cluster state")
    cbar_pca.set_ticks(np.arange(NUM_STATES))
    cbar_pca.set_ticklabels([wfp.state_label(i) for i in range(NUM_STATES)])

    # Panel B: log-scale transition matrix with cell-value annotations.
    Tplot = np.ma.masked_less(Tmat, TMIN)
    norm_mat = mcolors.LogNorm(vmin=TMIN, vmax=TMAX)
    im = ax_mat.imshow(Tplot, cmap="Blues", norm=norm_mat, origin="upper")
    ax_mat.set(xlabel="Next state j", ylabel="Current state i")
    ax_mat.set_xticks(np.arange(NUM_STATES))
    ax_mat.set_yticks(np.arange(NUM_STATES))
    ax_mat.set_xticklabels([wfp.state_label(i) for i in range(NUM_STATES)])
    ax_mat.set_yticklabels([wfp.state_label(i) for i in range(NUM_STATES)])
    for i in range(Tmat.shape[0]):
        for j in range(Tmat.shape[1]):
            ax_mat.text(j, i, f"{Tmat[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if Tmat[i, j] > 0.5 * Tmat.max() else "black")
    fig.colorbar(im, cax=cax_mat, orientation="horizontal", aspect=60, shrink=0.8).set_label("P(i → j)")

    out_path = os.path.join(FIGURES_OUT, "fig7_markov_matrix_blank.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[fig7 blank] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 9bis — Markovianity + cold-pool conditioning (two panels)
# ---------------------------------------------------------------------------
# (a) per-state empirical autocorrelation vs Markov prediction
# (b) State-1 zoom: empirical, top-25%-CP-at-t conditional, Markov reference
#
# K=5: target = State 1 (display index 1 = data index 0).
# This is the organized analogue of K=7 State 7 (overlap 0.96 per
# paper/reports/cluster_comparison/numbers.json) and the most non-Markovian
# at K=5 (RMSE 0.288 vs Markov reference).

TARGET_STATE_IDX_K5 = 0  # data index 0 = display State 1
SUBSAMPLE = 5             # match K=7 notebook
DT_LAG_MIN = 10           # chain step (minutes)
MAX_LAG_STEPS = 50        # 50 steps × 5 (subsample) × 10 min = 2500 min total window


def _ensure_autocorrelation_cache(d):
    """Compute C_obs, C_markov_per_state, s_sub, cp_sub once and cache."""
    if "C_obs" in d:
        return
    states = d["states"]
    Tmat = d["Tmat"]
    pi = d["pi"]

    print("[fig9bis] computing per-state autocorrelations...")
    s_sub = states[::SUBSAMPLE]
    cp = np.load(wfp.PATHS["cpi"])[: len(states)]
    cp_sub = cp[::SUBSAMPLE]

    C_obs = wfp.indicator_autocorrelation_normalized(s_sub, max_lag=MAX_LAG_STEPS)
    C_markov_per_state = np.array([
        wfp.markov_autocorrelation_indicator(Tmat, pi, state_idx=s, max_lag=MAX_LAG_STEPS)
        for s in range(NUM_STATES)
    ])

    d["C_obs"] = C_obs
    d["C_markov_per_state"] = C_markov_per_state
    d["s_sub"] = s_sub
    d["cp_sub"] = cp_sub
    d["cp"] = cp


def generate_fig9bis_markovianity_coldpools():
    d = setup()
    _ensure_autocorrelation_cache(d)

    C_obs = d["C_obs"]
    C_markov_per_state = d["C_markov_per_state"]
    s_sub = d["s_sub"]
    cp_sub = d["cp_sub"]
    pi = d["pi"]

    target = TARGET_STATE_IDX_K5  # 0 (display State 1)

    n_lag = C_obs.shape[1]
    # In K=7 fig9bis the lag axis was "n_lag_bis × DT_LAG_MIN" (without the
    # SUBSAMPLE factor). The K=7 notebook treats each subsampled step as
    # DT_LAG_MIN; we mirror that so units stay consistent across the two papers.
    lags_min = np.arange(n_lag) * DT_LAG_MIN

    # CP-conditioned survival for target state (top 25% CP at t).
    sv = wfp.survival_conditional_on_cp(
        s_sub, cp_sub, target_state=target, max_lag=n_lag - 1,
        low_q=0.25, high_q=0.75,
    )
    S_high, n_high = sv["high"]
    pi_target = pi[target]
    rho_high = (S_high - pi_target) / (1.0 - pi_target)  # back to AC scale

    target_color = STATE_COLORS[target]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # -- Left: per-state empirical AC vs Markov ---------------------------------
    for i in range(NUM_STATES):
        color = STATE_COLORS[i]
        axL.plot(
            lags_min, wfp.smooth_gaussian(C_obs[i], sigma=1.2),
            color=color, lw=2, label=f"State {wfp.state_label(i)}",
        )
        axL.plot(
            lags_min, C_markov_per_state[i],
            color=color, lw=1.2, linestyle="--", alpha=0.7,
        )
    axL.set_xlabel("Time lag (min)")
    axL.set_ylabel("Indicator autocorrelation")
    axL.set_ylim(-0.05, 1.1)
    axL.grid(True, alpha=0.3)
    axL.legend(
        ncol=NUM_STATES, fontsize=8,
        title="solid: empirical   dashed: Markov",
        title_fontsize=8,
    )
    axL.set_title("(a) per-state indicator autocorrelation")

    # -- Right: target state zoom with CP conditioning -------------------------
    axR.plot(
        lags_min, wfp.smooth_gaussian(C_obs[target], sigma=1.2),
        color=target_color, lw=2.6,
        label=f"State {wfp.state_label(target)} (empirical)",
    )
    axR.plot(
        lags_min, rho_high,
        color=target_color, lw=2.0, linestyle="--",
        label=fr"State {wfp.state_label(target)}, CP at $t$ in top quartile  (n={n_high})",
    )
    axR.plot(
        lags_min, C_markov_per_state[target],
        color="k", lw=1.4, linestyle=":",
        label=f"State {wfp.state_label(target)} (Markov prediction)",
    )
    axR.set_xlabel("Time lag (min)")
    axR.set_ylabel("Indicator autocorrelation")
    axR.set_ylim(-0.05, 1.1)
    axR.grid(True, alpha=0.3)
    axR.legend(fontsize=8, frameon=False)
    axR.set_title(
        f"(b) state {wfp.state_label(target)}  --  conditioning on CP at $t$"
    )

    plt.tight_layout()

    # Diagnostic prints (mirrors K=7 notebook).
    gap_emp_markov_200 = C_obs[target, 20] - C_markov_per_state[target][20]
    gap_cond_emp_200 = rho_high[20] - C_obs[target, 20]
    print(f"[fig9bis] target state = display {wfp.state_label(target)} (data idx {target})")
    print(f"[fig9bis] n_high (top 25% CP at t entries) = {n_high}")
    print(f"[fig9bis] pi_target = {pi_target:.4f}")
    print(f"[fig9bis] gap empirical - Markov  at 200 min = {gap_emp_markov_200:+.3f}")
    print(f"[fig9bis] gap conditional - empirical at 200 min = {gap_cond_emp_200:+.3f}")
    # Per-state RMSE between empirical AC and Markov prediction (numbers.json metric).
    for s in range(NUM_STATES):
        rmse = float(np.sqrt(np.mean((C_obs[s] - C_markov_per_state[s]) ** 2)))
        print(f"[fig9bis]   State {wfp.state_label(s)} RMSE(emp - markov) = {rmse:.4f}")

    out_path = os.path.join(FIGURES_OUT, "fig9bis_markovianity_coldpools.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig9bis] saved -> {out_path}")


# ===========================================================================
# SI figures
# ===========================================================================
# Mirror the K=7 figures_si.ipynb cells, retargeted at the K=5 model.
# fields_clusters.pdf is skipped — it needs raw 2D variable arrays from the
# LaCie drive (not accessible here).

import pickle
from matplotlib.ticker import LogLocator


def _setup_si():
    """Augment the main cache with SI-specific data (loss history, 3D PCA,
    physical grids, log-scaled data). Cached on first call."""
    d = setup()
    if "loss_train" in d:
        return d

    print("[si setup] loading K=5 training history + physical grids...")
    exp_dir_k5 = os.path.join(ROOT, "data", "runs", "exp_20260223_1140_5c129e")
    with open(os.path.join(exp_dir_k5, "loss_history.pkl"), "rb") as f:
        loss_dict = pickle.load(f)

    def _parse_logs(logs):
        keys = ["total", "recon", "kl_scaled", "hmm_scaled",
                "entropy_unscaled", "trans_acc", "dice_loss_scaled"]
        return {k: np.array([row[k] for row in logs]) for k in keys}

    epochs = np.arange(len(loss_dict["train"]))
    loss_train = _parse_logs(loss_dict["train"])
    loss_val = _parse_logs(loss_dict["val"])

    # Physical grid + log-scaled data (used for reconstruction / canonical
    # representative panels).
    S, Z = wfp.load_physical_grids()
    data_log = np.load(wfp.PATHS["data_log"])
    if data_log.ndim > 2:
        data_log = data_log.reshape(len(data_log), -1)

    # 3-component PCA for pca_3d.
    from sklearn.decomposition import PCA as _PCA
    pca3 = _PCA(n_components=3)
    emb_pca_3d = pca3.fit_transform(d["embeddings"])

    d.update(dict(
        loss_train=loss_train,
        loss_val=loss_val,
        epochs=epochs,
        S=S, Z=Z,
        data_log=data_log,
        emb_pca_3d=emb_pca_3d,
        exp_dir_k5=exp_dir_k5,
    ))
    return d


# ---- training_loss.pdf ----------------------------------------------------

def _plot_loss_curve(loss_series, epochs, out_path, title, ls):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(epochs, loss_series["total"], label="Total",
            lw=3.0, ls=ls, color="r", zorder=5)
    ax.plot(epochs, loss_series["recon"], label="Reconstruction",
            lw=2.2, ls=ls, color="#4C72B0")
    ax.plot(epochs, loss_series["kl_scaled"], label=r"KL ($\beta\,\mathrm{KL}$)",
            lw=2.2, ls=ls, color="#8FB1E3")
    ax.plot(epochs, loss_series["dice_loss_scaled"],
            label=r"Dice ($\lambda_{\mathrm{Dice}}$)",
            lw=2.2, ls=ls, color="#E1B12C")
    ax.plot(epochs, loss_series["hmm_scaled"],
            label=r"State transition ($\gamma_{\mathrm{MC}}$)",
            lw=2.2, ls=ls, color="#DD8452")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8)))
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.15)
    ax.legend(frameon=False, fontsize=10, loc="upper right", handlelength=2.8)
    ax.set_title(title, style="italic")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_training_loss():
    d = _setup_si()
    out = os.path.join(FIGURES_OUT, "training_loss.pdf")
    _plot_loss_curve(d["loss_train"], d["epochs"], out, "Training", ls="-")
    print(f"[training_loss] saved -> {out}")


def generate_validation_loss():
    d = _setup_si()
    out = os.path.join(FIGURES_OUT, "validation_loss.pdf")
    _plot_loss_curve(d["loss_val"], d["epochs"], out, "Validation", ls="--")
    print(f"[validation_loss] saved -> {out}")


# ---- reconstruction.pdf ---------------------------------------------------

def generate_reconstruction():
    import torch
    d = _setup_si()
    states = d["states"]
    embeddings = d["embeddings"]
    state_keep_indices = d["state_keep_indices"]
    data_log = d["data_log"]
    S, Z = d["S"], d["Z"]
    model = d["model"]
    device = next(model.parameters()).device

    vmin, vmax = -4, 4
    levels = np.sort(np.unique(np.concatenate([
        np.linspace(-12, -5, 3), np.linspace(-5, -1, 8),
        np.linspace(-1, 1, 20), np.linspace(1, 5, 8),
        np.linspace(5, 12, 3),
    ])))

    states_to_plot = list(range(NUM_STATES))  # all five at K=5
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(3 * len(states_to_plot), 7))
    gs = fig.add_gridspec(3, len(states_to_plot), height_ratios=[1, 1, 0.08], hspace=0.35)
    ax = np.empty((2, len(states_to_plot)), dtype=object)
    for j in range(len(states_to_plot)):
        ax[0, j] = fig.add_subplot(gs[0, j])
        ax[1, j] = fig.add_subplot(gs[1, j], sharex=ax[0, j], sharey=ax[0, j])
    cax = fig.add_subplot(gs[2, :])

    for j, s in enumerate(states_to_plot):
        if s not in state_keep_indices:
            continue
        idx = rng.choice(state_keep_indices[s])
        z_t = torch.tensor(embeddings[idx], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            recon_flat = model.decoder(z_t).cpu().numpy()[0]
        recon_img = wfp.create_image_from_flat_tensor_np(recon_flat)[0]
        true_img = wfp.create_image_from_flat_tensor_np(data_log[idx])[0]
        ax[0, j].contourf(S, Z, recon_img, levels=levels, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
        ax[1, j].contourf(S, Z, true_img, levels=levels, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
        ax[0, j].set_title(f"State {wfp.state_label(s)} | Recon")
        ax[1, j].set_title("Truth")
        for row in (0, 1):
            ax[row, j].grid(True, alpha=0.3)
        if j == 0:
            ax[0, j].set_ylabel("Z [km]")
            ax[1, j].set_ylabel("Z [km]")
        else:
            ax[0, j].set_yticks([])
            ax[1, j].set_yticks([])
        ax[1, j].set_xlabel("MSE bins")

    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                               cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="horizontal",
                 label=r"Mass Flux Anomaly [$kg\,m\,s^{-1}$]")

    out = os.path.join(FIGURES_OUT, "reconstruction.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[reconstruction] saved -> {out}")


# ---- canonical_representative.pdf -----------------------------------------

def generate_canonical_representative():
    import torch
    d = _setup_si()
    embeddings = d["embeddings"]
    state_keep_indices = d["state_keep_indices"]
    data_log = d["data_log"]
    S, Z = d["S"], d["Z"]
    model = d["model"]
    device = next(model.parameters()).device

    vmin, vmax = -4, 4
    levels = np.sort(np.unique(np.concatenate([
        np.linspace(-12, -5, 3), np.linspace(-5, -1, 8),
        np.linspace(-1, 1, 20), np.linspace(1, 5, 8),
        np.linspace(5, 12, 3),
    ])))

    states_to_plot = sorted(state_keep_indices.keys())

    fig, ax = plt.subplots(2, len(states_to_plot),
                           figsize=(2.2 * len(states_to_plot), 6),
                           sharex=True, sharey=True)
    for j, s in enumerate(states_to_plot):
        idxs = state_keep_indices[s]
        mean_orig_img = wfp.create_image_from_flat_tensor_np(data_log[idxs].mean(axis=0))[0]
        mean_latent_t = torch.tensor(embeddings[idxs].mean(axis=0),
                                     dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean_recon_img = wfp.create_image_from_flat_tensor_np(
                model.decoder(mean_latent_t).cpu().numpy()[0])[0]
        ax[0, j].contourf(S, Z, mean_orig_img, levels=levels, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
        ax[1, j].contourf(S, Z, mean_recon_img, levels=levels, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
        ax[0, j].set_title(f"State {wfp.state_label(s)}\n⟨x⟩")
        ax[1, j].set_title("Dec(⟨z⟩)")
        for row in (0, 1):
            ax[row, j].grid(True, alpha=0.3)
        if j == 0:
            ax[0, j].set_ylabel("Z [km]")
            ax[1, j].set_ylabel("Z [km]")
        ax[1, j].set_xlabel("MSE bins")

    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                               cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.12,
                 label=r"Mass Flux [$kg\cdot m\,s^{-1}$]", aspect=110, shrink=0.8)

    out = os.path.join(FIGURES_OUT, "canonical_representative.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[canonical_representative] saved -> {out}")


# ---- pca_3d.pdf -----------------------------------------------------------

def generate_pca_3d():
    d = _setup_si()
    states = d["states"]
    emb_pca_3d = d["emb_pca_3d"]

    # 3-component PCA's explained variance — refit briefly to read it back.
    from sklearn.decomposition import PCA as _PCA
    evr3 = _PCA(n_components=3).fit(d["embeddings"]).explained_variance_ratio_
    labels = [f"PCA {k+1}  ({100*evr3[k]:.0f}%)" for k in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(emb_pca_3d[:, i], emb_pca_3d[:, j], c=states,
                   cmap=STATE_CMAP, norm=STATE_NORM,
                   s=26, alpha=0.55, linewidth=0.25)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_OUT, "pca_3d.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[pca_3d] saved -> {out}")


# ---- fig8_trajectories.pdf ------------------------------------------------

def generate_fig8_trajectories():
    d = setup()
    states = d["states"]
    pw = d["pw"]
    prec = d["prec"]

    DT_RAW_MIN = 2  # raw output cadence (minutes)
    intervals = INTERVALS
    traj_labels = ["Trajectory A — strongly organized",
                   "Trajectory B — weakly organized"]
    panel_labels = ["a", "b"]

    PW_COLOR = "tab:red"
    PREC_COLOR = "tab:blue"
    pw_all = np.concatenate([pw[t0:t1] for (t0, t1) in intervals])
    prec_all = np.concatenate([prec[t0:t1] for (t0, t1) in intervals])
    pw_ylim = (np.floor(pw_all.min() * 10) / 10, np.ceil(pw_all.max() * 10) / 10)
    prec_ylim = (np.floor(prec_all.min() * 10) / 10, np.ceil(prec_all.max() * 10) / 10)

    fig = plt.figure(figsize=(12, 8))
    outer = gridspec.GridSpec(2, 1, hspace=0.45)

    for row, ((t0, t1), name, plbl) in enumerate(zip(intervals, traj_labels, panel_labels)):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row], height_ratios=[0.15, 1.0], hspace=0.05,
        )
        ax_strip = fig.add_subplot(inner[0])
        ax_ts = fig.add_subplot(inner[1], sharex=ax_strip)

        t_axis = np.arange(t0, t1) * DT_RAW_MIN / 60.0  # hours
        s_seg = states[t0:t1]

        for i, s in enumerate(s_seg):
            ax_strip.axvspan(
                t_axis[i] - DT_RAW_MIN / 120,
                t_axis[i] + DT_RAW_MIN / 120,
                color=STATE_COLORS[s], lw=0,
            )
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        ax_strip.set_ylabel("State", rotation=0, ha="right", va="center", fontsize=10)
        ax_strip.set_title(name, fontsize=12, loc="left")
        wfp.panel_label(ax_strip, plbl, x=-0.04, y=1.6)

        ax_ts.plot(t_axis, pw[t0:t1], color=PW_COLOR, lw=1.6, label="PW index")
        ax_ts.set_ylabel("PW index", color=PW_COLOR)
        ax_ts.tick_params(axis="y", labelcolor=PW_COLOR)
        ax_ts.set_ylim(pw_ylim)
        ax_ts.grid(True, linestyle="--", alpha=0.3)
        ax_ts.set_xlabel("Time (hours)")

        ax_ts_b = ax_ts.twinx()
        ax_ts_b.plot(t_axis, prec[t0:t1], color=PREC_COLOR, lw=1.6, label="Precipitation")
        ax_ts_b.set_ylabel("Precipitation", color=PREC_COLOR)
        ax_ts_b.tick_params(axis="y", labelcolor=PREC_COLOR)
        ax_ts_b.set_ylim(prec_ylim)

    legend_handles = [
        Line2D([0], [0], color=STATE_COLORS[s], lw=8, label=f"State {wfp.state_label(s)}")
        for s in range(NUM_STATES)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=NUM_STATES,
               frameon=True, fancybox=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(bottom=0.13)

    out = os.path.join(FIGURES_OUT, "fig8_trajectories.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig8_trajectories] saved -> {out}")


# ---- cluster_sensitivity.pdf ----------------------------------------------
# K=5 manuscript: comparator set is {3, 7, 10}. All checkpoints exist locally.

CLUSTER_SENSITIVITY_K = [3, 7, 10]
EXPLICIT_RUN_DIRS = {
    3:  os.path.join(ROOT, "data/runs/exp_20260223_1555_96a8dd"),
    7:  os.path.join(ROOT, "data/runs/exp_20260220_1809_7a8b88_paper"),
    10: os.path.join(ROOT, "data/runs/exp_20260225_1522_af2163"),
}


def generate_cluster_sensitivity():
    d = setup()
    loader = d["loader"]
    input_dim = d["input_dim"]

    # model_comparison helpers are in paper/model_comparison/
    sys.path.insert(0, os.path.join(PAPER_K7, "model_comparison"))
    from comparison_utils import process_experiment
    from sensitivity_test_plot_functions import plot_pca_clusters

    fig, axes = plt.subplots(1, len(CLUSTER_SENSITIVITY_K), figsize=(15, 5))
    for ax, k in zip(axes, CLUSTER_SENSITIVITY_K):
        exp_dir = EXPLICIT_RUN_DIRS[k]
        print(f"[cluster_sensitivity] processing K={k} from {os.path.basename(exp_dir)}")
        res = process_experiment(exp_dir, loader=loader, input_dim=input_dim)
        plot_pca_clusters(
            emb_pca=res["emb_pca"],
            states=res["states"],
            num_states=res["num_states"],
            pca=res["pca"],
            intervals=INTERVALS,
            smooth_win=TRAJ_SMOOTH_WIN,
            ax=ax,
        )
        ax.set_title(f"{k} clusters")
    plt.tight_layout()
    out = os.path.join(FIGURES_OUT, "cluster_sensitivity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[cluster_sensitivity] saved -> {out}")


# ---- fields_clusters.pdf --------------------------------------------------
# Needs raw 2D variable arrays on the LaCie drive
# (/Volumes/LaCie/000_POSTDOC_2025/long_high_res/).

LACIE_DATA_DIR = "/Volumes/LaCie/000_POSTDOC_2025/long_high_res/"
FIELDS_VAR_LIST = ["PW", "IWP", "Prec", "W500", "Q_sfc", "T_sfc"]
FIELDS_VLIMS = {
    "PW":    (30, 60),
    "IWP":   (0, 0.5),
    "Prec":  (0, 15),
    "W500":  (-3, 3),
    "Q_sfc": (14, 18),
    "T_sfc": (295, 300),
}


def generate_fields_clusters():
    d = setup()
    states = d["states"]
    state_keep_indices = d["state_keep_indices"]

    if not os.path.isdir(LACIE_DATA_DIR):
        print(f"[fields_clusters] LaCie not mounted at {LACIE_DATA_DIR}; skipping.")
        return

    nx, ny = 128, 128
    dtype = np.float32
    nt = states.shape[0]
    clusters = sorted(state_keep_indices.keys())
    closest_indices = {c: np.sort(state_keep_indices[c]) for c in clusters}

    mean_fields = {var: {} for var in FIELDS_VAR_LIST}
    for var in FIELDS_VAR_LIST:
        print(f"[fields_clusters] processing {var}...")
        path = os.path.join(LACIE_DATA_DIR, f"{var}_images_time_evolution.dat")
        mmap = np.memmap(path, dtype=dtype, mode="r", shape=(nt, nx, ny))
        for c in clusters:
            mean_fields[var][c] = mmap[closest_indices[c]].mean(axis=0)
        del mmap

    n_clusters = len(clusters)
    n_vars = len(FIELDS_VAR_LIST)
    fig, axes = plt.subplots(
        n_clusters, n_vars,
        figsize=(3.5 * n_vars, 3.0 * n_clusters),
        squeeze=False,
    )
    for i, c in enumerate(clusters):
        for j, var in enumerate(FIELDS_VAR_LIST):
            ax = axes[i, j]
            img = mean_fields[var][c]
            lim = FIELDS_VLIMS.get(var, (None, None))
            im = ax.imshow(img, origin="lower", vmin=lim[0], vmax=lim[1])
            if i == 0:
                ax.set_title(var, fontsize=13)
            if j == 0:
                ax.set_ylabel(f"Cluster {wfp.state_label(c)}", fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Canonical mean fields per latent state", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(FIGURES_OUT, "fields_clusters.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fields_clusters] saved -> {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DISPATCH = {
    "fig3": generate_fig3_clusters,
    "fig4": generate_fig4_latent_axis,
    "fig_pca_panels": generate_fig_pca_panels,
    "fig5": generate_fig5_vertical_profiles,
    "fig6": generate_fig6_mse_profiles,
    "fig7": generate_fig7_markov_matrix,
    "fig7_blank": generate_fig7_blank,
    "fig9bis": generate_fig9bis_markovianity_coldpools,
    "fig8": generate_fig8_trajectories,
    "training_loss": generate_training_loss,
    "validation_loss": generate_validation_loss,
    "reconstruction": generate_reconstruction,
    "canonical_representative": generate_canonical_representative,
    "pca_3d": generate_pca_3d,
    "cluster_sensitivity": generate_cluster_sensitivity,
    "fields_clusters": generate_fields_clusters,
}

WHICH = ["fig_pca_panels"]


def main():
    for name in WHICH:
        print(f"\n=== {name} ===")
        DISPATCH[name]()


if __name__ == "__main__":
    main()
