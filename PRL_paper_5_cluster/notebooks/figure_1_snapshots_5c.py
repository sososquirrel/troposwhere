"""Figure 1 for the K=5 manuscript: 3-row composite using the EXACT
single-panel layout of `paper/figure_1_test.py::plot_3d_visualization`
(the same one that produced `img_004015.png`).

Each row is rendered as its own (12, 5.8) figure with bbox_inches='tight'
— byte-identical aspect to img_004015.png — then the 3 cropped PNGs are
stacked vertically in a single composite PDF with panel labels (a., b., ...).

Requirements:
- LaCie drive mounted (/Volumes/LaCie/000_POSTDOC_2025/long_high_res).
- samenv conda env + KMP_DUPLICATE_LIB_OK=TRUE.

Output:
    paper_5clusters/manuscript/figures/fig1_snapshots.pdf
"""

import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from skimage import measure

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = "/Users/sophieabramian/Documents/troposwhere"
sys.path.insert(0, os.path.join(ROOT, "paper"))

from figure_1_test import (
    load_simulation, DATA_FOLDER_PATH,
    ocean_cmap, _style_axes3d, _style_axes2d, _style_colorbar,
    BG_COLOR, CLOUD_COLOR, GRID_COLOR, TEXT_COLOR, LIGHT_AZ, LIGHT_ALT,
)


OUT_DIR = os.path.join(ROOT, "paper_5clusters", "manuscript", "figures")
assert OUT_DIR.startswith(os.path.join(ROOT, "paper_5clusters")), \
    "Output must stay inside paper_5clusters/"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NAME = "fig1_snapshots.pdf"


SIMU = dict(velocity="8", temperature="300", bowen_ratio="1",
            microphysic="1", split="5")

# Each row: (split_file_idx → loads `3D/split_{idx+1}.nc`, t-index within that split).
# Row 1 (a/b): suppressed     — split_5.nc,  t=289     → K=5 state 2 (display "3")
# Row 2 (c/d): img_004015      — split_5.nc,  t=15      → K=5 state 1 (display "2")
# Row 3 (e/f): unorganized act — split_38.nc, t=344     → K=5 state 3 (display "4")
SNAPSHOTS = [(4, 289), (4, 15), (37, 344)]


def plot_single_row(i, qn, vsurf, x, y, z, z_iso, isentrop, mse):
    """Exact copy of plot_3d_visualization() but with banner texts removed
    and the corrected mass-flux colorbar label.
    """
    y2 = np.swapaxes(np.array(qn[i]), 0, 2)
    verts, faces, _, _ = measure.marching_cubes(y2, level=0.02)
    verts[:, 0] = x[verts[:, 0].astype(int)]
    verts[:, 1] = y[verts[:, 1].astype(int)]
    verts[:, 2] = z[verts[:, 2].astype(int)]

    vsurf_np = np.array(vsurf[i])
    anomaly = np.clip(vsurf_np - vsurf_np.mean(), -1.0, 0.0)
    XX, YY = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 5.8), facecolor=BG_COLOR)
    gs = fig.add_gridspec(nrows=10, ncols=20,
                          width_ratios=[2] * 10 + [1] * 10, wspace=0.0)
    ax_3d = fig.add_subplot(gs[:, :13], projection="3d", computed_zorder=False)
    ax_iso = fig.add_subplot(gs[3:9, 12:])

    _style_axes3d(ax_3d)
    _style_axes2d(ax_iso)

    ax_3d.contourf(XX, YY, anomaly, 40, zdir="z", offset=-0.05,
                   cmap=ocean_cmap, vmin=-1.0, vmax=0.0, alpha=0.95)
    cloud = ax_3d.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color=CLOUD_COLOR, edgecolor="none", linewidth=0,
        shade=True, antialiased=True,
        lightsource=LightSource(azdeg=LIGHT_AZ, altdeg=LIGHT_ALT),
    )
    cloud.set_alpha(0.95)
    cloud.set_zorder(5)

    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 20))
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 20))
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 5))
    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 4), minor=True)
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 4), minor=True)
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 1), minor=True)
    ax_3d.grid(which="minor", linewidth=0.0, color=GRID_COLOR, alpha=0.0)
    ax_3d.grid(which="major", linewidth=0.45, color=GRID_COLOR, alpha=0.9)
    ax_3d.set_xlabel("X [km]", labelpad=20, color=TEXT_COLOR)
    ax_3d.set_ylabel("Y [km]", labelpad=10, color=TEXT_COLOR)
    ax_3d.set_zlabel("Z [km]", color=TEXT_COLOR)
    ax_3d.set_zlim(0, 12)
    ax_3d.view_init(elev=22, azim=-120)
    ax_3d.set_box_aspect([1, 1, 0.20])

    sm_temp = plt.cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=0.0),
                                    cmap=ocean_cmap)
    sm_temp.set_array([])
    cb = fig.colorbar(sm_temp, ax=ax_3d, orientation="horizontal",
                      fraction=0.06, pad=-0.05, shrink=0.6, aspect=110)
    cb.set_label("Temperature anomaly [K]", color=TEXT_COLOR)
    cb.ax.xaxis.tick_top()
    ticks = np.linspace(-1.0, 0.0, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    _style_colorbar(cb, orientation="horizontal")

    vmin, vmax = -25, 25
    field = np.clip(isentrop[i], vmin, vmax)
    levels = np.unique(np.concatenate([
        np.linspace(-25, -18, 6), np.linspace(-10, -5, 5),
        np.linspace(-5, 5, 15), np.linspace(5, 10, 5),
        np.linspace(10, 25, 6),
    ]))
    SS, ZZ = np.meshgrid(mse, z_iso)
    ax_iso.contourf(SS, ZZ, field, levels=levels,
                    cmap="RdBu_r", vmin=vmin, vmax=vmax, extend="both")
    ax_iso.set_ylim(0, 15)
    ax_iso.set_xlabel("Moist Static Energy [kJ/kg]")
    ax_iso.set_ylabel("Height [km]")
    ax_iso.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    sm_iso = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                   cmap="RdBu_r")
    sm_iso.set_array([])
    cb_iso = fig.colorbar(sm_iso, ax=ax_iso, orientation="vertical",
                          fraction=0.18, pad=0.05, aspect=60, extend="both")
    cb_iso.set_label(r"Vertical mass flux [$10^{-3}$ kg m$^{-2}$ s$^{-1}$ K$^{-1}$]",
                    color=TEXT_COLOR, fontsize=8)
    cb_iso.set_ticks(np.linspace(vmin, vmax, 6))
    _style_colorbar(cb_iso, orientation="vertical")

    return fig


def render_row_to_array(fig, dpi=200):
    """Save fig to a PNG buffer with bbox_inches='tight' and return as ndarray."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ---------------------------------------------------------------------------
backup = os.path.join(DATA_FOLDER_PATH, "saved_simu")
_simu_cache = {}


def get_split(split_idx):
    """Lazily load + cache a simulation by split-file index (i in load_simulation)."""
    if split_idx in _simu_cache:
        return _simu_cache[split_idx]
    print(f"[fig1] loading split_{split_idx + 1}.nc ...")
    simu = load_simulation(SIMU, i=split_idx)
    if simu is None:
        raise RuntimeError(
            f"Failed to load split_{split_idx + 1}.nc. Is LaCie mounted?"
        )
    if os.path.exists(backup):
        simu.load(backup_folder_path=backup)
    qn_ = simu.dataset_3d.QN.values
    vsurf_ = simu.dataset_3d.TABS[:, 0]
    x_ = (simu.dataset_3d.x / 1000).values
    y_ = (simu.dataset_3d.y / 1000).values
    z_ = (simu.dataset_3d.z / 1000).values
    isentrop_ = simu.dataset_isentropic.RHO_W_sum.values
    nz_iso, nmse = isentrop_.shape[1], isentrop_.shape[2]
    z_iso_ = z_[:nz_iso]
    mse_axis_ = np.linspace(320, 350, nmse)
    bundle = dict(
        qn=qn_, vsurf=vsurf_, x=x_, y=y_, z=z_, z_iso=z_iso_,
        isentrop=isentrop_, mse_axis=mse_axis_, nt=qn_.shape[0],
    )
    _simu_cache[split_idx] = bundle
    return bundle


# Render each row separately so per-row aspect is identical to img_004015.png.
row_imgs = []
for split_idx, t in SNAPSHOTS:
    s = get_split(split_idx)
    if not 0 <= t < s["nt"]:
        raise IndexError(
            f"t={t} out of range for split_{split_idx + 1} (nt={s['nt']})"
        )
    print(f"[fig1] rendering row split_{split_idx + 1}.nc  t={t}")
    plt.ioff()
    row_fig = plot_single_row(
        t, s["qn"], s["vsurf"], s["x"], s["y"], s["z"], s["z_iso"],
        s["isentrop"], s["mse_axis"],
    )
    row_imgs.append(render_row_to_array(row_fig))

# Compose: stack the 3 row images in one figure with panel labels.
n_rows = len(row_imgs)
img_h, img_w = row_imgs[0].shape[:2]
row_in = 5.0   # display height per row (inches)
fig_w = 12.0
fig_h = row_in * n_rows
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
gs = fig.add_gridspec(nrows=n_rows, ncols=1, hspace=-0.18)

panel_alphabet = "abcdefghij"
for r, img in enumerate(row_imgs):
    ax = fig.add_subplot(gs[r, 0])
    ax.imshow(img, interpolation="bilinear")
    ax.set_axis_off()
    # Panel labels — placed in axes-fraction coords (transAxes is NOT inverted
    # by imshow). y close to 1.0 + va="top" puts them just inside the top edge.
    ax.text(0.10, 0.78, f"{panel_alphabet[2 * r]}.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, color=TEXT_COLOR, fontweight="bold")
    ax.text(0.65, 0.78, f"{panel_alphabet[2 * r + 1]}.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, color=TEXT_COLOR, fontweight="bold")

out_path = os.path.join(OUT_DIR, OUT_NAME)
fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
print(f"[fig1] saved -> {out_path}")

if __name__ == "__main__" and "DISPLAY" in os.environ:
    plt.show()
