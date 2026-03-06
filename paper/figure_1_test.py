"""
figure_1_test.py — 3-D cloud + time-averaged isentropic mass-flux panel (Fig. 1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
from skimage import measure
from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths


# ── Paths ─────────────────────────────────────────────────────
DATA_FOLDER_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
OUT_DIR          = '/Users/sophieabramian/Documents/troposwhere/paper/figures'


# ============================================================
# Custom Colormaps
# ============================================================

ocean_cmap = LinearSegmentedColormap.from_list("ocean_cmap", [
    (222/255, 243/255, 226/255),   # shallow
    (130/255, 200/255, 210/255),
    ( 20/255, 150/255, 200/255),   # turquoise
    ( 10/255, 120/255, 185/255),
    (  0/255,  90/255, 160/255),
    (  0/255,  65/255, 125/255),
    (  0/255,  40/255,  90/255),   # deep
    (  6/255,  66/255, 115/255),
])

cloud_cmap = LinearSegmentedColormap.from_list("cloudmap", [
    (0.0, (0.85, 0.85, 0.90)),
    (0.3, (0.92, 0.92, 0.95)),
    (1.0, (1.00, 1.00, 1.00)),
])


# ============================================================
# Simulation Loader
# ============================================================

def load_simulation(simu_parameters, i=None, path_raw_data=DATA_FOLDER_PATH):
    """Load a Simulation object. If `i` is given, builds paths from split index."""
    try:
        if i is not None:
            paths = {
                'path_3d': os.path.join(path_raw_data, f'3D/split_{i+1}.nc'),
                'path_2d': os.path.join(path_raw_data, f'2D/split_{i+1}.nc'),
                'path_1d': os.path.join(path_raw_data, f'1D/split_{i+1}.nc'),
            }
        else:
            paths = generate_simulation_paths(**simu_parameters, folder_path=path_raw_data)

        return Simulation(
            data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
            **simu_parameters,
        )
    except Exception as e:
        print(f"Simulation loading error: {e}")
        return None


# ============================================================
# Plotting
# ============================================================

def plot_3d_visualization(
    i, qn, vsurf, x, y, z, z_iso, isentrop,
    mse=np.linspace(320, 350, 48),
    isentropic_mode="instantaneous",   # "instantaneous" | "mean"
    isentrop_mean=None,
):
    # ── 1. Cloud surface (marching cubes) ────────────────────
    y2 = np.swapaxes(np.array(qn[i]), 0, 2)
    verts, faces, _, _ = measure.marching_cubes(y2, level=0.02)
    verts[:, 0] = x[verts[:, 0].astype(int)]
    verts[:, 1] = y[verts[:, 1].astype(int)]
    verts[:, 2] = z[verts[:, 2].astype(int)]

    # ── 2. Surface temperature anomaly ───────────────────────
    vsurf_np = np.array(vsurf[i])
    anomaly  = np.clip(vsurf_np - vsurf_np.mean(), -1.0, 0.0)
    XX, YY   = np.meshgrid(x, y)

    # ── 3. Layout ─────────────────────────────────────────────
    fig   = plt.figure(figsize=(12, 5.8))
    gs    = fig.add_gridspec(nrows=10, ncols=20,
                             width_ratios=[2]*10 + [1]*10, wspace=0.0)
    ax_3d = fig.add_subplot(gs[:, :13], projection="3d", computed_zorder=False)
    ax_iso = fig.add_subplot(gs[3:9, 12:])

    # ── 4. Surface temperature contour ───────────────────────
    cset = ax_3d.contourf(XX, YY, anomaly, 40, zdir="z", offset=-0.05,
                          cmap=ocean_cmap, vmin=-1.0, vmax=0.0, alpha=0.8)

    # ── 5. Cloud surface ──────────────────────────────────────
    ax_3d.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color=(1, 1, 1, 0.95), linewidth=0.02,
        shade=True, lightsource=LightSource(azdeg=315, altdeg=0),
    )

    # ── 6. 3D axis formatting ─────────────────────────────────
    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 20))
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 20))
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 5))
    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 4), minor=True)
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 4), minor=True)
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 1), minor=True)
    ax_3d.grid(which='minor', linewidth=0.15, color='gray',  alpha=0.45)
    ax_3d.grid(which='major', linewidth=0.50, color='white', alpha=0.60)
    ax_3d.set_xlabel("X [km]", labelpad=20)
    ax_3d.set_ylabel("Y [km]", labelpad=10)
    ax_3d.set_zlabel("Z [km]")
    ax_3d.set_zlim(0, 12)
    ax_3d.view_init(elev=22, azim=-120)
    ax_3d.set_box_aspect([1, 1, 0.20])
    ax_3d.text2D(0.1, 0.82, "Instantaneous\nNon-precipitating \ncondensate",
                 transform=ax_3d.transAxes, ha="left", va="top",
                 style="italic", fontsize=11)

    # ── 7. Temperature colorbar ───────────────────────────────
    cb = fig.colorbar(cset, ax=ax_3d, orientation='horizontal',
                      fraction=0.06, pad=-0.05, shrink=0.6, aspect=110)
    cb.set_label('Temperature anomaly [K]')
    cb.ax.xaxis.tick_top()
    ticks = np.linspace(-1.0, 0.0, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])

    # ── 8. Isentropic panel ───────────────────────────────────
    if isentropic_mode == "mean":
        if isentrop_mean is None:
            raise ValueError("isentrop_mean must be provided for mean mode")
        field = isentrop_mean.copy()
    else:
        field = isentrop[i]

    vmin, vmax = -25, 25
    field  = np.clip(field, vmin, vmax)
    levels = np.unique(np.concatenate([
        np.linspace(-25, -18,  6), np.linspace(-10,  -5,  5),
        np.linspace( -5,   5, 15), np.linspace(  5,  10,  5),
        np.linspace( 10,  25,  6),
    ]))
    SS, ZZ = np.meshgrid(mse, z_iso)
    pcm = ax_iso.contourf(SS, ZZ, field, levels=levels,
                          cmap="RdBu_r", vmin=vmin, vmax=vmax, extend="both")
    ax_iso.set_ylim(0, 15)
    ax_iso.set_xlabel("Moist Static Energy [kJ/kg]")
    ax_iso.set_ylabel("Height [km]")
    ax_iso.text(0.55, 1.2, "Time-averaged\nIsentropic mass flux",
                transform=ax_iso.transAxes, ha="center", va="top",
                style="italic", fontsize=11)
    ax_iso.grid(True)

    # ── 9. Isentropic colorbar ────────────────────────────────
    cb_iso = fig.colorbar(pcm, ax=ax_iso, orientation="vertical",
                          fraction=0.18, pad=0.05, aspect=60)
    cb_iso.set_label(r"[$10^{-3}$ kg m$^{2}$ s$^{-1}$ K$^{-1}$]")

    return fig


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rho_w_mean = np.load(os.path.join(DATA_FOLDER_PATH, 'reshaped_rho_w_sum.npy')).mean(axis=0)

    simu_parameters = {
        'velocity':    '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split':       '5',
    }

    simu = load_simulation(simu_parameters, i=4)
    if simu is None:
        print("Simulation failed to load.")
        return

    backup_path = os.path.join(DATA_FOLDER_PATH, 'saved_simu')
    if os.path.exists(backup_path):
        simu.load(backup_folder_path=backup_path)

    qn       = simu.dataset_3d.QN.values
    vsurf    = simu.dataset_3d.TABS[:, 0]
    x        = simu.dataset_3d.x / 1000
    y        = simu.dataset_3d.y / 1000
    z        = simu.dataset_3d.z / 1000
    z_iso    = z[:48]
    isentrop = simu.dataset_isentropic.RHO_W_sum.values

    for time_index in [15]:
        time_index = min(time_index, qn.shape[0] - 1)
        print(f"Rendering time index {time_index}...")
        plt.ioff()
        fig = plot_3d_visualization(
            time_index, qn, vsurf, x, y, z, z_iso, isentrop,
            isentropic_mode="mean",
            isentrop_mean=rho_w_mean,
        )
        out = os.path.join(OUT_DIR, f'img_{4:03d}{time_index:03d}.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
