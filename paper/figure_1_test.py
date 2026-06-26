"""
figure_1_test.py — 3-D cloud + time-averaged isentropic mass-flux panel (Fig. 1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
from skimage import measure
from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths


# ── Paths ─────────────────────────────────────────────────────
DATA_FOLDER_PATH = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
OUT_DIR          = '/Users/sophieabramian/Documents/troposwhere/paper/figures'


# ============================================================
# Light palette (slide-friendly: transparent background)
# ============================================================

BG_COLOR    = 'none'        # transparent — works on any slide background
OCEAN_COLOR = '#0b1d2a'     # deep ocean (matches paper map)
PANE_COLOR  = '#ffffff'     # pure white pane — brightest backdrop
PANE_ALPHA  = 0.85          # high opacity for a luminous box
TEXT_COLOR  = '#1d1d1f'     # near-black, easier than pure black
MUTED_TEXT  = '#6b7480'
GRID_COLOR  = '#eef1f5'     # airy grid lines
SPINE_COLOR = '#cdd3da'     # soft axis spines
CLOUD_COLOR = (1.0, 1.0, 1.0)   # pure bright white — no tint
LIGHT_AZ    = 315               # side-light direction
LIGHT_ALT   = 45                # higher sun → softer shadows, brighter scene


# ============================================================
# Custom Colormaps
# ============================================================

# Cold-pool cmap: deep ocean (no anomaly) → white (coldest pool).
# Mapping is inverted via vmin/vmax in contourf so vmin=-1 → white.
ocean_cmap = LinearSegmentedColormap.from_list("coldpool", [
    (1.00, 1.00, 1.00),                          # vmin (-1 K): white cold pool
    (0.78, 0.85, 0.92),
    (0.42, 0.58, 0.72),
    (0.18, 0.34, 0.48),
    (11/255, 29/255, 42/255),                    # vmax ( 0 K): OCEAN_COLOR
])


# ============================================================
# Light-theme styling helpers
# ============================================================

def _style_axes3d(ax):
    """Apply the light palette to a Matplotlib 3D axis."""
    ax.set_facecolor((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_alpha(PANE_ALPHA)
        axis.pane.set_edgecolor((0, 0, 0, 0))   # no hard pane edges
        axis.line.set_color(SPINE_COLOR)
        axis.line.set_linewidth(0.8)
        axis.label.set_color(TEXT_COLOR)
    ax.tick_params(colors=MUTED_TEXT, which='both', length=3, width=0.6)


def _style_axes2d(ax):
    """Apply the light palette to a regular 2D axis."""
    ax.set_facecolor((0, 0, 0, 0))
    for side, spine in ax.spines.items():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)
        if side in ('top', 'right'):
            spine.set_visible(False)
    ax.tick_params(colors=MUTED_TEXT, which='both')
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def _style_colorbar(cb, orientation='horizontal'):
    """Apply the light palette to a Matplotlib colorbar."""
    cb.outline.set_edgecolor(SPINE_COLOR)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=MUTED_TEXT, which='both')
    label_axis = cb.ax.xaxis if orientation == 'horizontal' else cb.ax.yaxis
    label_axis.label.set_color(TEXT_COLOR)
    for tick in label_axis.get_ticklabels():
        tick.set_color(MUTED_TEXT)


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
    # No smoothing — the raw QN field gives the tightest, most accurate
    # iso-surface; any gaussian filtering inflates the volumes.
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
    fig   = plt.figure(figsize=(12, 5.8), facecolor=BG_COLOR)
    gs    = fig.add_gridspec(nrows=10, ncols=20,
                             width_ratios=[2]*10 + [1]*10, wspace=0.0)
    ax_3d  = fig.add_subplot(gs[:, :13], projection="3d", computed_zorder=False)
    ax_iso = fig.add_subplot(gs[3:9, 12:])

    _style_axes3d(ax_3d)
    _style_axes2d(ax_iso)

    # ── 4. Surface temperature contour ───────────────────────
    ax_3d.contourf(XX, YY, anomaly, 40, zdir="z", offset=-0.05,
                   cmap=ocean_cmap, vmin=-1.0, vmax=0.0, alpha=0.95)

    # ── 5. Cloud surface — bright opaque white, side-lit for realism ──
    cloud = ax_3d.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color=CLOUD_COLOR,
        edgecolor='none', linewidth=0,
        shade=True, antialiased=True,
        lightsource=LightSource(azdeg=LIGHT_AZ, altdeg=LIGHT_ALT),
    )
    cloud.set_alpha(0.95)
    cloud.set_zorder(5)

    # ── 6. 3D axis formatting ─────────────────────────────────
    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 20))
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 20))
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 5))
    ax_3d.set_xticks(np.arange(0, x[-1] + 1, 4), minor=True)
    ax_3d.set_yticks(np.arange(0, y[-1] + 1, 4), minor=True)
    ax_3d.set_zticks(np.arange(0, z[-1] + 1, 1), minor=True)
    ax_3d.grid(which='minor', linewidth=0.0, color=GRID_COLOR, alpha=0.0)
    ax_3d.grid(which='major', linewidth=0.45, color=GRID_COLOR, alpha=0.9)
    ax_3d.set_xlabel("X [km]", labelpad=20, color=TEXT_COLOR)
    ax_3d.set_ylabel("Y [km]", labelpad=10, color=TEXT_COLOR)
    ax_3d.set_zlabel("Z [km]", color=TEXT_COLOR)
    ax_3d.set_zlim(0, 12)
    ax_3d.view_init(elev=22, azim=-120)
    ax_3d.set_box_aspect([1, 1, 0.20])
    ax_3d.text2D(0.1, 0.82, "Instantaneous\nNon-precipitating\ncondensate",
                 transform=ax_3d.transAxes, ha="left", va="top",
                 style="italic", fontsize=11, color=TEXT_COLOR)

    # ── 7. Temperature colorbar (fixed across frames) ────────
    sm_temp = plt.cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=0.0),
                                    cmap=ocean_cmap)
    sm_temp.set_array([])
    cb = fig.colorbar(sm_temp, ax=ax_3d, orientation='horizontal',
                      fraction=0.06, pad=-0.05, shrink=0.6, aspect=110)
    cb.set_label('Temperature anomaly [K]', color=TEXT_COLOR)
    cb.ax.xaxis.tick_top()
    ticks = np.linspace(-1.0, 0.0, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    _style_colorbar(cb, orientation='horizontal')

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
    ax_iso.contourf(SS, ZZ, field, levels=levels,
                    cmap="RdBu_r", vmin=vmin, vmax=vmax, extend="both")
    ax_iso.set_ylim(0, 15)
    ax_iso.set_xlabel("Moist Static Energy [kJ/kg]")
    ax_iso.set_ylabel("Height [km]")
    iso_title = ("Time-averaged\nIsentropic mass flux" if isentropic_mode == "mean"
                 else "Instantaneous\nIsentropic mass flux")
    ax_iso.text(0.55, 1.2, iso_title,
                transform=ax_iso.transAxes, ha="center", va="top",
                style="italic", fontsize=11, color=TEXT_COLOR)
    ax_iso.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    # ── 9. Isentropic colorbar (fixed across frames) ─────────
    sm_iso = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                   cmap="RdBu_r")
    sm_iso.set_array([])
    cb_iso = fig.colorbar(sm_iso, ax=ax_iso, orientation="vertical",
                          fraction=0.18, pad=0.05, aspect=60, extend="both")
    cb_iso.set_label(r"[$10^{-3}$ kg m$^{2}$ s$^{-1}$ K$^{-1}$]",
                     color=TEXT_COLOR)
    cb_iso.set_ticks(np.linspace(vmin, vmax, 6))
    _style_colorbar(cb_iso, orientation='vertical')

    return fig


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    simu_parameters = {
        'velocity':    '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split':       '5',
    }

    # Squall-line development window — frame names use the old convention
    # frame_index = i_file * 1000 + time_index, so 004379 → split 4, t=379
    #                                              005177 → split 5, t=177.
    FRAME_START = 4300
    FRAME_END   = 5177        # inclusive

    start_split, t_start = divmod(FRAME_START, 1000)
    end_split,   t_end   = divmod(FRAME_END,   1000)

    backup_path = os.path.join(DATA_FOLDER_PATH, 'saved_simu')

    for i_file in range(start_split, end_split + 1):
        simu = load_simulation(simu_parameters, i=i_file)
        if simu is None:
            print(f"Split {i_file} failed to load — skipping.")
            continue

        if os.path.exists(backup_path):
            simu.load(backup_folder_path=backup_path)

        qn       = simu.dataset_3d.QN.values
        vsurf    = simu.dataset_3d.TABS[:, 0]
        x        = simu.dataset_3d.x / 1000
        y        = simu.dataset_3d.y / 1000
        z        = simu.dataset_3d.z / 1000
        isentrop = simu.dataset_isentropic.RHO_W_sum.values

        # Size the isentropic axes to the actual array shape.
        # isentrop[t] is (nz_iso, nmse) — instantaneous slices are typically (64, 50).
        nz_iso, nmse = isentrop.shape[1], isentrop.shape[2]
        z_iso        = z[:nz_iso]
        mse_axis     = np.linspace(320, 350, nmse)

        nt   = qn.shape[0]
        t_lo = t_start if i_file == start_split else 0
        t_hi = t_end   if i_file == end_split   else nt - 1
        t_hi = min(t_hi, nt - 1)

        for t in range(t_lo, t_hi + 1):
            frame_index = i_file * 1000 + t
            print(f"Split {i_file}, t={t} → frame {frame_index:06d}")
            plt.ioff()
            fig = plot_3d_visualization(
                t, qn, vsurf, x, y, z, z_iso, isentrop,
                mse=mse_axis,
                isentropic_mode="instantaneous",   # isentropic panel evolves with frame
            )
            out = os.path.join(OUT_DIR, f'img_{frame_index:06d}.png')
            fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)


if __name__ == "__main__":
    main()
