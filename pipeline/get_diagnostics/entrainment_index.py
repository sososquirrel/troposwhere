import os
import numpy as np
from tqdm import tqdm

from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths
from tropokit.diagnostic_fmse import calculate_entrainment_detrainment

# --------------------------
# CONFIG
# --------------------------

PATH_RAW_DATA = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
PATH_SAVED    = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/saved_simu'
OUTPUT_DIR    = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res/indexes'

# Heights (m) at which to sample entrainment/detrainment
TARGET_HEIGHTS = np.array([200, 1000, 7000, 10000, 13000])

# One entry per time split (48 total)
data_dict = {
    f'split_{i}': {
        'velocity': '8',
        'temperature': '300',
        'bowen_ratio': '1',
        'microphysic': '1',
        'split': str(i),
    }
    for i in range(1, 49)
}

LIST_FILES = [f'split_{i}' for i in range(4, 40)]


# --------------------------
# HELPERS
# --------------------------

def load_simulation(simu_parameters, split_index=None):
    """
    Build a Simulation object from raw netCDF files.

    Args:
        simu_parameters: Dict of simulation parameters.
        split_index: If provided, load from split_<split_index+1>.nc files;
                     otherwise derive paths from simu_parameters.

    Returns:
        Simulation object, or None if loading failed.
    """
    try:
        if split_index is not None:
            paths = {
                'path_3d': os.path.join(PATH_RAW_DATA, f'3D/split_{split_index + 1}.nc'),
                'path_2d': os.path.join(PATH_RAW_DATA, f'2D/split_{split_index + 1}.nc'),
                'path_1d': os.path.join(PATH_RAW_DATA, f'1D/split_{split_index + 1}.nc'),
            }
        else:
            paths = generate_simulation_paths(**simu_parameters, folder_path=PATH_RAW_DATA)

        simu = Simulation(
            data_folder_paths=[paths['path_1d'], paths['path_2d'], paths['path_3d']],
            **simu_parameters,
        )
        return simu

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error loading simulation: {e}")

    return None


# --------------------------
# MAIN
# --------------------------

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entrainment_all = []
    detrainment_all = []
    net_all         = []  # E - D

    for i_file, file in tqdm(enumerate(LIST_FILES, start=4), desc='Simulations'):
        print(f'\nRun {file}')
        parameters = data_dict[file]

        simu = load_simulation(parameters, split_index=i_file)
        if simu is None:
            print(f'Skipping {file}: could not load simulation.')
            continue

        # Load previously computed diagnostics
        simu.load(backup_folder_path=PATH_SAVED)

        # Vertical coordinate and target level indices
        z          = simu.dataset_3d.z.values  # (nz,) in meters
        idx_levels = np.abs(z[:, None] - TARGET_HEIGHTS[None, :]).argmin(axis=0)

        sim_E  = []
        sim_D  = []
        sim_ED = []

        for t in tqdm(range(simu.nt), desc=file, leave=False):
            try:
                res = calculate_entrainment_detrainment(simu, t, epsilon=1.0)
            except Exception as e:
                print(f'Skipping t={t}: {e}')
                continue

            # Sample profiles at the target heights
            sim_E.append(res['E'][idx_levels])
            sim_D.append(res['D'][idx_levels])
            sim_ED.append(res['E_minus_D'][idx_levels])

        # Stack timesteps for this split
        entrainment_all.append(np.array(sim_E))
        detrainment_all.append(np.array(sim_D))
        net_all.append(np.array(sim_ED))

    # --------------------------
    # MERGE AND SAVE
    # --------------------------
    entrainment_all = np.vstack(entrainment_all)  # (N_total_t, n_levels)
    detrainment_all = np.vstack(detrainment_all)
    net_all         = np.vstack(net_all)

    np.save(os.path.join(OUTPUT_DIR, 'entrainment_levels.npy'), entrainment_all)
    np.save(os.path.join(OUTPUT_DIR, 'detrainment_levels.npy'), detrainment_all)
    np.save(os.path.join(OUTPUT_DIR, 'net_levels.npy'),         net_all)
    np.save(os.path.join(OUTPUT_DIR, 'entrainment_heights.npy'), TARGET_HEIGHTS)

    print('\nSaved entrainment/detrainment index time series.')
    print(f'Entrainment : {entrainment_all.shape}')
    print(f'Detrainment : {detrainment_all.shape}')
    print(f'Net (E-D)   : {net_all.shape}')
    print(f'Heights (m) : {TARGET_HEIGHTS}')
