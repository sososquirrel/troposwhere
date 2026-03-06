import os
import sys
import multiprocessing as mp

# Use the local tropokit copy instead of the installed package
sys.path.insert(0, '/Users/sophieabramian/Documents/troposwhere')

import numpy as np
from tqdm import tqdm

from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths
from tropokit.basic_variables import set_basic_variables_from_dataset
from tropokit.coldpool_tracking import get_coldpool_tracking_images
from tropokit.diagnostic_fmse import get_isentropic_dataset, add_counts_to_isentropic_dataset


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



#PATH_RAW_DATA = '/Volumes/LaCie/000_POSTDOC_2025/long_high_res'
PATH_RAW_DATA = '/Users/sophieabramian/Documents/troposwhere/data/squall_lines'
PATH_SAVED = '/Users/sophieabramian/Documents/troposwhere/data/squall_lines/saved_simu'


def load_simulation(simu_parameters, split_index=None):
    """
    Build a Simulation object from raw netCDF files.

    Args:
        simu_parameters: Dict of simulation parameters (velocity, temperature, …).
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



#LIST_FILES = [f'split_{i}' for i in range(11, 40)]
LIST_FILES = ['split_1', 'split_2']

if __name__ == "__main__":
    # Use spawn to avoid fork-related issues with multiprocessing on macOS/Linux
    mp.set_start_method("spawn", force=True)

    for i_file, file in tqdm(enumerate(LIST_FILES, start=0)):
        print(f"\n--- Processing {file} ---")
        parameters = data_dict[file]

        simu = load_simulation(parameters, split_index=i_file)
        if simu is None:
            print(f"Skipping {file}: could not load simulation.")
            continue
        
        print(f"Loaded: {simu.name}")

        # Compute basic thermodynamic variables (FMSE, buoyancy, mass flux, etc.)
        set_basic_variables_from_dataset(simu)

        # Detect and track cold pools using surface temperature (TABS at level 0)
        try:
            variable_images = simu.dataset_3d.TABS[:, 0].values
            # 5th percentile = cold pool core threshold, 10th = envelope threshold
            t_core, t_envelope = np.quantile(variable_images, [0.05, 0.1])
            get_coldpool_tracking_images(
                simulation=simu,
                variable_images=variable_images,
                low_threshold=t_envelope,
                high_threshold=t_core,
            )
        except Exception as e:
            print(f"Cold pool tracking failed: {e}")

        # Compute isentropic diagnostics (mass flux in FMSE coordinates)
        try:
            get_isentropic_dataset(simulation=simu)
        except Exception as e:
            print(f"Isentropic dataset failed: {e}")

        simu.save(backup_folder_path=PATH_SAVED, locking_h5=True)

