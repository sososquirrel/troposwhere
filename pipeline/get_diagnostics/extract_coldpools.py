import os
import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import tropokit
from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths
from tropokit.ColdPool import extract_cold_pools


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
PATH_SAVED    = '/Users/sophieabramian/Documents/troposwhere/data/squall_lines/saved_simu'
OUTPUT_PKL    = '/Users/sophieabramian/Documents/troposwhere/data/diagnostics/all_cold_pools.pkl'


def load_simulation(simu_parameters, split_index=None):
    """
    Build a Simulation object from raw netCDF files.

    Args:
        simu_parameters: Dict of simulation parameters (velocity, temperature, ...).
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


#LIST_FILES = [f'split_{i}' for i in range(4, 40)]
LIST_FILES = ['split_1', 'split_2']

if __name__ == "__main__":
    # Use spawn to avoid fork-related issues with multiprocessing on macOS/Linux
    mp.set_start_method("spawn", force=True)

    all_cold_pools = {}

    for i_file, file in tqdm(enumerate(LIST_FILES, start=0)):
        print(f"\n--- Processing {file} ---")
        parameters = data_dict[file]

        simu = load_simulation(parameters, split_index=i_file)
        if simu is None:
            print(f"Skipping {file}: could not load simulation.")
            continue

        print(f"Loaded: {simu.name}")

        # Load previously computed diagnostics (cold pool labels, etc.)
        simu.load(backup_folder_path=PATH_SAVED)

        # Extract cold pool objects from precomputed label and moisture fields
        label_array = simu.dataset_computed_2d.CP_LABELS.values
        qv_array    = simu.dataset_3d.QV[:, 0].values
        cold_pools  = extract_cold_pools(label_array, qv_array)

        all_cold_pools[simu.name] = cold_pools

    # Save all cold pool data to disk
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(all_cold_pools, f)

    print(f"\nSaved cold pool data to {OUTPUT_PKL}")
    print(f"Simulations processed: {len(all_cold_pools)}")
