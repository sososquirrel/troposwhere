import os
import sys
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# Use the local tropokit copy instead of the installed package
sys.path.insert(0, '/Users/sophieabramian/Documents/troposwhere')

import tropokit
from tropokit.Simulation import Simulation
from tropokit.utils import generate_simulation_paths

from pipeline.raw_to_simulations.process_raw_to_simulations import data_dict, load_simulation


# --------------------------
# CONFIG
# --------------------------

PATH_SAVED  = '/Users/sophieabramian/Documents/troposwhere/data/squall_lines/saved_simu'
OUTPUT_PATH = '/Users/sophieabramian/Documents/troposwhere/data/ml_data/rho_w_dataset.npy'

#LIST_FILES = [f'split_{i}' for i in range(4, 40)]
LIST_FILES = ['split_1', 'split_2']


# --------------------------
# MAIN
# --------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    all_reshaped_data = []

    for i_file, file in tqdm(enumerate(LIST_FILES, start=0)):
        print(f"\n--- Processing {file} ---")
        parameters = data_dict[file]

        simu = load_simulation(parameters, split_index=i_file)
        if simu is None:
            print(f"Skipping {file}: could not load simulation.")
            continue

        print(f"Loaded: {simu.name}")
        simu.load(backup_folder_path=PATH_SAVED)

        # Extract and reshape isentropic mass flux field
        try:
            new_variable = simu.dataset_isentropic.RHO_W_sum[:, :48, 1:49]
            reshaped_data = np.array(new_variable).reshape(-1, 48, 48)
            all_reshaped_data.append(reshaped_data)
        except Exception as e:
            print(f"Failed to extract data for {file}: {e}")
            continue

    # Concatenate and save
    if all_reshaped_data:
        final_data = np.concatenate(all_reshaped_data, axis=0)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        np.save(OUTPUT_PATH, final_data)
        print(f"\nSaved: {OUTPUT_PATH}  shape={final_data.shape}")
    else:
        print("No data collected. Nothing to save.")
