import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# Use the local tropokit copy instead of the installed package
sys.path.insert(0, '/Users/sophieabramian/Documents/troposwhere')

import tropokit

# --------------------------
# CONFIG
# --------------------------

PKL_PATH   = '/Users/sophieabramian/Documents/troposwhere/data/diagnostics/all_cold_pools.pkl'
OUTPUT_DIR = '/Users/sophieabramian/Documents/troposwhere/data/diagnostics'


# --------------------------
# MAIN
# --------------------------

if __name__ == '__main__':
    # Load cold pool entries produced by extract_coldpools.py
    with open(PKL_PATH, 'rb') as f:
        all_cold_pools = pickle.load(f)

    print(f'Loaded cold pool entries: {len(all_cold_pools)}')

    # Flatten all ColdPool objects across simulations into a single list
    cold_pools = [cp for cps in all_cold_pools.values() for cp in cps]

    # Build a raw CPI time series: total cold pool area at each timestep
    max_ts  = max(t for cp in cold_pools for t in cp.cluster['timesteps'])
    CPI_raw = np.zeros(max_ts + 1)

    for cp in tqdm(cold_pools, desc='Building CPI'):
        for t, size in zip(cp.cluster['timesteps'], cp.cluster['sizes']):
            CPI_raw[t] += size

    # Derived normalizations
    mean_cpi = np.mean(CPI_raw)
    std_cpi  = np.std(CPI_raw)

    CPI_norm = CPI_raw / (mean_cpi + 1e-12)
    CPI_anom = CPI_raw - mean_cpi
    CPI_std  = CPI_anom / (std_cpi + 1e-12)

    # Save all variants
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f'{OUTPUT_DIR}/CPI_raw.npy',     CPI_raw)
    np.save(f'{OUTPUT_DIR}/CPI_norm.npy',    CPI_norm)
    np.save(f'{OUTPUT_DIR}/CPI_anomaly.npy', CPI_anom)
    np.save(f'{OUTPUT_DIR}/CPI_std.npy',     CPI_std)

    print('Saved CPI time series.')
    print(f'Length : {len(CPI_raw)}')
    print(f'Mean   : {mean_cpi:.4f}')
    print(f'Std    : {std_cpi:.4f}')
