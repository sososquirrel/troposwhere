#!/bin/bash

#SBATCH -A glab
#SBATCH -N 32                    # Number of nodes
#SBATCH -c 2                     # CPUs per task
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-10:30           # Runtime (D-HH:MM)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sga2133@columbia.edu
#SBATCH --mem=128G

module load python/3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate samenv

python process_raw_to_simulations.py