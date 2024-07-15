#!/bin/bash
#SBATCH --job-name=task1_td3    # Job name
#SBATCH --output=output_%A_%a.log            # Output log file (with array job and task IDs)
#SBATCH --error=error_%A_%a.log              # Error log file (with array job and task IDs)
#SBATCH --ntasks=1                           # Number of tasks per array job
#SBATCH --cpus-per-task=5                   # Number of CPU cores per task
#SBATCH --mem=10G                             # Memory per node
#SBATCH --time=24:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=standard                 # Partition name
#SBATCH --array=1-4                         # Array job with tasks

# Load necessary modules (if any)
module load anaconda3

# Activate the conda environment
source activate tianshou1

# Set the PYTHONPATH
export PYTHONPATH=./

# Run the Python script
python DTRBench/run_RL/online_discrete_search.py

# Deactivate the conda environment
conda deactivate
