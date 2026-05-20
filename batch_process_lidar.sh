#!/bin/bash
#SBATCH --job-name=batch_process_lidar
#SBATCH --account=chw06003
#SBATCH --partition=priority
#SBATCH --qos=co-pi-epyc
#SBATCH --constraint=epyc128
#SBATCH --mem-per-cpu=3G  # Increased to 4G per CPU
#SBATCH --ntasks=100  # Total CPUs
#SBATCH --nodes=1  # Assuming all CPUs are on a single node
#SBATCH --time=4-00:00:00



# Activate your Python environment
conda activate JKL



# Run the Python script
python3 batch_process_lidar.py

