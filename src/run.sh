#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=main_out.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# Load the numactl module to enable numa library linking
module load numactl

# Compile
g++ -O3 -fopenmp main.cpp seam_dp.cpp image_energy.cpp -lm -lnuma -o main

# Run
srun main test_images/1024x768.png valve-out.png
