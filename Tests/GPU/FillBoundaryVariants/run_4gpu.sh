#!/bin/bash
#SBATCH -A m4546_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 01:15:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -J fbv_4gpu
#SBATCH -o fbv_4gpu.o%j

# Submit from this directory: sbatch run_4gpu.sh
# Uses the default Perlmutter modules; build with the same.

NNODES=1
source "${SLURM_SUBMIT_DIR}/fb_variants.sh"
