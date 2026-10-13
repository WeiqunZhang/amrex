#!/bin/bash
#SBATCH -A m4546_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 00:45:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -J fb_4gpu
#SBATCH -o fb_4gpu.o%j

# Submit from this directory: sbatch run_4gpu.sh
NNODES=1
source "${SLURM_SUBMIT_DIR}/fb_variants.sh"
