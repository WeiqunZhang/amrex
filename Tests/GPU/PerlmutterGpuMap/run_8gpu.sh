#!/bin/bash
#SBATCH -A m4546_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 00:45:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -J fb_8gpu
#SBATCH -o fb_8gpu.o%j

# Submit from this directory: sbatch run_8gpu.sh
NNODES=2
source "${SLURM_SUBMIT_DIR}/fb_variants.sh"
