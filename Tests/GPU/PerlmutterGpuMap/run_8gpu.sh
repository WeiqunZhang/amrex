#!/bin/bash
#SBATCH -A m4546_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 01:15:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -J fb_8gpu
#SBATCH -o fb_8gpu.o%j

# Submit from this directory: sbatch run_8gpu.sh
# Must match the modules used to build the executable.
# Comment these out to use the current defaults (CPE 26.03, CUDA 13.2).
module load cpe/25.09
module load cudatoolkit/12.9
module load craype-accel-nvidia80
# Non-default CPE: otherwise the loader picks the default MPI from /opt/cray/pe/lib64.
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

NNODES=2
source "${SLURM_SUBMIT_DIR}/fb_variants.sh"
