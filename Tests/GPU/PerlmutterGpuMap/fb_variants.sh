# Sourced by run_4gpu.sh and run_8gpu.sh. Needs NNODES set.

cd "${SLURM_SUBMIT_DIR}"

EXE=${EXE:-$(ls ./main3d.*CUDA*.ex | head -1)}
REPS=${REPS:-2}
NRANKS=$((NNODES*4))
OUTDIR=results_${NRANKS}gpu_${SLURM_JOB_ID}
mkdir -p "${OUTDIR}"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

{
    echo "exe: ${EXE}"
    echo "nodes: ${SLURM_JOB_NODELIST}"
    echo "CRAY_MPICH_VERSION: ${CRAY_MPICH_VERSION}"
    module list 2>&1
    nvidia-smi topo -m
    ldd "${EXE}" | grep -E "cudart|gtl|libmpi"
} > "${OUTDIR}/env_info.txt" 2>&1

# Problem sizes: name and arguments.
SIZES=(
    "small:n_cell=128 max_grid_size=32 ncomp=4 nghost=2 nrounds=200"
    "medium:n_cell=256 max_grid_size=64 ncomp=8 nghost=2 nrounds=100"
    "large:n_cell=512 max_grid_size=128 ncomp=8 nghost=2 nrounds=20"
)

# Plain --gpu-bind=single:1 was dropped: GTL cuIpcOpenMemHandle fails under Slurm GPU
# cgroups and the run hangs. -t 2 caps each step in case another one hangs.
# Variants: label | srun GPU options | per-task shell prefix | AMReX arguments | srun launcher.
# The prefix runs inside each task, so $SLURM_LOCALID is the task's own.
BIND_NONE="--gpus-per-node=4 --gpu-bind=none"
VARIANTS=(
    "amrex_swap|${BIND_NONE}||amrex.use_gpu_aware_mpi=1"
    "amrex_noswap|${BIND_NONE}||amrex.use_gpu_aware_mpi=1 device.perlmutter_swap=0"
    'cvd_reverse|'"${BIND_NONE}"'|export CUDA_VISIBLE_DEVICES=$((3-SLURM_LOCALID));|amrex.use_gpu_aware_mpi=1'
    'cvd_identity|'"${BIND_NONE}"'|export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;|amrex.use_gpu_aware_mpi=1'
    "preinit_reverse|${BIND_NONE}|export FB_PREINIT_DEVICE=reverse;|amrex.use_gpu_aware_mpi=1"
    "preinit_identity|${BIND_NONE}|export FB_PREINIT_DEVICE=identity;|amrex.use_gpu_aware_mpi=1 device.perlmutter_swap=0"
    "host_swap|${BIND_NONE}||amrex.use_gpu_aware_mpi=0"
    'host_cvd_reverse|'"${BIND_NONE}"'|export CUDA_VISIBLE_DEVICES=$((3-SLURM_LOCALID));|amrex.use_gpu_aware_mpi=0'
    # Bindings from the NERSC docs. gpus_per_task drops the job's --gpus-per-node so that
    # --gpus-per-task=1 (implicit per_task:1 binding) doesn't conflict with it.
    "gpus_per_task|--gpus-per-task=1||amrex.use_gpu_aware_mpi=1|env -u SLURM_GPUS_PER_NODE"
    "map_gpu_identity|--gpus-per-node=4 --gpu-bind=map_gpu:0,1,2,3||amrex.use_gpu_aware_mpi=1"
    "map_gpu_reverse|--gpus-per-node=4 --gpu-bind=map_gpu:3,2,1,0||amrex.use_gpu_aware_mpi=1"
    "slurm_single_noipc|--gpus-per-node=4 --gpu-bind=single:1|export MPICH_GPU_IPC_ENABLED=0;|amrex.use_gpu_aware_mpi=1"
)

SUMMARY="${OUTDIR}/summary.txt"
: > "${SUMMARY}"

for rep in $(seq 1 "${REPS}"); do
    for size in "${SIZES[@]}"; do
        sname=${size%%:*}
        sargs=${size#*:}
        for v in "${VARIANTS[@]}"; do
            IFS='|' read -r label gpuopts prefix args launch <<< "${v}"
            log="${OUTDIR}/${label}_${sname}_r${rep}.log"
            echo "running ${label} ${sname} rep ${rep}"
            ${launch} srun -t 2 -N "${NNODES}" -n "${NRANKS}" --ntasks-per-node=4 -c 32 --cpu-bind=cores ${gpuopts} \
                bash -c "${prefix} exec ${EXE} label=${label} ${sargs} ${args}" \
                > "${log}" 2>&1
            ngtl=$(grep -c "not using the same device" "${log}")
            if grep -q '^RESULT' "${log}"; then
                grep '^RESULT' "${log}" | sed "s/\$/ size=${sname} rep=${rep} gtl_warnings=${ngtl}/" >> "${SUMMARY}"
            else
                reason=error
                grep -q "DUE TO TIME LIMIT" "${log}" && reason=timeout
                echo "FAILED label=${label} size=${sname} rep=${rep} reason=${reason} log=${log}" >> "${SUMMARY}"
            fi
        done
    done
done

python3 summarize.py "${SUMMARY}" | tee "${OUTDIR}/table.txt"
