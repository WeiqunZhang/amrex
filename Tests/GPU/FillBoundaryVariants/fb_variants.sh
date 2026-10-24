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

# Variants: label | AMReX arguments. All use GPU-aware MPI except host_mpi.
# default is the default code path; engine is the variant path with nothing switched on.
# On one node every peer is on-node, so the host_off_node variants fake two-rank nodes.
FAKE_NODE=""
[ "${NNODES}" -eq 1 ] && FAKE_NODE="amrex.fb.debug_ranks_per_node=2"
GA="amrex.use_gpu_aware_mpi=1"
ALL_DEVICE="amrex.fb.local_first=1 amrex.fb.async_unpack=1 amrex.fb.persistent=1 amrex.fb.unpack_on_arrival=1"
VARIANTS=(
    "default|${GA}"
    "host_mpi|amrex.use_gpu_aware_mpi=0"
    "engine|${GA} amrex.fb.engine=1"
    "host_off_node|${GA} amrex.fb.host_off_node=1 ${FAKE_NODE}"
    "host_small|${GA} amrex.fb.host_threshold=65536"
    "local_first|${GA} amrex.fb.local_first=1"
    "async_unpack|${GA} amrex.fb.async_unpack=1"
    "persistent|${GA} amrex.fb.persistent=1"
    "unpack_arrival|${GA} amrex.fb.unpack_on_arrival=1"
    "all_device|${GA} ${ALL_DEVICE}"
    "all_host_off|${GA} ${ALL_DEVICE} amrex.fb.host_off_node=1 ${FAKE_NODE}"
)

SUMMARY="${OUTDIR}/summary.txt"
: > "${SUMMARY}"

# -t 2 caps each step in case one hangs.
for rep in $(seq 1 "${REPS}"); do
    for size in "${SIZES[@]}"; do
        sname=${size%%:*}
        sargs=${size#*:}
        for v in "${VARIANTS[@]}"; do
            IFS='|' read -r label args <<< "${v}"
            log="${OUTDIR}/${label}_${sname}_r${rep}.log"
            echo "running ${label} ${sname} rep ${rep}"
            srun -t 2 -N "${NNODES}" -n "${NRANKS}" --ntasks-per-node=4 -c 32 --cpu-bind=cores \
                --gpus-per-node=4 --gpu-bind=none \
                "${EXE}" label="${label}" ${sargs} ${args} > "${log}" 2>&1
            if grep -q '^RESULT' "${log}"; then
                grep '^RESULT' "${log}" | sed "s/\$/ size=${sname} rep=${rep}/" >> "${SUMMARY}"
                grep -q '^CHECK FAILED' "${log}" && echo "check failed: see ${log}"
            else
                reason=error
                grep -q "DUE TO TIME LIMIT" "${log}" && reason=timeout
                echo "FAILED label=${label} size=${sname} rep=${rep} reason=${reason} log=${log}" >> "${SUMMARY}"
            fi
        done
    done
done

python3 summarize.py "${SUMMARY}" | tee "${OUTDIR}/table.txt"
