# FillBoundary variants benchmark

This benchmark times `FillBoundary` with the experimental `amrex.fb.*` variants of this
branch, each alone and combined, next to the default code path and host-staged MPI. Every
run also checks its result against the default path.

Build on Perlmutter with the default modules (the GNUmakefile ignores `AMREX_HOME` from
the environment and uses the enclosing checkout):

    make -j16

Run from this directory:

    sbatch run_4gpu.sh    # 1 node
    sbatch run_8gpu.sh    # 2 nodes

The results go to `results_<N>gpu_<jobid>/`:
- `table.txt`: median ms per FillBoundary for each variant and size, plus a `check` column.
- `summary.txt`: one RESULT line per run.
- one log per run.
- `env_info.txt`: modules and `nvidia-smi topo -m`.

`REPS=3 sbatch run_4gpu.sh` changes the number of repetitions (default 2).

| variant | `amrex.fb.*` setting |
|---|---|
| default | none (default path, GPU-aware MPI) |
| host_mpi | none, `amrex.use_gpu_aware_mpi=0` |
| engine | `engine=1`: the variant code path with nothing switched on |
| host_off_node | `host_off_node=1`: pinned host buffers for off-node peers |
| host_small | `host_threshold=65536`: pinned host buffers for messages under 64 KiB |
| local_first | `local_first=1` |
| async_unpack | `async_unpack=1` |
| persistent | `persistent=1` |
| unpack_arrival | `unpack_on_arrival=1` |
| all_device | local_first + async_unpack + persistent + unpack_on_arrival |
| all_host_off | all_device + host_off_node |

On one node (`run_4gpu.sh`) all peers are on-node, so the host_off_node variants run with
`amrex.fb.debug_ranks_per_node=2`, which treats ranks 0-1 and 2-3 as separate nodes. The
2-node job is the one that exercises real off-node traffic.

Each run compares its FillBoundary result, valid and ghost cells, with the default code
path and all `amrex.fb.*` variants off. The RESULT line reports `check=ok` or `check=FAIL`
with the largest difference (`check=0` skips it). In the table, `FAIL(n)` means n runs
differed.

The executable can also be run by hand, for example on a workstation:

    mpiexec -n 4 ./main3d.gnu.MPI.CUDA.ex label=persistent amrex.fb.persistent=1
