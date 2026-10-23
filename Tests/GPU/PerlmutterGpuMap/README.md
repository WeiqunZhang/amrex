# Perlmutter GPU mapping vs. FillBoundary speed

This benchmark times `FillBoundary` under different ways of assigning GPUs to ranks. It
was written to look into the Cray MPICH GTL warning "not using the same device it did
during gtl_init".

Build on Perlmutter with the default modules:

    make -j16

Run from this directory:

    sbatch run_4gpu.sh    # 1 node
    sbatch run_8gpu.sh    # 2 nodes

The results go to `results_<N>gpu_<jobid>/`:
- `table.txt`: median ms per FillBoundary for each variant and size.
- `summary.txt`: one RESULT line per run.
- one log per run, with `MAP` lines showing each rank's device, PCI bus and CPUs.
- `env_info.txt`: modules and `nvidia-smi topo -m`.

`REPS=3 sbatch run_4gpu.sh` changes the number of repetitions (default 2).

| variant | GPU binding | how the device is chosen |
|---|---|---|
| amrex_swap | none | AMReX Perlmutter swap (current default) |
| amrex_noswap | none | `device.perlmutter_swap=0`, so device = local rank |
| cvd_reverse | none | `CUDA_VISIBLE_DEVICES=3-localid` (WarpX template) |
| cvd_identity | none | `CUDA_VISIBLE_DEVICES=localid` |
| preinit_reverse | none | `cudaSetDevice(3-localid)` before `MPI_Init`, then swap |
| preinit_identity | none | `cudaSetDevice(localid)` before `MPI_Init`, swap off |
| host_swap | none | amrex_swap with `amrex.use_gpu_aware_mpi=0` |
| host_cvd_reverse | none | cvd_reverse with `amrex.use_gpu_aware_mpi=0` |
| gpus_per_task | `--gpus-per-task=1` (implicit `per_task:1`) | Slurm (NERSC example) |
| map_gpu_identity | `map_gpu:0,1,2,3` | Slurm (NERSC example) |
| map_gpu_reverse | `map_gpu:3,2,1,0` | Slurm, NUMA-matched order |
| slurm_single_noipc | `single:1` | Slurm, with `MPICH_GPU_IPC_ENABLED=0` (HPE advice under cgroups) |

GTL warnings (`gtl_warn` column) are expected for `amrex_swap` and `amrex_noswap`, which
switch device after `MPI_Init`. `host_swap` switches too, but may not warn, because MPI
never sees GPU buffers in that variant. The `preinit_*` variants keep all GPUs visible
but set the device before `MPI_Init`. They emulate a possible AMReX fix.

Every run also checks its result: the same FillBoundary is done with the default code
path and all `amrex.fb.*` variants off, and the RESULT line reports `check=ok` or
`check=FAIL` with the largest difference over valid and ghost cells (`check=0` skips it).
The table has a `check` column; `FAIL(n)` means n runs differed.

`VARIANT_SET=fb sbatch run_8gpu.sh` times the experimental `amrex.fb.*` FillBoundary
variants instead of the GPU mappings:

| variant | `amrex.fb.*` setting |
|---|---|
| fb_default | none (default path, GPU-aware MPI) |
| fb_host_mpi | none, `amrex.use_gpu_aware_mpi=0` |
| fb_engine | `engine=1`: the variant code path with nothing switched on |
| fb_host_off_node | `host_off_node=1`: pinned host buffers for off-node peers |
| fb_host_small | `host_threshold=65536`: pinned host buffers for messages under 64 KiB |
| fb_local_first | `local_first=1` |
| fb_async_unpack | `async_unpack=1` |
| fb_persistent | `persistent=1` |
| fb_unpack_arrival | `unpack_on_arrival=1` |
| fb_all_device | local_first + async_unpack + persistent + unpack_on_arrival |
| fb_all_host_off | fb_all_device + host_off_node |

On one node (`run_4gpu.sh`) all peers are on-node, so the host_off_node variants run with
`amrex.fb.debug_ranks_per_node=2`, which treats ranks 0-1 and 2-3 as separate nodes.

To check `amrex::Gpu::SelectDeviceBeforeMPIInit`, run the `fix` set (`amrex_init`,
`app_init_select`, `app_init_plain`, `cvd_reverse`):

    VARIANT_SET=fix sbatch run_8gpu.sh

`HAVE_SELECT=FALSE` on the `make` line drops the `app_init_select` code path, for building
against an AMReX without that function.
