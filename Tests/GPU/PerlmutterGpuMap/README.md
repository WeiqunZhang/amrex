# Perlmutter GPU mapping vs. FillBoundary speed

This benchmark times `FillBoundary` under different ways of assigning GPUs to ranks. It
was written to look into the Cray MPICH GTL warning "not using the same device it did
during gtl_init".

Build on Perlmutter (`PrgEnv-gnu`, `cudatoolkit`, `craype-accel-nvidia80`):

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

To check `amrex::Gpu::SelectDeviceBeforeMPIInit`, run the `fix` set (`amrex_init`,
`app_init_select`, `app_init_plain`, `cvd_reverse`):

    VARIANT_SET=fix sbatch run_8gpu.sh

`HAVE_SELECT=FALSE` on the `make` line drops the `app_init_select` code path, for building
against an AMReX without that function.
