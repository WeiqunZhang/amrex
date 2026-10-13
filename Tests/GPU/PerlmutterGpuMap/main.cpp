#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef AMREX_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

using namespace amrex;

namespace {

int local_rank_from_env ()
{
    for (auto const* name : {"SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                             "MPI_LOCALRANKID", "PMI_LOCAL_RANK"}) {
        if (auto const* v = std::getenv(name)) { return std::atoi(v); }
    }
    return -1;
}

// Set the device before MPI_Init, so Cray GTL starts on the device AMReX
// will use. FB_PREINIT_DEVICE=reverse (Perlmutter order) or identity.
int preinit_device ()
{
#ifdef AMREX_USE_CUDA
    auto const* mode = std::getenv("FB_PREINIT_DEVICE");
    int lrank = local_rank_from_env();
    int ndev = 0;
    if (mode == nullptr || lrank < 0 || cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 1) {
        return -1;
    }
    int dev = lrank % ndev;
    if (std::string(mode) == "reverse") { dev = ndev - dev - 1; }
    if (cudaSetDevice(dev) != cudaSuccess) { return -1; }
    return dev;
#else
    return -1;
#endif
}

std::string cpu_list ()
{
    std::string r;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) { return "?"; }
    int i = 0;
    while (i < CPU_SETSIZE) {
        if (CPU_ISSET(i, &set)) {
            int j = i;
            while (j+1 < CPU_SETSIZE && CPU_ISSET(j+1, &set)) { ++j; }
            if (!r.empty()) { r += ","; }
            r += std::to_string(i);
            if (j > i) { r += "-" + std::to_string(j); }
            i = j+1;
        } else {
            ++i;
        }
    }
#endif
    return r.empty() ? std::string("?") : r;
}

void print_mapping (int preinit_dev)
{
    char host[256] = "?";
#ifdef __linux__
    gethostname(host, sizeof(host));
#endif
    auto const* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    std::string pcibus = "n/a";
    int ndev = 0;
#ifdef AMREX_USE_CUDA
    char buf[64];
    if (cudaDeviceGetPCIBusId(buf, sizeof(buf), Gpu::Device::deviceId()) == cudaSuccess) {
        pcibus = buf;
    }
    cudaGetDeviceCount(&ndev);
#endif
    for (int r = 0; r < ParallelDescriptor::NProcs(); ++r) {
        if (r == ParallelDescriptor::MyProc()) {
            AllPrint() << "MAP rank=" << r << " host=" << host
                       << " localid=" << local_rank_from_env()
                       << " cvd=" << (cvd ? cvd : "unset")
                       << " nvisible=" << ndev
                       << " preinit=" << preinit_dev
                       << " device=" << Gpu::Device::deviceId()
                       << " pcibus=" << pcibus
                       << " cpus=" << cpu_list() << "\n";
        }
        ParallelDescriptor::Barrier();
    }
}

}

int main (int argc, char* argv[])
{
    int preinit_dev = preinit_device();

    amrex::Initialize(argc, argv);
    {
        std::string label = "none";
        int n_cell = 256;
        int max_grid_size = 64;
        int ncomp = 8;
        int nghost = 2;
        int nwarmup = 10;
        int nrounds = 100;
        int ntrials = 5;
        {
            ParmParse pp;
            pp.query("label", label);
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("ncomp", ncomp);
            pp.query("nghost", nghost);
            pp.query("nwarmup", nwarmup);
            pp.query("nrounds", nrounds);
            pp.query("ntrials", ntrials);
        }

        print_mapping(preinit_dev);

        Box domain(IntVect(0), IntVect(n_cell-1));
        Geometry geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                      CoordSys::cartesian, Array<int,AMREX_SPACEDIM>{AMREX_D_DECL(1,1,1)});
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm(ba);

        MultiFab mf(ba, dm, ncomp, nghost);
        mf.setVal(1.0);

        for (int i = 0; i < nwarmup; ++i) {
            mf.FillBoundary(geom.periodicity());
        }
        Gpu::streamSynchronize();

        std::vector<double> ms(ntrials);
        for (int t = 0; t < ntrials; ++t) {
            ParallelDescriptor::Barrier();
            double t0 = ParallelDescriptor::second();
            for (int i = 0; i < nrounds; ++i) {
                mf.FillBoundary(geom.periodicity());
            }
            Gpu::streamSynchronize();
            double dt = ParallelDescriptor::second() - t0;
            ParallelDescriptor::ReduceRealMax(dt);
            ms[t] = dt / nrounds * 1.e3;
        }
        std::sort(ms.begin(), ms.end());

        Print() << "RESULT label=" << label
                << " nprocs=" << ParallelDescriptor::NProcs()
                << " n_cell=" << n_cell << " mgs=" << max_grid_size
                << " nboxes=" << ba.size()
                << " ncomp=" << ncomp << " nghost=" << nghost
                << " gpu_aware_mpi=" << ParallelDescriptor::UseGpuAwareMpi()
                << " ms_min=" << ms.front()
                << " ms_med=" << ms[ms.size()/2]
                << " ms_max=" << ms.back() << "\n";
    }
    amrex::Finalize();
}
