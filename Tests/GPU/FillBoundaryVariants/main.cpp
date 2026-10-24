#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <string>
#include <vector>

using namespace amrex;

namespace {

// Valid cells get a value unique to (i,j,k,n); ghost cells start at -1.
void init_data (MultiFab& mf)
{
    mf.setVal(-1.0);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const& a = mf.array(mfi);
        amrex::ParallelFor(mfi.validbox(), mf.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            a(i,j,k,n) = Real(i) + Real(j)*1.e3 + Real(k)*1.e6 + Real(n)*0.25;
        });
    }
    Gpu::streamSynchronize();
}

// Max abs difference over valid and ghost cells, reduced over all ranks.
Real max_diff (MultiFab const& a, MultiFab const& b)
{
    MultiFab d(a.boxArray(), a.DistributionMap(), a.nComp(), a.nGrowVect());
    MultiFab::Copy(d, a, 0, 0, a.nComp(), a.nGrowVect());
    MultiFab::Subtract(d, b, 0, 0, a.nComp(), a.nGrowVect());
    Real m = 0.0;
    for (int n = 0; n < d.nComp(); ++n) {
        m = std::max(m, d.norm0(n, d.nGrow()));
    }
    return m;
}

}

int main (int argc, char* argv[])
{
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
        int check = 1;
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
            pp.query("check", check);
        }

        Box domain(IntVect(0), IntVect(n_cell-1));
        Geometry geom(domain, RealBox(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)),
                      CoordSys::cartesian, Array<int,AMREX_SPACEDIM>{AMREX_D_DECL(1,1,1)});
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm(ba);

        // Reference: the default FillBoundary path with every amrex.fb.* variant off.
        MultiFab ref;
        if (check) {
            ref.define(ba, dm, ncomp, nghost);
            init_data(ref);
            auto const saved = FabArrayBase::m_fbv;
            FabArrayBase::m_fbv = FabArrayBase::FBVariants{};
            ref.FillBoundary(geom.periodicity());
            FabArrayBase::m_fbv = saved;
            Gpu::streamSynchronize();
        }

        MultiFab mf(ba, dm, ncomp, nghost);
        init_data(mf);

        for (int i = 0; i < nwarmup; ++i) {
            mf.FillBoundary(geom.periodicity());
        }
        Gpu::streamSynchronize();
        Real maxdiff = check ? max_diff(mf, ref) : -1.0;

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

        if (check) { maxdiff = std::max(maxdiff, max_diff(mf, ref)); }
        std::string check_str = !check ? "skipped" : (maxdiff == 0.0 ? "ok" : "FAIL");
        if (check_str == "FAIL") {
            Print() << "CHECK FAILED: FillBoundary differs from the default path by "
                    << maxdiff << "\n";
        }

        Print() << "RESULT label=" << label
                << " check=" << check_str << " maxdiff=" << maxdiff
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
