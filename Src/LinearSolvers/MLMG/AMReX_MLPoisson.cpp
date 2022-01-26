
#include <AMReX_MLPoisson.H>
#include <AMReX_MLALaplacian.H>

namespace amrex {

MLPoisson::MLPoisson (const Vector<Geometry>& a_geom,
                      const Vector<BoxArray>& a_grids,
                      const Vector<DistributionMapping>& a_dmap,
                      const LPInfo& a_info,
                      const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    define(a_geom, a_grids, a_dmap, a_info, a_factory);
}

MLPoisson::MLPoisson (const Vector<Geometry>& a_geom,
                      const Vector<BoxArray>& a_grids,
                      const Vector<DistributionMapping>& a_dmap,
                      const Vector<iMultiFab const*>& a_overset_mask,
                      const LPInfo& a_info,
                      const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    define(a_geom, a_grids, a_dmap, a_overset_mask, a_info, a_factory);
}

void
MLPoisson::define (const Vector<Geometry>& a_geom,
                   const Vector<BoxArray>& a_grids,
                   const Vector<DistributionMapping>& a_dmap,
                   const LPInfo& a_info,
                   const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    BL_PROFILE("MLPoisson::define()");
    MLCellABecLap::define(a_geom, a_grids, a_dmap, a_info, a_factory);
}

void
MLPoisson::define (const Vector<Geometry>& a_geom,
                   const Vector<BoxArray>& a_grids,
                   const Vector<DistributionMapping>& a_dmap,
                   const Vector<iMultiFab const*>& a_overset_mask,
                   const LPInfo& a_info,
                   const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
    BL_PROFILE("MLPoisson::define(overset)");
    MLCellABecLap::define(a_geom, a_grids, a_dmap, a_overset_mask, a_info, a_factory);
}

MLPoisson::~MLPoisson ()
{}

void
MLPoisson::prepareForSolve ()
{
    BL_PROFILE("MLPoisson::prepareForSolve()");

    MLCellABecLap::prepareForSolve();

    m_is_singular.clear();
    m_is_singular.resize(m_num_amr_levels, false);
    auto itlo = std::find(m_lobc[0].begin(), m_lobc[0].end(), BCType::Dirichlet);
    auto ithi = std::find(m_hibc[0].begin(), m_hibc[0].end(), BCType::Dirichlet);
    if (itlo == m_lobc[0].end() && ithi == m_hibc[0].end())
    {  // No Dirichlet
        for (int alev = 0; alev < m_num_amr_levels; ++alev)
        {
            // For now this assumes that overset regions are treated as Dirichlet bc's
            if (m_domain_covered[alev] && !m_overset_mask[alev][0])
            {
                m_is_singular[alev] = true;
            }
        }
    }
}

void
MLPoisson::Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const
{
    BL_PROFILE("MLPoisson::Fapply()");
    FapplyT(amrlev, mglev, out, in);
}

void
MLPoisson::Fapply_s (int amrlev, int mglev, fMultiFab& out, const fMultiFab& in) const
{
    BL_PROFILE("MLPoisson::Fapply_s()");
    FapplyT(amrlev, mglev, out, in);
}

void
MLPoisson::normalize (int amrlev, int mglev, MultiFab& mf) const
{
    amrex::ignore_unused(amrlev,mglev,mf);
#if (AMREX_SPACEDIM != 3)
    BL_PROFILE("MLPoisson::normalize()");

    if (!m_has_metric_term) return;

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    AMREX_D_TERM(const Real dhx = dxinv[0]*dxinv[0];,
                 const Real dhy = dxinv[1]*dxinv[1];,
                 const Real dhz = dxinv[2]*dxinv[2];);
    const Real dx = m_geom[amrlev][mglev].CellSize(0);
    const Real probxlo = m_geom[amrlev][mglev].ProbLo(0);

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion() && mf.isFusingCandidate()) {
        auto const& ma = mf.arrays();
        ParallelFor(mf,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            mlpoisson_normalize(i,j,k, ma[box_no], AMREX_D_DECL(dhx,dhy,dhz), dx, probxlo);
        });
        Gpu::streamSynchronize();
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto& fab = mf.array(mfi);

#if (AMREX_SPACEDIM == 2)
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlpoisson_normalize(i,j,k, fab, dhx, dhy, dx, probxlo);
            });
#else
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlpoisson_normalize(i,j,k, fab, dhx, dx, probxlo);
            });
#endif
        }
    }
#endif
}

void
MLPoisson::Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const
{
    BL_PROFILE("MLPoisson::Fsmooth()");
    FsmoothT(amrlev, mglev, sol, rhs, redblack, m_undrrelxr[amrlev][mglev]);
}

void
MLPoisson::Fsmooth_s (int amrlev, int mglev, fMultiFab& sol, const fMultiFab& rhs, int redblack) const
{
    BL_PROFILE("MLPoisson::Fsmooth_s()");
    FsmoothT(amrlev, mglev, sol, rhs, redblack, m_undrrelxr_s[amrlev][mglev]);
}

void
MLPoisson::FFlux (int amrlev, const MFIter& mfi,
                  const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                  const FArrayBox& sol, Location, const int face_only) const
{
    AMREX_ASSERT(!hasHiddenDimension());

    BL_PROFILE("MLPoisson::FFlux()");

    const int mglev = 0;
    const Box& box = mfi.tilebox();
    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();

    AMREX_D_TERM(const auto& fxarr = flux[0]->array();,
                 const auto& fyarr = flux[1]->array();,
                 const auto& fzarr = flux[2]->array(););
    const auto& solarr = sol.array();

#if (AMREX_SPACEDIM != 3)
    const Real dx = m_geom[amrlev][mglev].CellSize(0);
    const Real probxlo = m_geom[amrlev][mglev].ProbLo(0);
#endif

#if (AMREX_SPACEDIM == 3)
    if (face_only) {
        Real fac = dxinv[0];
        Box blo = amrex::bdryLo(box, 0);
        int blen = box.length(0);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlpoisson_flux_xface(tbox, fxarr, solarr, fac, blen);
        });
        fac = dxinv[1];
        blo = amrex::bdryLo(box, 1);
        blen = box.length(1);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlpoisson_flux_yface(tbox, fyarr, solarr, fac, blen);
        });
        fac = dxinv[2];
        blo = amrex::bdryLo(box, 2);
        blen = box.length(2);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
        {
            mlpoisson_flux_zface(tbox, fzarr, solarr, fac, blen);
        });
    } else {
        Real fac = dxinv[0];
        Box bflux = amrex::surroundingNodes(box, 0);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlpoisson_flux_x(tbox, fxarr, solarr, fac);
        });
        fac = dxinv[1];
        bflux = amrex::surroundingNodes(box, 1);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlpoisson_flux_y(tbox, fyarr, solarr, fac);
        });
        fac = dxinv[2];
        bflux = amrex::surroundingNodes(box, 2);
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
        {
            mlpoisson_flux_z(tbox, fzarr, solarr, fac);
        });
    }
#elif (AMREX_SPACEDIM == 2)
    if (face_only) {
        Real fac = dxinv[0];
        Box blo = amrex::bdryLo(box, 0);
        int blen = box.length(0);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_xface_m(tbox, fxarr, solarr, fac, blen, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_xface(tbox, fxarr, solarr, fac, blen);
            });
        }
        fac = dxinv[1];
        blo = amrex::bdryLo(box, 1);
        blen = box.length(1);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_yface_m(tbox, fyarr, solarr, fac, blen, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_yface(tbox, fyarr, solarr, fac, blen);
            });
        }
    } else {
        Real fac = dxinv[0];
        Box bflux = amrex::surroundingNodes(box, 0);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_x_m(tbox, fxarr, solarr, fac, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_x(tbox, fxarr, solarr, fac);
            });
        }
        fac = dxinv[1];
        bflux = amrex::surroundingNodes(box, 1);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_y_m(tbox, fyarr, solarr, fac, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_y(tbox, fyarr, solarr, fac);
            });
        }
    }
#else
    if (face_only) {
        Real fac = dxinv[0];
        Box blo = amrex::bdryLo(box, 0);
        int blen = box.length(0);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_xface_m(tbox, fxarr, solarr, fac, blen, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( blo, tbox,
            {
                mlpoisson_flux_xface(tbox, fxarr, solarr, fac, blen);
            });
        }
    } else {
        Real fac = dxinv[0];
        Box bflux = amrex::surroundingNodes(box, 0);
        if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_x_m(tbox, fxarr, solarr, fac, dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( bflux, tbox,
            {
                mlpoisson_flux_x(tbox, fxarr, solarr, fac);
            });
        }
    }
#endif
}

bool
MLPoisson::supportNSolve () const
{
    bool support = true;
    if (m_domain_covered[0]) support = false;
    if (doAgglomeration()) support = false;
    if (AMREX_SPACEDIM != 3) support = false;
    return support;
}

std::unique_ptr<MLLinOp>
MLPoisson::makeNLinOp (int grid_size) const
{
    const Geometry& geom = m_geom[0].back();
    const BoxArray& ba = makeNGrids(grid_size);

    DistributionMapping dm;
    {
        const std::vector<std::vector<int> >& sfc = DistributionMapping::makeSFC(ba);
        Vector<int> pmap(ba.size());
        AMREX_ALWAYS_ASSERT(ParallelContext::CommunicatorSub() == ParallelDescriptor::Communicator());
        const int nprocs = ParallelDescriptor::NProcs();
        for (int iproc = 0; iproc < nprocs; ++iproc) {
            for (int ibox : sfc[iproc]) {
                pmap[ibox] = iproc;
            }
        }
        dm.define(std::move(pmap));
    }

    LPInfo minfo{};
    minfo.has_metric_term = info.has_metric_term;

    std::unique_ptr<MLLinOp> r{new MLALaplacian({geom}, {ba}, {dm}, minfo)};
    auto nop = dynamic_cast<MLALaplacian*>(r.get());
    if (!nop) {
        return nullptr;
    }

    nop->m_parent = this;

    nop->setMaxOrder(maxorder);
    nop->setVerbose(verbose);

    nop->setDomainBC(m_lobc, m_hibc);

    if (needsCoarseDataForBC())
    {
        const Real* dx0 = m_geom[0][0].CellSize();
        const Real fac = Real(0.5)*m_coarse_data_crse_ratio;
        RealVect cbloc {AMREX_D_DECL(dx0[0]*fac, dx0[1]*fac, dx0[2]*fac)};
        nop->setCoarseFineBCLocation(cbloc);
    }

    nop->setScalars(1.0, -1.0);

    const Real* dxinv = geom.InvCellSize();
    Real dxscale = dxinv[0];
#if (AMREX_SPACEDIM >= 2)
    dxscale = std::max(dxscale,dxinv[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    dxscale = std::max(dxscale,dxinv[2]);
#endif

    MultiFab alpha(ba, dm, 1, 0);
    alpha.setVal(Real(1.e30)*dxscale*dxscale);

    MultiFab foo(m_grids[0].back(), m_dmap[0].back(), 1, 0, MFInfo().SetAlloc(false));
    const FabArrayBase::CPC& cpc = alpha.getCPC(IntVect(0),foo,IntVect(0),Periodicity::NonPeriodic());
    alpha.setVal(0.0, cpc, 0, 1);

    nop->setACoeffs(0, alpha);

    return r;
}

void
MLPoisson::copyNSolveSolution (MultiFab& dst, MultiFab const& src) const
{
    dst.ParallelCopy(src);
}

}
