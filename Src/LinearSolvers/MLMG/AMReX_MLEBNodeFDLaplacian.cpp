#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLEBNodeFDLap_K.H>
#include <AMReX_MLNodeLinOp_K.H>
#include <AMReX_MLNodeTensorLap_K.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_SingleBoxCGSolver.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EBMultiFabUtil.H>
#endif

namespace amrex {

#ifdef AMREX_USE_EB
MLEBNodeFDLaplacian::MLEBNodeFDLaplacian (
    const Vector<Geometry>& a_geom,
    const Vector<BoxArray>& a_grids,
    const Vector<DistributionMapping>& a_dmap,
    const LPInfo& a_info,
    const Vector<EBFArrayBoxFactory const*>& a_factory)
{
    define(a_geom, a_grids, a_dmap, a_info, a_factory);
}
#endif

MLEBNodeFDLaplacian::MLEBNodeFDLaplacian (
    const Vector<Geometry>& a_geom,
    const Vector<BoxArray>& a_grids,
    const Vector<DistributionMapping>& a_dmap,
    const LPInfo& a_info)
{
    define(a_geom, a_grids, a_dmap, a_info);
}

void
MLEBNodeFDLaplacian::setSigma (Array<Real,AMREX_SPACEDIM> const& a_sigma) noexcept
{
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        m_sigma[i] = a_sigma[i];
    }
}

void
MLEBNodeFDLaplacian::setSigma (int amrlev, MultiFab const& a_sigma)
{
    m_needs_update = true;
    m_has_sigma_mf = true;
    m_sigma_mf[amrlev][0] = std::make_unique<MultiFab>
        (this->m_grids[amrlev][0], this->m_dmap[amrlev][0], 1, 1, MFInfo{},
         *(this->m_factory[amrlev][0]));
    MultiFab::Copy(*m_sigma_mf[amrlev][0], a_sigma, 0, 0, 1, 0);
#ifdef AMREX_USE_EB
    amrex::EB_set_covered(*m_sigma_mf[amrlev][0], Real(0.0));
#endif
}

void
MLEBNodeFDLaplacian::setRZ (bool flag) // NOLINT
{
#if (AMREX_SPACEDIM == 2)
    m_rz = flag;
#else
    amrex::ignore_unused(flag, m_rz);
#endif
}

void
MLEBNodeFDLaplacian::setAlpha (Real a_alpha) // NOLINT
{
#if (AMREX_SPACEDIM == 2)
    m_rz_alpha = a_alpha;
#else
    amrex::ignore_unused(a_alpha);
#endif
}

#ifdef AMREX_USE_EB

void
MLEBNodeFDLaplacian::setEBDirichlet (Real a_phi_eb)
{
    m_s_phi_eb = a_phi_eb;
}

void
MLEBNodeFDLaplacian::define (const Vector<Geometry>& a_geom,
                             const Vector<BoxArray>& a_grids,
                             const Vector<DistributionMapping>& a_dmap,
                             const LPInfo& a_info,
                             const Vector<EBFArrayBoxFactory const*>& a_factory)
{
    static_assert(AMREX_SPACEDIM > 1, "MLEBNodeFDLaplacian: 1D not supported");

    BL_PROFILE("MLEBNodeFDLaplacian::define()");

    // This makes sure grids are cell-centered;
    Vector<BoxArray> cc_grids = a_grids;
    for (auto& ba : cc_grids) {
        ba.enclosedCells();
    }

    if (a_grids.size() > 1) {
        amrex::Abort("MLEBNodeFDLaplacian: multi-level not supported");
    }

    Vector<FabFactory<FArrayBox> const*> _factory;
    for (const auto *x : a_factory) {
        _factory.push_back(static_cast<FabFactory<FArrayBox> const*>(x));
    }

    int eb_limit_coarsening = true;
    m_coarsening_strategy = CoarseningStrategy::Sigma; // This will fill nodes outside Neumann BC
    MLNodeLinOp::define(a_geom, cc_grids, a_dmap, a_info, _factory, eb_limit_coarsening);

    m_sigma_mf.resize(this->m_num_amr_levels);
    for (int ilev = 0; ilev < this->m_num_amr_levels; ++ilev) {
        m_sigma_mf[ilev].resize(this->m_num_mg_levels[ilev]);
    }
}

#endif

void
MLEBNodeFDLaplacian::define (const Vector<Geometry>& a_geom,
                             const Vector<BoxArray>& a_grids,
                             const Vector<DistributionMapping>& a_dmap,
                             const LPInfo& a_info)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM>1, "MLEBNodeFDLaplacian: 1D not supported");

    BL_PROFILE("MLEBNodeFDLaplacian::define()");

    // This makes sure grids are cell-centered;
    Vector<BoxArray> cc_grids = a_grids;
    for (auto& ba : cc_grids) {
        ba.enclosedCells();
    }

    if (a_grids.size() > 1) {
        amrex::Abort("MLEBNodeFDLaplacian: multi-level not supported");
    }

    m_coarsening_strategy = CoarseningStrategy::Sigma; // This will fill nodes outside Neumann BC
    MLNodeLinOp::define(a_geom, cc_grids, a_dmap, a_info);

    m_sigma_mf.resize(this->m_num_amr_levels);
    for (int ilev = 0; ilev < this->m_num_amr_levels; ++ilev) {
        m_sigma_mf[ilev].resize(this->m_num_mg_levels[ilev]);
    }
}

#ifdef AMREX_USE_EB
std::unique_ptr<FabFactory<FArrayBox> >
MLEBNodeFDLaplacian::makeFactory (int amrlev, int mglev) const
{
    if (EB2::TopIndexSpaceIfPresent()) {
        auto const* fact0 = Factory(amrlev,0) ? Factory(amrlev, 0) : Factory(0,0);
        return makeEBFabFactory(static_cast<EBFArrayBoxFactory const*>(fact0)->getEBIndexSpace(),
                                m_geom[amrlev][mglev],
                                m_grids[amrlev][mglev],
                                m_dmap[amrlev][mglev],
                                {1,1,1}, EBSupport::full);
    } else {
        return MLNodeLinOp::makeFactory(amrlev, mglev);
    }
}
#endif

void
MLEBNodeFDLaplacian::restriction (int amrlev, int cmglev, MultiFab& crse, MultiFab& fine) const
{
    BL_PROFILE("MLEBNodeFDLaplacian::restriction()");

    applyBC(amrlev, cmglev-1, fine, BCMode::Homogeneous, StateMode::Solution);

    IntVect const ratio = (amrlev > 0) ? IntVect(2) : mg_coarsen_ratio_vec[cmglev-1];
    int semicoarsening_dir = info.semicoarsening_direction;

    bool need_parallel_copy = !amrex::isMFIterSafe(crse, fine);
    MultiFab cfine;
    if (need_parallel_copy) {
        const BoxArray& ba = amrex::coarsen(fine.boxArray(), ratio);
        cfine.define(ba, fine.DistributionMap(), 1, 0);
    }

    MultiFab* pcrse = (need_parallel_copy) ? &cfine : &crse;
    const iMultiFab& dmsk = *m_dirichlet_mask[amrlev][cmglev-1];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*pcrse, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real> cfab = pcrse->array(mfi);
        Array4<Real const> const& ffab = fine.const_array(mfi);
        Array4<int const> const& mfab = dmsk.const_array(mfi);
        if (ratio == 2) {
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlndlap_restriction(i,j,k,cfab,ffab,mfab);
            });
        } else {
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlndlap_semi_restriction(i,j,k,cfab,ffab,mfab, semicoarsening_dir);
            });
        }
    }

    if (need_parallel_copy) {
        crse.ParallelCopy(cfine);
    }
}

void
MLEBNodeFDLaplacian::interpolation (int amrlev, int fmglev, MultiFab& fine,
                                    const MultiFab& crse) const
{
    BL_PROFILE("MLEBNodeFDLaplacian::interpolation()");

    IntVect const ratio = (amrlev > 0) ? IntVect(2) : mg_coarsen_ratio_vec[fmglev];
    int semicoarsening_dir = info.semicoarsening_direction;

    bool need_parallel_copy = !amrex::isMFIterSafe(crse, fine);
    MultiFab cfine;
    const MultiFab* cmf = &crse;
    if (need_parallel_copy) {
        const BoxArray& ba = amrex::coarsen(fine.boxArray(), ratio);
        cfine.define(ba, fine.DistributionMap(), 1, 0);
        cfine.ParallelCopy(crse);
        cmf = &cfine;
    }

    const iMultiFab& dmsk = *m_dirichlet_mask[amrlev][fmglev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fine, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        Array4<Real> const& ffab = fine.array(mfi);
        Array4<Real const> const& cfab = cmf->const_array(mfi);
        Array4<int const> const& mfab = dmsk.const_array(mfi);
        if (ratio == 2) {
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlndtslap_interpadd(i,j,k,ffab,cfab,mfab);
            });
        } else {
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlndtslap_semi_interpadd(i,j,k,ffab,cfab,mfab,semicoarsening_dir);
            });
        }
    }
}

void
MLEBNodeFDLaplacian::prepareForSolve ()
{
    BL_PROFILE("MLEBNodeFDLaplacian::prepareForSolve()");

    MLNodeLinOp::prepareForSolve();

    buildMasks();

#ifdef AMREX_USE_EB
    // Set covered nodes to Dirichlet, but with a negative value.
    // compGrad relies on the negative value to detect EB.
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
            const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
            if (factory && !factory->isAllRegular()) {
                auto const& levset_mf = factory->getLevelSet();
                auto const& levset_ar = levset_mf.const_arrays();
                auto& dmask_mf = *m_dirichlet_mask[amrlev][mglev];
                auto const& dmask_ar = dmask_mf.arrays();
                amrex::ParallelFor(dmask_mf,
                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                {
                    if (levset_ar[box_no](i,j,k) >= Real(0.0)) {
                        dmask_ar[box_no](i,j,k) = -1;
                    }
                });
            }
        }
    }
#endif

    {
        int amrlev = 0;
        int mglev = m_num_mg_levels[amrlev]-1;
        auto const& dotmasks = m_bottom_dot_mask.arrays();
        auto const& dirmasks = m_dirichlet_mask[amrlev][mglev]->const_arrays();
        amrex::ParallelFor(m_bottom_dot_mask,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            if (dirmasks[box_no](i,j,k)) {
                dotmasks[box_no](i,j,k) = Real(0.);
            }
        });
    }

    AMREX_ASSERT(!isBottomSingular());

    Gpu::streamSynchronize();

#if (AMREX_SPACEDIM == 2)
    if (m_rz) {
        if (m_geom[0][0].ProbLo(0) == 0._rt) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_lobc[0][0] == BCType::Neumann,
                                             "The lo-x BC must be Neumann for 2d RZ");
        }
        if (m_sigma[0] == 0._rt) {
            m_sigma[0] = 1._rt; // For backward compatibility
        }
    }
#endif

    if (m_has_sigma_mf) {
        update_sigma();
    }
}

#ifdef AMREX_USE_EB
bool
MLEBNodeFDLaplacian::scaleRHS (int amrlev, MultiFab* rhs) const
{
    const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());

    if (!factory) {return false; }

    if (rhs) {
        auto const& dmask = *m_dirichlet_mask[amrlev][0];
        auto const& edgecent = factory->getEdgeCent();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*rhs,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.tilebox();
            Array4<Real> const& rhsarr = rhs->array(mfi);
            Array4<int const> const& dmarr = dmask.const_array(mfi);
            bool cutfab = edgecent[0]->ok(mfi);
            if (cutfab) {
                AMREX_D_TERM(Array4<Real const> const& ecx = edgecent[0]->const_array(mfi);,
                             Array4<Real const> const& ecy = edgecent[1]->const_array(mfi);,
                             Array4<Real const> const& ecz = edgecent[2]->const_array(mfi));
                AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                {
                    mlebndfdlap_scale_rhs(i,j,k,rhsarr,dmarr,AMREX_D_DECL(ecx,ecy,ecz));
                });
            }
        }
    }

    return true;
}
#endif

void
MLEBNodeFDLaplacian::Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const
{
    BL_PROFILE("MLEBNodeFDLaplacian::Fapply()");

    const auto dxinv = m_geom[amrlev][mglev].InvCellSizeArray();
#if (AMREX_SPACEDIM == 2)
    const auto sig0 = m_sigma[0];
    const auto dx0 = m_geom[amrlev][mglev].CellSize(0);
    const auto dx1 = m_geom[amrlev][mglev].CellSize(1)/std::sqrt(m_sigma[1]);
    const auto xlo = m_geom[amrlev][mglev].ProbLo(0);
    const auto alpha = m_rz_alpha;
#endif
    AMREX_D_TERM(const Real bx = m_sigma[0]*dxinv[0]*dxinv[0];,
                 const Real by = m_sigma[1]*dxinv[1]*dxinv[1];,
                 const Real bz = m_sigma[2]*dxinv[2]*dxinv[2];)

    auto const& dmask = *m_dirichlet_mask[amrlev][mglev];

#ifdef AMREX_USE_EB
    const auto phieb = (m_in_solution_mode && !this->m_precond_mode) ? m_s_phi_eb : Real(0.0);
    const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    Array<const MultiCutFab*,AMREX_SPACEDIM> edgecent {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    if (factory) {
        edgecent = factory->getEdgeCent();
    }
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.tilebox();
        Array4<Real const> const& xarr = in.const_array(mfi);
        Array4<Real> const& yarr = out.array(mfi);
        Array4<int const> const& dmarr = dmask.const_array(mfi);
#ifdef AMREX_USE_EB
        bool cutfab = edgecent[0] && edgecent[0]->ok(mfi);
        if (cutfab && factory) { // clang-tidy is not that smart
            AMREX_D_TERM(Array4<Real const> const& ecx = edgecent[0]->const_array(mfi);,
                         Array4<Real const> const& ecy = edgecent[1]->const_array(mfi);,
                         Array4<Real const> const& ecz = edgecent[2]->const_array(mfi));
            auto const& levset = factory->getLevelSet().const_array(mfi);
            if (phieb == std::numeric_limits<Real>::lowest()) {
                auto const& phiebarr = m_phi_eb[amrlev].const_array(mfi);
#if (AMREX_SPACEDIM == 2)
                if (m_rz) {
                    if (m_has_sigma_mf) {
                        auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                        auto const& vfrc = factory->getVolFrac().const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_sig_adotx_rz_eb(i,j,k,yarr,xarr,levset,dmarr,ecx,ecy,
                                                        sigarr, vfrc, phiebarr, dx0, dx1, xlo, alpha);
                        });
                    } else {
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_adotx_rz_eb(i,j,k,yarr,xarr,levset,dmarr,ecx,ecy,
                                                    phiebarr, sig0, dx0, dx1, xlo, alpha);
                        });
                    }
                } else
#endif
                if (m_has_sigma_mf) {
                    auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                    auto const& vfrc = factory->getVolFrac().const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_sig_adotx_eb(i,j,k,yarr,xarr,levset,dmarr,AMREX_D_DECL(ecx,ecy,ecz),
                                                 sigarr, vfrc, phiebarr, AMREX_D_DECL(bx,by,bz));
                    });
                } else {
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_adotx_eb(i,j,k,yarr,xarr,levset,dmarr,AMREX_D_DECL(ecx,ecy,ecz),
                                             phiebarr, AMREX_D_DECL(bx,by,bz));
                    });
                }
            } else {
#if (AMREX_SPACEDIM == 2)
                if (m_rz) {
                    if (m_has_sigma_mf) {
                        auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                        auto const& vfrc = factory->getVolFrac().const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_sig_adotx_rz_eb(i,j,k,yarr,xarr,levset,dmarr,ecx,ecy,
                                                        sigarr, vfrc, phieb, dx0, dx1, xlo, alpha);
                        });
                    } else {
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_adotx_rz_eb(i,j,k,yarr,xarr,levset,dmarr,ecx,ecy,
                                                    phieb, sig0, dx0, dx1, xlo, alpha);
                        });
                    }
                } else
#endif
                if (m_has_sigma_mf) {
                    auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                    auto const& vfrc = factory->getVolFrac().const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_sig_adotx_eb(i,j,k,yarr,xarr,levset,dmarr,AMREX_D_DECL(ecx,ecy,ecz),
                                                 sigarr, vfrc, phieb, AMREX_D_DECL(bx,by,bz));
                    });
                } else {
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_adotx_eb(i,j,k,yarr,xarr,levset,dmarr,AMREX_D_DECL(ecx,ecy,ecz),
                                             phieb, AMREX_D_DECL(bx,by,bz));
                    });
                }
            }
        } else
#endif // AMREX_USE_EB
        {
#if (AMREX_SPACEDIM == 2)
            if (m_rz) {
                if (m_has_sigma_mf) {
                    auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_sig_adotx_rz(i,j,k,yarr,xarr,dmarr,sigarr,dx0,dx1,xlo,alpha);
                    });
                } else {
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_adotx_rz(i,j,k,yarr,xarr,dmarr,sig0,dx0,dx1,xlo,alpha);
                    });
                }
            } else
#endif
            if (m_has_sigma_mf) {
                auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                {
                    mlebndfdlap_sig_adotx(i,j,k,yarr,xarr,dmarr,sigarr,AMREX_D_DECL(bx,by,bz));
                });
            } else {
                AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                {
                    mlebndfdlap_adotx(i,j,k,yarr,xarr,dmarr,AMREX_D_DECL(bx,by,bz));
                });
            }
        }
    }
}

void
MLEBNodeFDLaplacian::Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs) const
{
    BL_PROFILE("MLEBNodeFDLaplacian::Fsmooth()");

    const auto dxinv = m_geom[amrlev][mglev].InvCellSizeArray();
#if (AMREX_SPACEDIM == 2)
    const auto sig0 = m_sigma[0];
    const auto dx0 = m_geom[amrlev][mglev].CellSize(0);
    const auto dx1 = m_geom[amrlev][mglev].CellSize(1)/std::sqrt(m_sigma[1]);
    const auto xlo = m_geom[amrlev][mglev].ProbLo(0);
    const auto alpha = m_rz_alpha;
#endif
    AMREX_D_TERM(const Real bx = m_sigma[0]*dxinv[0]*dxinv[0];,
                 const Real by = m_sigma[1]*dxinv[1]*dxinv[1];,
                 const Real bz = m_sigma[2]*dxinv[2]*dxinv[2];)

    auto const& dmask = *m_dirichlet_mask[amrlev][mglev];

    for (int redblack = 0; redblack < 2; ++redblack) {
        if (redblack > 0) {
            applyBC(amrlev, mglev, sol, BCMode::Homogeneous, StateMode::Correction);
        }

#ifdef AMREX_USE_EB
        const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
        Array<const MultiCutFab*,AMREX_SPACEDIM> edgecent {AMREX_D_DECL(nullptr,nullptr,nullptr)};
        if (factory) {
            edgecent = factory->getEdgeCent();
        }
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(sol,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.tilebox();
            Array4<Real> const& solarr = sol.array(mfi);
            Array4<Real const> const& rhsarr = rhs.const_array(mfi);
            Array4<int const> const& dmskarr = dmask.const_array(mfi);
#ifdef AMREX_USE_EB
            bool cutfab = edgecent[0] && edgecent[0]->ok(mfi);
            if (cutfab && factory) { // clang-tidy is not that smart
                AMREX_D_TERM(Array4<Real const> const& ecx = edgecent[0]->const_array(mfi);,
                             Array4<Real const> const& ecy = edgecent[1]->const_array(mfi);,
                             Array4<Real const> const& ecz = edgecent[2]->const_array(mfi));
                auto const& levset = factory->getLevelSet().const_array(mfi);
#if (AMREX_SPACEDIM == 2)
                if (m_rz) {
                    if (m_has_sigma_mf) {
                        auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                        auto const& vfrc = factory->getVolFrac().const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_sig_gsrb_rz_eb(i,j,k,solarr,rhsarr,levset,dmskarr,ecx,ecy,
                                                       sigarr, vfrc, dx0, dx1, xlo, redblack, alpha);
                        });
                    } else {
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_gsrb_rz_eb(i,j,k,solarr,rhsarr,levset,dmskarr,ecx,ecy,
                                                   sig0, dx0, dx1, xlo, redblack, alpha);
                        });
                    }
                } else
#endif
                if (m_has_sigma_mf) {
                    auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                    auto const& vfrc = factory->getVolFrac().const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_sig_gsrb_eb(i,j,k,solarr,rhsarr,levset,dmskarr,AMREX_D_DECL(ecx,ecy,ecz),
                                                sigarr, vfrc, AMREX_D_DECL(bx,by,bz), redblack);
                    });
                } else {
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_gsrb_eb(i,j,k,solarr,rhsarr,levset,dmskarr,AMREX_D_DECL(ecx,ecy,ecz),
                                            AMREX_D_DECL(bx,by,bz), redblack);
                    });
                }
            } else
#endif // AMREX_USE_EB
            {
#if (AMREX_SPACEDIM == 2)
                if (m_rz) {
                    if (m_has_sigma_mf) {
                        auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_sig_gsrb_rz(i,j,k,solarr,rhsarr,dmskarr,sigarr,
                                                    dx0, dx1, xlo, redblack, alpha);
                        });
                    } else {
                        AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                        {
                            mlebndfdlap_gsrb_rz(i,j,k,solarr,rhsarr,dmskarr,
                                                sig0, dx0, dx1, xlo, redblack, alpha);
                        });
                    }
                } else
#endif
                if (m_has_sigma_mf) {
                    auto const& sigarr = m_sigma_mf[amrlev][mglev]->const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_sig_gsrb(i,j,k,solarr,rhsarr,dmskarr,sigarr,
                                             AMREX_D_DECL(bx,by,bz), redblack);
                    });
                } else {
                    AMREX_HOST_DEVICE_FOR_3D(box, i, j, k,
                    {
                        mlebndfdlap_gsrb(i,j,k,solarr,rhsarr,dmskarr,
                                         AMREX_D_DECL(bx,by,bz), redblack);
                    });
                }
            }
        }
    }

    nodalSync(amrlev, mglev, sol);
}

void
MLEBNodeFDLaplacian::normalize (int amrlev, int mglev, MultiFab& mf) const
{
    amrex::ignore_unused(amrlev, mglev, mf);
}

void
MLEBNodeFDLaplacian::fixUpResidualMask (int /*amrlev*/, iMultiFab& /*resmsk*/)
{
    amrex::Abort("MLEBNodeFDLaplacian::fixUpResidualMask: TODO");
}

void
MLEBNodeFDLaplacian::compGrad (int amrlev, const Array<MultiFab*,AMREX_SPACEDIM>& grad,
                               MultiFab& sol, Location /*loc*/) const
{
    BL_PROFILE("MLEBNodeFDLaplacian::compGrad()");

    AMREX_ASSERT(AMREX_D_TERM(grad[0]->ixType() == IndexType(IntVect(AMREX_D_DECL(0,1,1))),
                           && grad[1]->ixType() == IndexType(IntVect(AMREX_D_DECL(1,0,1))),
                           && grad[2]->ixType() == IndexType(IntVect(AMREX_D_DECL(1,1,0)))));
    const int mglev = 0;
    AMREX_D_TERM(const auto dxi = m_geom[amrlev][mglev].InvCellSize(0);,
                 const auto dyi = m_geom[amrlev][mglev].InvCellSize(1);,
                 const auto dzi = m_geom[amrlev][mglev].InvCellSize(2);)

#ifdef AMREX_USE_EB
    auto const& dmask = *m_dirichlet_mask[amrlev][mglev];
    const auto phieb = m_s_phi_eb;
    const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    Array<const MultiCutFab*,AMREX_SPACEDIM> edgecent {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    if (factory) {
        edgecent = factory->getEdgeCent();
    }
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*grad[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        AMREX_D_TERM(const Box& xbox = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,1)));,
                     const Box& ybox = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,1)));,
                     const Box& zbox = mfi.tilebox(IntVect(AMREX_D_DECL(1,1,0)));)
        Array4<Real const> const& p = sol.const_array(mfi);
        AMREX_D_TERM(Array4<Real> const& gpx = grad[0]->array(mfi);,
                     Array4<Real> const& gpy = grad[1]->array(mfi);,
                     Array4<Real> const& gpz = grad[2]->array(mfi);)
#ifdef AMREX_USE_EB
        if (factory && !factory->isAllRegular()) {
            Array4<int const> const& dmarr = dmask.const_array(mfi);
            bool cutfab = edgecent[0] && edgecent[0]->ok(mfi);
            AMREX_D_TERM(Array4<Real const> const& ecx
                             = cutfab ? edgecent[0]->const_array(mfi) : Array4<Real const>{};,
                         Array4<Real const> const& ecy
                             = cutfab ? edgecent[1]->const_array(mfi) : Array4<Real const>{};,
                         Array4<Real const> const& ecz
                             = cutfab ? edgecent[2]->const_array(mfi) : Array4<Real const>{};)
            if (phieb == std::numeric_limits<Real>::lowest()) {
                auto const& phiebarr = m_phi_eb[amrlev].const_array(mfi);
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM(
                    xbox, txbox,
                    {
                        mlebndfdlap_grad_x(txbox, gpx, p, dmarr, ecx, phiebarr, dxi);
                    }
                    , ybox, tybox,
                    {
                        mlebndfdlap_grad_y(tybox, gpy, p, dmarr, ecy, phiebarr, dyi);
                    }
                    , zbox, tzbox,
                    {
                        mlebndfdlap_grad_z(tzbox, gpz, p, dmarr, ecz, phiebarr, dzi);
                    });
            } else {
                AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM(
                    xbox, txbox,
                    {
                        mlebndfdlap_grad_x(txbox, gpx, p, dmarr, ecx, phieb, dxi);
                    }
                    , ybox, tybox,
                    {
                        mlebndfdlap_grad_y(tybox, gpy, p, dmarr, ecy, phieb, dyi);
                    }
                    , zbox, tzbox,
                    {
                        mlebndfdlap_grad_z(tzbox, gpz, p, dmarr, ecz, phieb, dzi);
                    });
            }
        } else
#endif
        {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA_DIM(
                    xbox, txbox,
                    {
                        mlebndfdlap_grad_x(txbox, gpx, p, dxi);
                    }
                    , ybox, tybox,
                    {
                        mlebndfdlap_grad_y(tybox, gpy, p, dyi);
                    }
                    , zbox, tzbox,
                    {
                        mlebndfdlap_grad_z(tzbox, gpz, p, dzi);
                    });
        }
    }
}

#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
void
MLEBNodeFDLaplacian::fillIJMatrix (MFIter const& /*mfi*/,
                                   Array4<HypreNodeLap::AtomicInt const> const& /*gid*/,
                                   Array4<int const> const& /*lid*/,
                                   HypreNodeLap::Int* /*ncols*/,
                                   HypreNodeLap::Int* /*cols*/,
                                   Real* /*mat*/) const
{
    amrex::Abort("MLEBNodeFDLaplacian::fillIJMatrix: todo");
}

void
MLEBNodeFDLaplacian::fillRHS (MFIter const& /*mfi*/, Array4<int const> const& /*lid*/,
                              Real* /*rhs*/, Array4<Real const> const& /*bfab*/) const
{
    amrex::Abort("MLEBNodeFDLaplacian::fillRHS: todo");
}
#endif

void
MLEBNodeFDLaplacian::postSolve (Vector<MultiFab*> const& sol) const
{
#ifdef AMREX_USE_EB
    if (this->m_precond_mode) { return; }
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        const auto phieb = m_s_phi_eb;
        const auto *factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][0].get());
        if (!factory || factory->isAllRegular()) { return; }
        auto const& levset_mf = factory->getLevelSet();
        auto const& levset_ar = levset_mf.const_arrays();
        MultiFab& mf = *sol[amrlev];
        auto const& sol_ar = mf.arrays();
        if (phieb == std::numeric_limits<Real>::lowest()) {
            auto const& phieb_ar = m_phi_eb[amrlev].const_arrays();
            amrex::ParallelFor(mf, IntVect(1),
            [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
            {
                if (levset_ar[bi](i,j,k) >= Real(0.0)) {
                    sol_ar[bi](i,j,k) = phieb_ar[bi](i,j,k);
                }
            });
        } else {
            amrex::ParallelFor(mf, IntVect(1),
            [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
            {
                if (levset_ar[bi](i,j,k) >= Real(0.0)) {
                    sol_ar[bi](i,j,k) = phieb;
                }
            });
        }
    }
#else
    amrex::ignore_unused(sol);
#endif
}

void
MLEBNodeFDLaplacian::update ()
{
    if (MLNodeLinOp::needsUpdate()) {
        MLNodeLinOp::update();
    }

    if (m_needs_update && m_has_sigma_mf) {
        update_sigma();
    }
    m_needs_update = false;
}

void
MLEBNodeFDLaplacian::update_sigma ()
{
    AMREX_D_TERM(m_sigma[0] = Real(1.0);,
                 m_sigma[1] = Real(1.0);,
                 m_sigma[2] = Real(1.0));
    AMREX_ALWAYS_ASSERT(this->m_num_amr_levels == 1);
    for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
        for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
            if (m_sigma_mf[amrlev][mglev] == nullptr) {
                m_sigma_mf[amrlev][mglev] = std::make_unique<MultiFab>
                    (this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1,
                     MFInfo{}, *(this->m_factory[amrlev][mglev]));
            }
            IntVect const ratio = (amrlev > 0) ? IntVect (2)
                : this->mg_coarsen_ratio_vec[mglev-1];
#ifdef AMREX_USE_EB
            amrex::EB_average_down
#else
            amrex::average_down
#endif
                (*m_sigma_mf[amrlev][mglev-1],
                 *m_sigma_mf[amrlev][mglev], 0, 1, ratio);
        }

        for (int mglev = 0; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
            auto const& geom = this->m_geom[amrlev][mglev];
            auto& sigma = *m_sigma_mf[amrlev][mglev];
            sigma.FillBoundary(geom.periodicity());

            const Box& domain = geom.Domain();
            const auto lobc = LoBC();
            const auto hibc = HiBC();

            MFItInfo mfi_info;
            if (Gpu::notInLaunchRegion()) { mfi_info.SetDynamic(true); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(sigma, mfi_info); mfi.isValid(); ++mfi)
            {
                Array4<Real> const& sfab = sigma.array(mfi);
                mlndlap_fillbc_cc<Real>(mfi.validbox(),sfab,domain,lobc,hibc);
            }
        }
    }
}

namespace {
    struct LPBase
    {
        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real xdoty (IntVect const& iv, int, Real vx, Real vy) const
        {
            return dotmsk(iv)*vx*vy;
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void normalize (IntVect const&, int, Real&) const {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void getNeighbors (IntVect const& iv, int idim,
                           IntVect& ivm, IntVect& ivp) const
        {
            ivm = iv;
            ivp = iv;

            // There is only a single Box. Thus a box boundary node is
            // either Dirichlet (including domain Dirichlet boundary or
            // coarse/fine Dirichlet boundary) or on the periodic or Neumann
            // domain boundary. Because this is called on non-Dirichlet
            // nodes only, if a boundary node (e.g., iv[0] == dlo[0]) gets
            // here, it's either periodic or Neumann.

            if (iv[idim] == dlo[idim]) {
                ivm[idim] = is_periodic[idim] ? dhi[idim]-1 : iv[idim]+1;
            } else {
                --ivm[idim];
            }

            if (iv[idim] == dhi[idim]) {
                ivp[idim] = is_periodic[idim] ? dlo[idim]+1 : iv[idim]-1;
            } else {
                ++ivp[idim];
            }
        }

        Array4<Real const> dotmsk;
        Array4<int const> dirmsk;
        GpuArray<Real,AMREX_SPACEDIM> beta;
        IntVect dlo, dhi;
        GpuArray<bool,AMREX_SPACEDIM> is_periodic;
    };

    struct LP
        : public LPBase
    {
        LP (LPBase const& a_lpbase)
            : LPBase(a_lpbase)
            {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real apply (IntVect const& iv, int, Array4<Real> const& xa, int n) const
        {
            Real y = 0;
            if (!dirmsk(iv)) {
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    IntVect ivm, ivp;
                    getNeighbors(iv, idim, ivm, ivp);

                    y += beta[idim] *
                        (xa(ivm,n) + xa(ivp,n) - Real(2.0)*xa(iv,n));
                }
            }
            return y;
        }
    };

    struct LPSigma
        : public LPBase
    {
        LPSigma (LPBase const& a_lpbase, Array4<Real const> const& a_sigma)
            : LPBase(a_lpbase), sigma(a_sigma)
            {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real apply (IntVect const& iv, int, Array4<Real> const& xa, int n) const
        {
            Real y = 0;
            if (!dirmsk(iv)) {
                constexpr int nsig = 1 << (AMREX_SPACEDIM-1);
                constexpr Real fac = Real(1.0) / Real(nsig);
                Real const xcenter = xa(iv,n);

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    IntVect ivm, ivp;
                    getNeighbors(iv, idim, ivm, ivp);

                    Real sigm = 0;
                    Real sigp = 0;
                    for (int isig = 0; isig < nsig; ++isig) {
                        IntVect ism = iv;
                        IntVect isp = iv;
                        --ism[idim];

                        int ibit = 0;
                        for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                            if (jdim != idim) {
                                int const offset = ((isig >> ibit) & 1) - 1;
                                ism[jdim] += offset;
                                isp[jdim] += offset;
                                ++ibit;
                            }
                        }

                        sigm += sigma(ism);
                        sigp += sigma(isp);
                    }

                    y += beta[idim] * fac *
                        (sigm*(xa(ivm,n)-xcenter) + sigp*(xa(ivp,n)-xcenter));
                }
            }
            return y;
        }

        Array4<Real const> sigma;
    };

#if (AMREX_SPACEDIM == 2)
    struct LPRZ
        : public LPBase
    {
        LPRZ (LPBase const& a_lpbase, Real a_sigr, Real a_dr, Real a_dz,
              Real a_rlo, Real a_alpha, Array4<Real const> const& a_sigma,
              Array4<Real const> const& a_vfrac, Array4<Real const> const& a_levset,
              GpuArray<Array4<Real const>,AMREX_SPACEDIM> const& a_edgecent,
              bool a_has_sigma, bool a_use_eb)
            : LPBase(a_lpbase), sigr(a_sigr), dr(a_dr), dz(a_dz), rlo(a_rlo),
              alpha(a_alpha), sigma(a_sigma), vfrac(a_vfrac), levset(a_levset),
              edgecent(a_edgecent), has_sigma(a_has_sigma), use_eb(a_use_eb)
            {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real radialEdgeSigma (IntVect const& edge) const
        {
            if (!has_sigma) {
                return sigr;
            }

            IntVect cellm = edge;
            --cellm[1];
            if (use_eb) {
                Real const vfm = vfrac(cellm);
                Real const vfp = vfrac(edge);
                return (sigma(cellm)*vfm + sigma(edge)*vfp) / (vfm+vfp);
            } else {
                return Real(0.5)*(sigma(cellm)+sigma(edge));
            }
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real axialEdgeSigma (IntVect const& edge, Real r) const
        {
            if (!has_sigma) {
                return Real(1.0);
            }

            IntVect cellm = edge;
            --cellm[0];
            Real const rp = r + Real(0.5)*dr;
            Real const rm = amrex::Math::abs(r-Real(0.5)*dr);
            if (use_eb) {
                Real const wm = vfrac(cellm)*rm;
                Real const wp = vfrac(edge)*rp;
                return (sigma(cellm)*wm + sigma(edge)*wp) / (wm+wp);
            } else {
                return (sigma(cellm)*rm + sigma(edge)*rp) / (rm+rp);
            }
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real apply (IntVect const& iv, int, Array4<Real> const& xa, int n) const
        {
            Real const r = rlo + Real(iv[0])*dr;
            if (dirmsk(iv) || (r == Real(0.0) && alpha != Real(0.0))) {
                return Real(0.0);
            }

            IntVect ivxm, ivxp, ivym, ivyp;
            getNeighbors(iv, 0, ivxm, ivxp);
            getNeighbors(iv, 1, ivym, ivyp);

            Real const xcenter = xa(iv,n);
            Real out;
            Real scale = Real(1.0);

            if (r == Real(0.0)) {
                Real const sigp = radialEdgeSigma(iv);
                if (use_eb && levset(ivxp) >= Real(0.0)) {
                    Real const hp = (edgecent[0](iv) == Real(1.0))
                        ? Real(1.0) : Real(1.0)+Real(2.0)*edgecent[0](iv);
                    out = -Real(4.0)*sigp*xcenter/(dr*dr*hp*hp);
                    scale = hp;
                } else {
                    out = Real(4.0)*sigp*(xa(ivxp,n)-xcenter)/(dr*dr);
                }
            } else {
                Real hp = Real(1.0);
                Real hm = Real(1.0);
                if (use_eb) {
                    hp = (edgecent[0](iv) == Real(1.0))
                        ? Real(1.0) : Real(1.0)+Real(2.0)*edgecent[0](iv);
                    IntVect edgem = iv;
                    --edgem[0];
                    hm = (edgecent[0](edgem) == Real(1.0))
                        ? Real(1.0) : Real(1.0)-Real(2.0)*edgecent[0](edgem);
                }

                Real const sigp = radialEdgeSigma(iv);
                IntVect edgem = iv;
                --edgem[0];
                Real const sigm = radialEdgeSigma(edgem);
                Real tmp = sigp*(xa(ivxp,n)-xcenter)*(r+Real(0.5)*dr);
                if (use_eb && levset(ivxp) >= Real(0.0)) {
                    tmp = -sigp*xcenter/hp*(r+Real(0.5)*hp*dr);
                }
                if (use_eb && levset(ivxm) >= Real(0.0)) {
                    tmp -= sigm*xcenter/hm*(r-Real(0.5)*hm*dr);
                } else {
                    tmp += sigm*(xa(ivxm,n)-xcenter)*(r-Real(0.5)*dr);
                }
                out = tmp*Real(2.0)/((hp+hm)*r*dr*dr);
                scale = amrex::min(hm,hp);
            }

            Real hp = Real(1.0);
            Real hm = Real(1.0);
            if (use_eb) {
                hp = (edgecent[1](iv) == Real(1.0))
                    ? Real(1.0) : Real(1.0)+Real(2.0)*edgecent[1](iv);
                IntVect edgem = iv;
                --edgem[1];
                hm = (edgecent[1](edgem) == Real(1.0))
                    ? Real(1.0) : Real(1.0)-Real(2.0)*edgecent[1](edgem);
            }

            Real const sigp = axialEdgeSigma(iv,r);
            IntVect edgem = iv;
            --edgem[1];
            Real const sigm = axialEdgeSigma(edgem,r);
            Real tmp = sigp*(xa(ivyp,n)-xcenter);
            if (use_eb && levset(ivyp) >= Real(0.0)) {
                tmp = -sigp*xcenter/hp;
            }
            if (use_eb && levset(ivym) >= Real(0.0)) {
                tmp -= sigm*xcenter/hm;
            } else {
                tmp += sigm*(xa(ivym,n)-xcenter);
            }
            out += tmp*Real(2.0)/((hp+hm)*dz*dz);
            scale = amrex::min(scale,hm,hp);

            if (r != Real(0.0)) {
                out -= alpha*xcenter/(r*r);
            }
            return out*scale;
        }

        Real sigr;
        Real dr;
        Real dz;
        Real rlo;
        Real alpha;
        Array4<Real const> sigma;
        Array4<Real const> vfrac;
        Array4<Real const> levset;
        GpuArray<Array4<Real const>,AMREX_SPACEDIM> edgecent;
        bool has_sigma;
        bool use_eb;
    };
#endif

#ifdef AMREX_USE_EB
    struct LPEB
        : public LPBase
    {
        LPEB (LPBase const& a_lpbase, Array4<Real const> const& a_levset,
              GpuArray<Array4<Real const>,AMREX_SPACEDIM> const& a_edgecent,
              Array4<Real const> const& a_sigma, Array4<Real const> const& a_vfrac,
              bool a_has_sigma)
            : LPBase(a_lpbase), levset(a_levset), edgecent(a_edgecent),
              sigma(a_sigma), vfrac(a_vfrac), has_sigma(a_has_sigma)
            {}

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real edgeSigma (IntVect const& edge, int idim) const
        {
            if (!has_sigma) {
                return Real(1.0);
            }

            constexpr int nsig = 1 << (AMREX_SPACEDIM-1);
            Real sigsum = 0;
            Real vfsum = 0;
            for (int isig = 0; isig < nsig; ++isig) {
                IntVect cell = edge;
                int ibit = 0;
                for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                    if (jdim != idim) {
                        cell[jdim] += ((isig >> ibit) & 1) - 1;
                        ++ibit;
                    }
                }
                Real const vf = vfrac(cell);
                sigsum += sigma(cell)*vf;
                vfsum += vf;
            }
            return sigsum/vfsum;
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        Real apply (IntVect const& iv, int, Array4<Real> const& xa, int n) const
        {
            Real y = 0;
            if (!dirmsk(iv)) {
                Real const xcenter = xa(iv,n);
                Real scale = Real(1.0);

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    IntVect ivm, ivp;
                    getNeighbors(iv, idim, ivm, ivp);

                    IntVect edgem = iv;
                    --edgem[idim];

                    Real const hp = (edgecent[idim](iv) == Real(1.0))
                        ? Real(1.0) : Real(1.0)+Real(2.0)*edgecent[idim](iv);
                    Real const hm = (edgecent[idim](edgem) == Real(1.0))
                        ? Real(1.0) : Real(1.0)-Real(2.0)*edgecent[idim](edgem);

                    Real const sigp = edgeSigma(iv, idim);
                    Real const sigm = edgeSigma(edgem, idim);
                    Real tmp = (levset(ivp) < Real(0.0))
                        ? sigp*(xa(ivp,n)-xcenter)
                        : -sigp*xcenter/hp;
                    tmp += (levset(ivm) < Real(0.0))
                        ? sigm*(xa(ivm,n)-xcenter)
                        : -sigm*xcenter/hm;

                    y += beta[idim]*tmp*Real(2.0)/(hp+hm);
                    scale = amrex::min(scale, hm, hp);
                }

                y *= scale;
            }
            return y;
        }

        Array4<Real const> levset;
        GpuArray<Array4<Real const>,AMREX_SPACEDIM> edgecent;
        Array4<Real const> sigma;
        Array4<Real const> vfrac;
        bool has_sigma;
    };
#endif
}

void
MLEBNodeFDLaplacian::customBottomSolve (MLMGT<MultiFab>* mlmg, MultiFab& x, const MultiFab& b,
                                        Real eps_rel, Real eps_abs, int maxiter)
{
    amrex::ignore_unused(eps_rel, eps_abs, maxiter);

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    constexpr Long old_solver_min_points = Long(97)*97*97;
    bool use_custom_solver = (x.size() == 1) &&
        (x.boxArray()[0].numPts() < old_solver_min_points);
    if (use_custom_solver)
    {
        int const amrlev = 0;
        int const mglev = NMGLevels(0) - 1;

#ifdef AMREX_USE_EB
        auto const* eb_factory = dynamic_cast<EBFArrayBoxFactory const*>
            (m_factory[amrlev][mglev].get());
        bool const use_eb = eb_factory && !eb_factory->isAllRegular();
#endif

        int ret;
        if (ParallelDescriptor::MyProc() == x.DistributionMap()[0])
        {
            auto const& geom = m_geom[amrlev][mglev];
            const auto dxinv = geom.InvCellSizeArray();
#if (AMREX_SPACEDIM == 2)
            const auto sig0 = m_sigma[0];
            const auto dx0 = geom.CellSize(0);
            const auto dx1 = geom.CellSize(1)/std::sqrt(m_sigma[1]);
            const auto xlo = geom.ProbLo(0);
            const auto alpha = m_rz_alpha;
#endif
            AMREX_D_TERM(const Real bx = m_sigma[0]*dxinv[0]*dxinv[0];,
                         const Real by = m_sigma[1]*dxinv[1]*dxinv[1];,
                         const Real bz = m_sigma[2]*dxinv[2]*dxinv[2];)

            auto const& dotmsk = m_bottom_dot_mask[0].const_array();
            auto const& dirmsk = (*m_dirichlet_mask[amrlev][mglev])[0].const_array();
            Box box = x.boxArray()[0];
            Box dbox = amrex::convert(geom.Domain(),IntVect(1));
            LPBase lpbase{dotmsk,dirmsk,
                          GpuArray<Real,AMREX_SPACEDIM>{AMREX_D_DECL(bx,by,bz)},
                          dbox.smallEnd(),dbox.bigEnd(),
                          GpuArray<bool,AMREX_SPACEDIM>{AMREX_D_DECL(geom.isPeriodic(0),
                                                                     geom.isPeriodic(1),
                                                                     geom.isPeriodic(2))}};
#if (AMREX_SPACEDIM == 2)
            if (m_rz) {
                Array4<Real const> sigma;
                if (m_has_sigma_mf) {
                    sigma = (*m_sigma_mf[amrlev][mglev])[0].const_array();
                }
#ifdef AMREX_USE_EB
                if (use_eb) {
                    auto const& edgecent = eb_factory->getEdgeCent();
                    AMREX_ALWAYS_ASSERT(edgecent[0] && edgecent[0]->ok(0));
                    GpuArray<Array4<Real const>,AMREX_SPACEDIM> const ec
                        {(*edgecent[0])[0].const_array(), (*edgecent[1])[0].const_array()};
                    auto const& levset = eb_factory->getLevelSet()[0].const_array();
                    Array4<Real const> vfrac;
                    if (m_has_sigma_mf) {
                        vfrac = eb_factory->getVolFrac()[0].const_array();
                    }
                    LPRZ lp(lpbase, sig0, dx0, dx1, xlo, alpha, sigma, vfrac,
                            levset, ec, m_has_sigma_mf, true);
                    ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
                } else
#endif
                {
                    LPRZ lp(lpbase, sig0, dx0, dx1, xlo, alpha, sigma, {}, {}, {},
                            m_has_sigma_mf, false);
                    ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
                }
            } else
#endif
#ifdef AMREX_USE_EB
            if (use_eb) {
                auto const& edgecent = eb_factory->getEdgeCent();
                AMREX_ALWAYS_ASSERT(edgecent[0] && edgecent[0]->ok(0));
                GpuArray<Array4<Real const>,AMREX_SPACEDIM> const ec
                    {AMREX_D_DECL((*edgecent[0])[0].const_array(),
                                  (*edgecent[1])[0].const_array(),
                                  (*edgecent[2])[0].const_array())};
                auto const& levset = eb_factory->getLevelSet()[0].const_array();
                if (m_has_sigma_mf) {
                    auto const& sigma = (*m_sigma_mf[amrlev][mglev])[0].const_array();
                    auto const& vfrac = eb_factory->getVolFrac()[0].const_array();
                    LPEB lp(lpbase, levset, ec, sigma, vfrac, true);
                    ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
                } else {
                    LPEB lp(lpbase, levset, ec, {}, {}, false);
                    ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
                }
            } else
#endif
            if (m_has_sigma_mf) {
                auto const& sigma = (*m_sigma_mf[amrlev][mglev])[0].const_array();
                LPSigma lp(lpbase, sigma);
                ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
            } else {
                LP lp(lpbase);
                ret = bicgstab_solve(box, x[0], b[0], lp, eps_rel, eps_abs, maxiter);
            }
        }

        if (ParallelContext::NProcsSub() > 1) {
            int root = ParallelContext::global_to_local_rank(x.DistributionMap()[0]);
            ParallelDescriptor::Bcast(&ret, 1, root, ParallelContext::CommunicatorSub());
        }

        mlmg->postCG(ret);
    } else
#endif
    {
        int ret = mlmg->bottomSolveWithCG(x, b, MLCGSolverT<MultiFab>::Type::BiCGStab);
        mlmg->postCG(ret);
    }
}

}
