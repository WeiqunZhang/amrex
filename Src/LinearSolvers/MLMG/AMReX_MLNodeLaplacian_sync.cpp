#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLNodeLap_K.H>
#include <AMReX_MultiFabUtil.H>

namespace amrex {

void
MLNodeLaplacian::compSyncResidualCoarse (MultiFab& sync_resid, const MultiFab& a_phi,
                                         const MultiFab& vold, const MultiFab* rhcc,
                                         const BoxArray& fine_grids, const IntVect& ref_ratio)
{
    BL_PROFILE("MLNodeLaplacian::SyncResCrse()");

    sync_resid.setVal(0.0);

    const Geometry& geom = m_geom[0][0];
    const DistributionMapping& dmap = m_dmap[0][0];
    const BoxArray& ccba = m_grids[0][0];
    const BoxArray& ndba = amrex::convert(ccba, IntVect::TheNodeVector());
    const BoxArray& ccfba = amrex::convert(fine_grids, IntVect::TheZeroVector());

    const auto lobc = LoBC();
    const auto hibc = HiBC();

    // cell-center, 1: coarse; 0: covered by fine
    const int owner = 1;
    const int nonowner = 0;
    iMultiFab crse_cc_mask = amrex::makeFineMask(ccba, dmap, IntVect(1), ccfba, ref_ratio,
                                                 geom.periodicity(), owner, nonowner);

    const Box& ccdom = geom.Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(crse_cc_mask); mfi.isValid(); ++mfi)
    {
        Array4<int> const& fab = crse_cc_mask.array(mfi);
        mlndlap_fillbc_cc<int>(mfi.validbox(),fab,ccdom,lobc,hibc);
    }

    MultiFab phi(ndba, dmap, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Box& gbx = mfi.growntilebox();
        Array4<Real> const& fab = phi.array(mfi);
        Array4<Real const> const& a_fab = a_phi.const_array(mfi);
        Array4<int const> const& msk = crse_cc_mask.const_array(mfi);
        AMREX_HOST_DEVICE_FOR_3D(gbx, i, j, k,
        {
            if (bx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                fab(i,j,k) = a_fab(i,j,k);
                mlndlap_zero_fine(i,j,k,fab,msk,nonowner);
            } else {
                fab(i,j,k) = 0.0;
            }
        });
    }

    const auto& nddom = amrex::surroundingNodes(ccdom);

    const auto dxinv = geom.InvCellSizeArray();

#if (AMREX_SPACEDIM == 2)
    bool is_rz = m_is_rz;
#endif

    const auto& sigma_orig = m_sigma[0][0][0];
    const iMultiFab& dmsk = *m_dirichlet_mask[0][0];

#ifdef AMREX_USE_EB
    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[0][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* intg = m_integral[0].get();
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
#endif

    bool neumann_doubling = true; // yes even for RAP, because unimposeNeumannBC will be called on rhs

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox rhs, u;
#ifdef AMREX_USE_EB
        FArrayBox sten, cn;
#endif
        for (MFIter mfi(sync_resid,mfi_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            auto typ = FabType::regular;
#ifdef AMREX_USE_EB
            if (factory) {
                typ = (*flags)[mfi].getType(amrex::enclosedCells(bx));
            }
#endif
            if (typ != FabType::covered)
            {
                const Box& bxg1 = amrex::grow(bx,1);
                const Box& ccbxg1 = amrex::enclosedCells(bxg1);
                Array4<int const> const& cccmsk = crse_cc_mask.const_array(mfi);

                bool has_fine;
                if (Gpu::inLaunchRegion()) {
                    AMREX_ASSERT(ccbxg1 == crse_cc_mask[mfi].box());
                    has_fine = Reduce::AnyOf(ccbxg1,
                                             [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> bool
                    {
                        return cccmsk(i,j,k) == nonowner;
                    });
                } else {
                    has_fine = mlndlap_any_fine_sync_cells(ccbxg1,cccmsk,nonowner);
                }

                if (has_fine)
                {
                    const Box& ccvbx = amrex::enclosedCells(mfi.validbox());

                    u.resize(ccbxg1, AMREX_SPACEDIM);
                    Elixir ueli = u.elixir();
                    Array4<Real> const& uarr = u.array();

                    Box b = ccbxg1 & ccvbx;
                    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                    {
                        if (m_lobc[0][idim] == LinOpBCType::inflow)
                        {
                            if (b.smallEnd(idim) == ccdom.smallEnd(idim)) {
                                b.growLo(idim, 1);
                            }
                        }
                        if (m_hibc[0][idim] == LinOpBCType::inflow)
                        {
                            if (b.bigEnd(idim) == ccdom.bigEnd(idim)) {
                                b.growHi(idim, 1);
                            }
                        }
                    }

                    Array4<Real const> const& voarr = vold.const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                    {
                        if (b.contains(IntVect(AMREX_D_DECL(i,j,k))) && cccmsk(i,j,k)){
                            AMREX_D_TERM(uarr(i,j,k,0) = voarr(i,j,k,0);,
                                         uarr(i,j,k,1) = voarr(i,j,k,1);,
                                         uarr(i,j,k,2) = voarr(i,j,k,2););
                        } else {
                            AMREX_D_TERM(uarr(i,j,k,0) = 0.0;,
                                         uarr(i,j,k,1) = 0.0;,
                                         uarr(i,j,k,2) = 0.0;);
                        }
                    });

                    rhs.resize(bx);
                    Elixir rhseli = rhs.elixir();
                    Array4<Real> const& rhsarr = rhs.array();
                    Array4<int const> const& dmskarr = dmsk.const_array(mfi);

#ifdef AMREX_USE_EB
                    if (typ == FabType::singlevalued)
                    {
                        Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                        Array4<Real const> const& intgarr = intg->const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                        {
                            mlndlap_divu_eb(i,j,k,rhsarr,uarr,vfracarr,intgarr,dmskarr,dxinv,nddom,lobc,hibc);
                        });
                    }
                    else
#endif
                    {
#if (AMREX_SPACEDIM == 2)
                        AMREX_HOST_DEVICE_PARALLEL_FOR_3D (bx, i, j, k,
                        {
                            mlndlap_divu(i,j,k,rhsarr,uarr,dmskarr,dxinv,nddom,lobc,hibc,is_rz);
                        });
#else
                        AMREX_HOST_DEVICE_PARALLEL_FOR_3D (bx, i, j, k,
                        {
                            mlndlap_divu(i,j,k,rhsarr,uarr,dmskarr,dxinv,nddom,lobc,hibc);
                        });
#endif
                    }

                    if (rhcc)
                    {
                        Array4<Real> rhccarr = uarr;
                        Array4<Real const> const& rhccarr_orig = rhcc->const_array(mfi);
                        const Box& b2 = ccbxg1 & ccvbx;
                        AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                        {
                            if (b2.contains(IntVect(AMREX_D_DECL(i,j,k))) && cccmsk(i,j,k)){
                                rhccarr(i,j,k) = rhccarr_orig(i,j,k);
                            } else {
                                rhccarr(i,j,k) = 0.0;
                            }
                        });

#ifdef AMREX_USE_EB
                        if (typ == FabType::singlevalued)
                        {
                            Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                            Array4<Real const> const& intgarr = intg->const_array(mfi);
                            AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                            {
                                Real rhs2 = mlndlap_rhcc_eb(i,j,k,rhccarr,vfracarr,intgarr,dmskarr);
                                rhsarr(i,j,k) += rhs2;
                            });
                        }
                        else
#endif
                        {
                            AMREX_HOST_DEVICE_FOR_3D (bx, i, j, k,
                            {
                                Real rhs2 = mlndlap_rhcc(i,j,k,rhccarr,dmskarr);
                                rhsarr(i,j,k) += rhs2;
                            });
                        }
                    }

                    Array4<Real> const& sync_resid_a = sync_resid.array(mfi);
                    Array4<Real const> const& phiarr = phi.const_array(mfi);
#ifdef AMREX_USE_EB
                    if (typ == FabType::singlevalued)
                    {
                        Array4<Real const> const& sigmaarr_orig = sigma_orig->const_array(mfi);

                        Box stbx = bx;
                        AMREX_D_TERM(stbx.growLo(0,1);, stbx.growLo(1,1);, stbx.growLo(2,1));
                        Box const& sgbx = amrex::grow(amrex::enclosedCells(stbx),1);

                        constexpr int ncomp_s = (AMREX_SPACEDIM == 2) ? 5 : 9;
                        sten.resize(stbx,ncomp_s);
                        Elixir steneli = sten.elixir();
                        Array4<Real> const& stenarr = sten.array();

                        constexpr int ncomp_c = (AMREX_SPACEDIM == 2) ? 6 : 27;
                        cn.resize(sgbx,ncomp_c+1);
                        Elixir cneli = cn.elixir();
                        Array4<Real> const& cnarr = cn.array();
                        Array4<Real> const& sgarr = cn.array(ncomp_c);

                        Array4<EBCellFlag const> const& flagarr = flags->const_array(mfi);
                        Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                        Array4<Real const> const& intgarr = intg->const_array(mfi);

                        const Box& ibx = sgbx & amrex::enclosedCells(mfi.validbox());
                        AMREX_HOST_DEVICE_FOR_3D(sgbx, i, j, k,
                        {
                            if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k))) && cccmsk(i,j,k)) {
                                mlndlap_set_connection(i,j,k,cnarr,intgarr,vfracarr,flagarr);
                                sgarr(i,j,k) = sigmaarr_orig(i,j,k);
                            } else {
                                for (int n = 0; n < ncomp_c; ++n) {
                                    cnarr(i,j,k,n) = 0.0;
                                }
                                sgarr(i,j,k) = 0.0;
                            }
                        });

                        AMREX_HOST_DEVICE_FOR_3D(stbx, i, j, k,
                        {
                            mlndlap_set_stencil_eb(i, j, k, stenarr, sgarr, cnarr, dxinv);
                        });

                        AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                        {
                            mlndlap_set_stencil_s0(i, j, k, stenarr);
                            sync_resid_a(i,j,k) = mlndlap_adotx_sten(i, j, k, phiarr, stenarr, dmskarr);
                            mlndlap_crse_resid(i, j, k, sync_resid_a, rhsarr, cccmsk, nddom, lobc, hibc, neumann_doubling);
                        });
                    }
                    else
#endif
                    {
                        Array4<Real> sigmaarr = uarr;
                        const Box& ibx = ccbxg1 & amrex::enclosedCells(mfi.validbox());
                        if (sigma_orig) {
                            Array4<Real const> const& sigmaarr_orig = sigma_orig->const_array(mfi);
                            AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                            {
                                if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k))) && cccmsk(i,j,k)) {
                                    sigmaarr(i,j,k) = sigmaarr_orig(i,j,k);
                                } else {
                                    sigmaarr(i,j,k) = 0.0;
                                }
                            });
                        } else {
                            Real const_sigma = m_const_sigma;
                            AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                            {
                                if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k))) && cccmsk(i,j,k)) {
                                    sigmaarr(i,j,k) = const_sigma;
                                } else {
                                    sigmaarr(i,j,k) = 0.0;
                                }
                            });
                        }

#if (AMREX_SPACEDIM == 2)
                        AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                        {
                            sync_resid_a(i,j,k) = mlndlap_adotx_aa(i, j, k, phiarr, sigmaarr, dmskarr, is_rz, dxinv);
                            mlndlap_crse_resid(i, j, k, sync_resid_a, rhsarr, cccmsk, nddom, lobc, hibc, neumann_doubling);
                        });
#else
                        AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                        {
                            sync_resid_a(i,j,k) = mlndlap_adotx_aa(i, j, k, phiarr, sigmaarr, dmskarr, dxinv);
                            mlndlap_crse_resid(i, j, k, sync_resid_a, rhsarr, cccmsk, nddom, lobc, hibc, neumann_doubling);
                        });
#endif
                    }
                }
            }
        }
    }
}

void
MLNodeLaplacian::compSyncResidualFine (MultiFab& sync_resid, const MultiFab& phi, const MultiFab& vold,
                                       const MultiFab* rhcc)
{
    BL_PROFILE("MLNodeLaplacian::SyncResFine()");

    const auto& sigma_orig = m_sigma[0][0][0];
    const iMultiFab& dmsk = *m_dirichlet_mask[0][0];

    const auto lobc = LoBC();
    const auto hibc = HiBC();

#ifdef AMREX_USE_EB
    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[0][0].get());
    const FabArray<EBCellFlagFab>* flags = (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    const MultiFab* intg = m_integral[0].get();
    const MultiFab* vfrac = (factory) ? &(factory->getVolFrac()) : nullptr;
#endif

    const Geometry& geom = m_geom[0][0];
    const Box& ccdom = geom.Domain();
    const Box& nddom = amrex::surroundingNodes(ccdom);
    const auto dxinv = geom.InvCellSizeArray();
#if (AMREX_SPACEDIM == 2)
    bool is_rz = m_is_rz;
#endif

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox rhs, u;
        IArrayBox tmpmask;
#ifdef AMREX_USE_EB
        FArrayBox sten, cn;
#endif
        for (MFIter mfi(sync_resid,mfi_info); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            auto typ = FabType::regular;
#ifdef AMREX_USE_EB
            if (factory) {
                typ = (*flags)[mfi].getType(amrex::enclosedCells(bx));
            }
#endif
            if (typ != FabType::covered)
            {
                const Box& gbx = mfi.growntilebox();
                const Box& vbx = mfi.validbox();
                const Box& ccvbx = amrex::enclosedCells(vbx);
                const Box& bxg1 = amrex::grow(bx,1);
                const Box& ccbxg1 = amrex::enclosedCells(bxg1);

                u.resize(ccbxg1, AMREX_SPACEDIM);
                Elixir ueli = u.elixir();
                Array4<Real> const& uarr = u.array();

                Box ovlp = ccvbx & ccbxg1;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    if (m_lobc[0][idim] == LinOpBCType::inflow)
                    {
                        if (ovlp.smallEnd(idim) == ccdom.smallEnd(idim)) {
                            ovlp.growLo(idim, 1);
                        }
                    }
                    if (m_hibc[0][idim] == LinOpBCType::inflow)
                    {
                        if (ovlp.bigEnd(idim) == ccdom.bigEnd(idim)) {
                            ovlp.growHi(idim, 1);
                        }
                    }
                }

                Array4<Real const> const& voarr = vold.const_array(mfi);
                AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                {
                    if (ovlp.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                        AMREX_D_TERM(uarr(i,j,k,0) = voarr(i,j,k,0);,
                                     uarr(i,j,k,1) = voarr(i,j,k,1);,
                                     uarr(i,j,k,2) = voarr(i,j,k,2););
                    } else {
                        AMREX_D_TERM(uarr(i,j,k,0) = 0.0;,
                                     uarr(i,j,k,1) = 0.0;,
                                     uarr(i,j,k,2) = 0.0;);
                    }
                });

                tmpmask.resize(bx);
                Elixir tmeli = tmpmask.elixir();
                Array4<int> const& tmpmaskarr = tmpmask.array();
                Array4<int const> const& dmskarr = dmsk.const_array(mfi);
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    tmpmaskarr(i,j,k) = 1-dmskarr(i,j,k);
                });

                rhs.resize(bx);
                Elixir rhseli = rhs.elixir();
                Array4<Real> const& rhsarr = rhs.array();

#ifdef AMREX_USE_EB
                if (typ == FabType::singlevalued)
                {
                    Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                    Array4<Real const> const& intgarr = intg->const_array(mfi);
                    AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                    {
                        mlndlap_divu_eb(i,j,k,rhsarr,uarr,vfracarr,intgarr,tmpmaskarr,dxinv,nddom,lobc,hibc);
                    });
                }
                else
#endif
                {
#if (AMREX_SPACEDIM == 2)
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D (bx, i, j, k,
                    {
                        mlndlap_divu(i,j,k,rhsarr,uarr,tmpmaskarr,dxinv,nddom,lobc,hibc,is_rz);
                    });
#else
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D (bx, i, j, k,
                    {
                        mlndlap_divu(i,j,k,rhsarr,uarr,tmpmaskarr,dxinv,nddom,lobc,hibc);
                    });
#endif
                }

                if (rhcc)
                {
                    Array4<Real> rhccarr = uarr;
                    Array4<Real const> const& rhccarr_orig = rhcc->const_array(mfi);
                    const Box& ovlp3 = ccvbx & ccbxg1;
                    AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                    {
                        if (ovlp3.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                            rhccarr(i,j,k) = rhccarr_orig(i,j,k);
                        } else {
                            rhccarr(i,j,k) = 0.0;
                        }
                    });
#ifdef AMREX_USE_EB
                    if (typ == FabType::singlevalued)
                    {
                        Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                        Array4<Real const> const& intgarr = intg->const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                        {
                            Real rhs2 = mlndlap_rhcc_eb(i,j,k,rhccarr,vfracarr,intgarr,tmpmaskarr);
                            rhsarr(i,j,k) += rhs2;
                        });
                    }
                    else
#endif
                    {
                        AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                        {
                            Real rhs2 = mlndlap_rhcc(i,j,k,rhccarr,tmpmaskarr);
                            rhsarr(i,j,k) += rhs2;
                        });
                    }
                }

                Array4<Real> const& sync_resid_a = sync_resid.array(mfi);
                Array4<Real const> const& phiarr = phi.const_array(mfi);
#ifdef AMREX_USE_EB
                if (typ == FabType::singlevalued)
                {
                    Array4<Real const> const& sigmaarr_orig = sigma_orig->const_array(mfi);

                    Box stbx = bx;
                    AMREX_D_TERM(stbx.growLo(0,1);, stbx.growLo(1,1);, stbx.growLo(2,1));
                    Box const& sgbx = amrex::grow(amrex::enclosedCells(stbx),1);

                    constexpr int ncomp_s = (AMREX_SPACEDIM == 2) ? 5 : 9;
                    sten.resize(stbx,ncomp_s);
                    Elixir steneli = sten.elixir();
                    Array4<Real> const& stenarr = sten.array();

                    constexpr int ncomp_c = (AMREX_SPACEDIM == 2) ? 6 : 27;
                    cn.resize(sgbx,ncomp_c+1);
                    Elixir cneli = cn.elixir();
                    Array4<Real> const& cnarr = cn.array();
                    Array4<Real> const& sgarr = cn.array(ncomp_c);

                    Array4<EBCellFlag const> const& flagarr = flags->const_array(mfi);
                    Array4<Real const> const& vfracarr = vfrac->const_array(mfi);
                    Array4<Real const> const& intgarr = intg->const_array(mfi);

                    const Box& ibx = sgbx & amrex::enclosedCells(mfi.validbox());
                    AMREX_HOST_DEVICE_FOR_3D(sgbx, i, j, k,
                    {
                        if (ibx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                            mlndlap_set_connection(i,j,k,cnarr,intgarr,vfracarr,flagarr);
                            sgarr(i,j,k) = sigmaarr_orig(i,j,k);
                        } else {
                            for (int n = 0; n < ncomp_c; ++n) {
                                cnarr(i,j,k,n) = 0.0;
                            }
                            sgarr(i,j,k) = 0.0;
                        }
                    });

                    AMREX_HOST_DEVICE_FOR_3D(stbx, i, j, k,
                    {
                        mlndlap_set_stencil_eb(i, j, k, stenarr, sgarr, cnarr, dxinv);
                    });

                    AMREX_HOST_DEVICE_FOR_3D(gbx, i, j, k,
                    {
                        if (bx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                            mlndlap_set_stencil_s0(i, j, k, stenarr);
                            sync_resid_a(i,j,k) = mlndlap_adotx_sten(i,j,k, phiarr, stenarr, tmpmaskarr);
                            sync_resid_a(i,j,k) = rhsarr(i,j,k) - sync_resid_a(i,j,k);
                        } else {
                            sync_resid_a(i,j,k) = 0.0;
                        }
                    });
                }
                else
#endif
                {
                    Array4<Real> sigmaarr = uarr;
                    const Box& ovlp2 = ccvbx & ccbxg1;
                    if (sigma_orig) {
                        Array4<Real const> const& sigmaarr_orig = sigma_orig->const_array(mfi);
                        AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                        {
                            if (ovlp2.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                                sigmaarr(i,j,k) = sigmaarr_orig(i,j,k);
                            } else {
                                sigmaarr(i,j,k) = 0.0;
                            }
                        });
                    } else {
                        Real const_sigma = m_const_sigma;
                        AMREX_HOST_DEVICE_FOR_3D(ccbxg1, i, j, k,
                        {
                            if (ovlp2.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                                sigmaarr(i,j,k) = const_sigma;
                            } else {
                                sigmaarr(i,j,k) = 0.0;
                            }
                        });
                    }

#if (AMREX_SPACEDIM == 2)
                    AMREX_HOST_DEVICE_FOR_3D(gbx, i, j, k,
                    {
                        if (bx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                            sync_resid_a(i,j,k) = mlndlap_adotx_aa(i,j,k, phiarr, sigmaarr, tmpmaskarr,
                                                                   is_rz, dxinv);
                            sync_resid_a(i,j,k) = rhsarr(i,j,k) - sync_resid_a(i,j,k);
                        } else {
                            sync_resid_a(i,j,k) = 0.0;
                        }
                    });
#else
                    AMREX_HOST_DEVICE_FOR_3D(gbx, i, j, k,
                    {
                        if (bx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                            sync_resid_a(i,j,k) = mlndlap_adotx_aa(i,j,k, phiarr, sigmaarr, tmpmaskarr,
                                                                   dxinv);
                            sync_resid_a(i,j,k) = rhsarr(i,j,k) - sync_resid_a(i,j,k);
                        } else {
                            sync_resid_a(i,j,k) = 0.0;
                        }
                    });
#endif
                }
            }

            // Do not impose neumann bc here because how SyncRegister works.
        }
    }
}

}
