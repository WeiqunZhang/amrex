#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLNodeLap_K.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

namespace amrex {

void
MLNodeLaplacian::reflux (int crse_amrlev,
                         MultiFab& res, const MultiFab& crse_sol, const MultiFab& crse_rhs,
                         MultiFab& a_fine_res, MultiFab& fine_sol, const MultiFab& fine_rhs) const
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(crse_amrlev,res,crse_sol,crse_rhs,a_fine_res,fine_sol,fine_rhs);
#else
    //
    //  Note that the residue we copmute on a coarse/fine node is not a
    //  composite divergence.  It has been restricted so that it is suitable
    //  as RHS for our geometric mulitgrid solver with a MG hirerachy
    //  including multiple AMR levels.
    //

    BL_PROFILE("MLNodeLaplacian::reflux()");

    const int amrrr = AMRRefRatio(crse_amrlev);
    AMREX_ALWAYS_ASSERT(amrrr == 2 || m_coarsening_strategy == CoarseningStrategy::Sigma);

    const Geometry& cgeom = m_geom[crse_amrlev  ][0];
    const Geometry& fgeom = m_geom[crse_amrlev+1][0];
    const auto cdxinv = cgeom.InvCellSizeArray();
    const auto fdxinv = fgeom.InvCellSizeArray();
    const Box& c_cc_domain = cgeom.Domain();
    const Box& c_cc_domain_p = cgeom.growPeriodicDomain(1);
    const Box& c_nd_domain = amrex::surroundingNodes(c_cc_domain);
    const Box& f_nd_domain = amrex::surroundingNodes(fgeom.Domain());

    const auto lobc = LoBC();
    const auto hibc = HiBC();

    bool neumann_doubling = false;
    if (m_coarsening_strategy == CoarseningStrategy::Sigma) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            neumann_doubling = neumann_doubling || (lobc[idim] == LinOpBCType::inflow  ||
                                                    lobc[idim] == LinOpBCType::Neumann ||
                                                    hibc[idim] == LinOpBCType::inflow  ||
                                                    hibc[idim] == LinOpBCType::Neumann);
        }
    }

#if (AMREX_SPACEDIM == 2)
    bool is_rz = m_is_rz;
#endif

    const BoxArray& fba = fine_sol.boxArray();
    const DistributionMapping& fdm = fine_sol.DistributionMap();

    const iMultiFab& fdmsk = *m_dirichlet_mask[crse_amrlev+1][0];
    const auto& stencil    =  m_nosigma_stencil[crse_amrlev+1];

    MultiFab fine_res_for_coarse(amrex::coarsen(fba, amrrr), fdm, 1, 0);

    std::unique_ptr<MultiFab> tmp_fine_res;
    if (amrrr == 4 && !a_fine_res.nGrowVect().allGE(IntVect(3))) {
        tmp_fine_res = std::make_unique<MultiFab>(a_fine_res.boxArray(),
                                                  a_fine_res.DistributionMap(), 1, 3);
        MultiFab::Copy(*tmp_fine_res, a_fine_res, 0, 0, 1, 0);
    }
    MultiFab& fine_res = (tmp_fine_res) ? *tmp_fine_res :  a_fine_res;

    fine_res.FillBoundary(fgeom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fine_res_for_coarse, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& cfab = fine_res_for_coarse.array(mfi);
        Array4<Real const> const& ffab = fine_res.const_array(mfi);
        Array4<int const> const& mfab = fdmsk.const_array(mfi);
        if (m_coarsening_strategy == CoarseningStrategy::Sigma) {
            if (amrrr == 2) {
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                {
                    mlndlap_restriction<2>(i,j,k,cfab,ffab,mfab,f_nd_domain,lobc,hibc);
                });
            } else {
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                {
                    mlndlap_restriction<4>(i,j,k,cfab,ffab,mfab,f_nd_domain,lobc,hibc);
                });
            }
        } else {
            Array4<Real const> const& stfab = stencil->const_array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
            {
                mlndlap_restriction_rap(i,j,k,cfab,ffab,stfab,mfab);
            });
        }
    }
    res.ParallelCopy(fine_res_for_coarse, cgeom.periodicity());

    MultiFab fine_contrib(amrex::coarsen(fba, amrrr), fdm, 1, 0);

    const auto& fsigma = m_sigma[crse_amrlev+1][0][0];

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fine_contrib,mfi_info); mfi.isValid(); ++mfi)
    {
        const Box& cbx = mfi.tilebox();
        const Box& fvbx = amrex::refine(mfi.validbox(),amrrr);
        const Box& cc_fvbx = amrex::enclosedCells(fvbx);

        Array4<Real> const& farr = fine_contrib.array(mfi);
        Array4<Real const> const& resarr = fine_res.const_array(mfi);
        Array4<Real const> const& rhsarr = fine_rhs.const_array(mfi);
        Array4<Real const> const& solarr = fine_sol.const_array(mfi);
        Array4<int const> const& marr = fdmsk.const_array(mfi);

        if (fsigma) {
            Array4<Real const> const& sigarr = fsigma->const_array(mfi);
#if (AMREX_SPACEDIM == 2)
            if (amrrr == 2) {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib<2>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                               sigarr,marr,is_rz,fdxinv);
                });
            } else {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib<4>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                               sigarr,marr,is_rz,fdxinv);
                });
            }
#elif (AMREX_SPACEDIM == 3)
            if (amrrr == 2) {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib<2>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                               sigarr,marr,fdxinv);
                });
            } else {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib<4>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                               sigarr,marr,fdxinv);
                });
            }
#endif
        } else {
            Real const_sigma = m_const_sigma;
#if (AMREX_SPACEDIM == 1)
            amrex::ignore_unused(const_sigma);
#elif (AMREX_SPACEDIM == 2)
            if (amrrr == 2) {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib_cs<2>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                                  const_sigma,marr,is_rz,fdxinv);
                });
            } else {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib_cs<4>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                                  const_sigma,marr,is_rz,fdxinv);
                });
            }
#elif (AMREX_SPACEDIM == 3)
            if (amrrr == 2) {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib_cs<2>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                                  const_sigma,marr,fdxinv);
                });
            } else {
                AMREX_HOST_DEVICE_FOR_3D(cbx, i, j, k,
                {
                    mlndlap_Ax_fine_contrib_cs<4>(i,j,k,fvbx,cc_fvbx,farr,resarr,rhsarr,solarr,
                                                  const_sigma,marr,fdxinv);
                });
            }
#endif
        }
    }

    MultiFab fine_contrib_on_crse(crse_sol.boxArray(), crse_sol.DistributionMap(), 1, 0);
    fine_contrib_on_crse.setVal(0.0);
    fine_contrib_on_crse.ParallelAdd(fine_contrib, cgeom.periodicity());

    const iMultiFab& cdmsk = *m_dirichlet_mask[crse_amrlev][0];
    const auto& nd_mask     = m_nd_fine_mask[crse_amrlev];
    const auto& cc_mask     = m_cc_fine_mask[crse_amrlev];
    const auto& has_fine_bndry = m_has_fine_bndry[crse_amrlev];

    const auto& csigma = m_sigma[crse_amrlev][0][0];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(res,mfi_info); mfi.isValid(); ++mfi)
    {
        if ((*has_fine_bndry)[mfi])
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> const& resarr = res.array(mfi);
            Array4<Real const> const& csolarr = crse_sol.const_array(mfi);
            Array4<Real const> const& crhsarr = crse_rhs.const_array(mfi);
            Array4<int const> const& cdmskarr = cdmsk.const_array(mfi);
            Array4<int const> const& ndmskarr = nd_mask->const_array(mfi);
            Array4<int const> const& ccmskarr = cc_mask->const_array(mfi);
            Array4<Real const> const& fcocarr = fine_contrib_on_crse.const_array(mfi);

            if (csigma) {
                Array4<Real const> const& csigarr = csigma->const_array(mfi);
#if (AMREX_SPACEDIM == 2)
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    mlndlap_res_cf_contrib(i,j,k,resarr,csolarr,crhsarr,csigarr,
                                           cdmskarr,ndmskarr,ccmskarr,fcocarr,
                                           cdxinv,c_cc_domain_p,c_nd_domain,
                                           is_rz,
                                           lobc,hibc, neumann_doubling);
                });
#elif (AMREX_SPACEDIM == 3)
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    mlndlap_res_cf_contrib(i,j,k,resarr,csolarr,crhsarr,csigarr,
                                           cdmskarr,ndmskarr,ccmskarr,fcocarr,
                                           cdxinv,c_cc_domain_p,c_nd_domain,
                                           lobc,hibc, neumann_doubling);
                });
#endif
            } else {
                Real const_sigma = m_const_sigma;
#if (AMREX_SPACEDIM == 1)
                amrex::ignore_unused(const_sigma);
#elif (AMREX_SPACEDIM == 2)
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    mlndlap_res_cf_contrib_cs(i,j,k,resarr,csolarr,crhsarr,const_sigma,
                                              cdmskarr,ndmskarr,ccmskarr,fcocarr,
                                              cdxinv,c_cc_domain_p,c_nd_domain,
                                              is_rz,
                                              lobc,hibc, neumann_doubling);
                });
#elif (AMREX_SPACEDIM == 3)
                AMREX_HOST_DEVICE_FOR_3D(bx, i, j, k,
                {
                    mlndlap_res_cf_contrib_cs(i,j,k,resarr,csolarr,crhsarr,const_sigma,
                                              cdmskarr,ndmskarr,ccmskarr,fcocarr,
                                              cdxinv,c_cc_domain_p,c_nd_domain,
                                              lobc,hibc, neumann_doubling);
                });
#endif
            }
        }
    }
#ifdef AMREX_USE_EB
    // Make sure to zero out the residual on any nodes completely surrounded by covered cells
    amrex::EB_set_covered(res,0.0);
#endif
#endif
}

}
