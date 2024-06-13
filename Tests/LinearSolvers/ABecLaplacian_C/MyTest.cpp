#include "MyTest.H"

//#include <AMReX_GMRES.H>
//#include <AMReX_GMRES_MLMG.H>
//#include <AMReX_MLNodeABecLaplacian.H>
//#include <AMReX_MLABecLaplacian.H>
//#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

#ifdef AMREX_USE_HYPRE
#include <AMReX_HypreMLABecLap.H>
#endif

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::solve ()
{
    solveMLHypre();
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("ref_ratio", ref_ratio);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.query("composite_solve", composite_solve);

    pp.query("use_mlhypre", use_mlhypre);

    pp.query("prob_type", prob_type);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("linop_maxorder", linop_maxorder);
    pp.query("agglomeration", agglomeration);
    pp.query("consolidation", consolidation);
    pp.query("semicoarsening", semicoarsening);
    pp.query("max_coarsening_level", max_coarsening_level);
    pp.query("max_semicoarsening_level", max_semicoarsening_level);

    pp.query("use_gmres", use_gmres);
    AMREX_ALWAYS_ASSERT(use_gmres == false || prob_type == 2);

#ifdef AMREX_USE_HYPRE
    pp.query("use_hypre", use_hypre);
#endif
#ifdef AMREX_USE_PETSC
    pp.query("use_petsc", use_petsc);
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(use_hypre && use_petsc),
                                     "use_hypre & use_petsc cannot be both true");
}

void
MyTest::initData ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);
    dmap.resize(nlevels);

    solution.resize(nlevels);
    rhs.resize(nlevels);
    exact_solution.resize(nlevels);

    if (prob_type == 2 || prob_type == 3 || prob_type == 4) {
        acoef.resize(nlevels);
        bcoef.resize(nlevels);
    }

    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(ref_ratio);
    }

    domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        grids[ilev].define(domain);
        grids[ilev].maxSize(max_grid_size);
        domain.grow(-n_cell/4);   // fine level cover the middle of the coarse domain
        domain.refine(ref_ratio);
    }

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        BoxArray ba = grids[ilev];
        if (prob_type == 4) {
            ba.surroundingNodes();
        }
        solution      [ilev].define(ba, dmap[ilev], 1, 1);
        rhs           [ilev].define(ba, dmap[ilev], 1, 0);
        exact_solution[ilev].define(ba, dmap[ilev], 1, 0);
        if (!acoef.empty()) {
            acoef[ilev].define(ba         , dmap[ilev], 1, 0);
            const int ngb = (prob_type == 4) ? 0 : 1;
            bcoef[ilev].define(grids[ilev], dmap[ilev], 1, ngb);
        }
    }

    if (prob_type == 1) {
        initProbPoisson();
    } else if (prob_type == 2) {
        initProbABecLaplacian();
    } else {
        amrex::Abort("Unknown prob_type "+std::to_string(prob_type));
    }
}

#ifdef AMREX_USE_HYPRE
void
MyTest::solveMLHypre ()
{
    const auto tol_rel = Real(1.e-10);
    const auto tol_abs = Real(0.0);

    const auto nlevels = static_cast<int>(geom.size());

    if (prob_type == 1) { // Poisson
        if (composite_solve) {
            HypreMLABecLap hypre_mlabeclap(geom, grids, dmap);
            hypre_mlabeclap.setVerbose(verbose);

            hypre_mlabeclap.setup(Real(0.0), Real(-1.0), {}, {},
                                  {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                LinOpBCType::Dirichlet,
                                                LinOpBCType::Dirichlet)},
                                  {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                LinOpBCType::Dirichlet,
                                                LinOpBCType::Dirichlet)},
                                  GetVecOfConstPtrs(solution));

            hypre_mlabeclap.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs),
                                  tol_rel, tol_abs);
        } else {
            for (int ilev = 0; ilev < nlevels; ++ilev) {
                HypreMLABecLap hypre_mlabeclap({geom[ilev]}, {grids[ilev]}, {dmap[ilev]});
                hypre_mlabeclap.setVerbose(verbose);

                std::pair<MultiFab const*, IntVect> coarse_bc{nullptr,IntVect(0)};
                if (ilev > 0) {
                    coarse_bc.first = &solution[ilev-1];
                    coarse_bc.second = IntVect(ref_ratio);
                }

                hypre_mlabeclap.setup(Real(0.0), Real(-1.0), {}, {},
                                      {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet)},
                                      {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet,
                                                    LinOpBCType::Dirichlet)},
                                      {&solution[ilev]},
                                      coarse_bc);

                hypre_mlabeclap.solve({&solution[ilev]}, {&rhs[ilev]}, tol_rel, tol_abs);
            }
        }
    } else if (prob_type == 2) { // ABecLaplacian
        Vector<Array<MultiFab,AMREX_SPACEDIM>> face_bcoef(nlevels);
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                const BoxArray& ba = amrex::convert(bcoef[ilev].boxArray(),
                                                    IntVect::TheDimensionVector(idim));
                face_bcoef[ilev][idim].define(ba, bcoef[ilev].DistributionMap(), 1, 0);
            }
            amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoef[ilev]),
                                              bcoef[ilev], geom[ilev]);
        }

        if (composite_solve) {
            HypreMLABecLap hypre_mlabeclap(geom, grids, dmap);
            hypre_mlabeclap.setVerbose(verbose);

            hypre_mlabeclap.setup(ascalar, bscalar,
                                  GetVecOfConstPtrs(acoef),
                                  GetVecOfArrOfConstPtrs(face_bcoef),
                                  {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                LinOpBCType::Neumann,
                                                LinOpBCType::Neumann)},
                                  {AMREX_D_DECL(LinOpBCType::Neumann,
                                                LinOpBCType::Dirichlet,
                                                LinOpBCType::Neumann)},
                                  GetVecOfConstPtrs(solution));

            hypre_mlabeclap.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs),
                                  tol_rel, tol_abs);
        } else {
            for (int ilev = 0; ilev < nlevels; ++ilev) {
                HypreMLABecLap hypre_mlabeclap({geom[ilev]}, {grids[ilev]}, {dmap[ilev]});
                hypre_mlabeclap.setVerbose(verbose);

                std::pair<MultiFab const*, IntVect> coarse_bc{nullptr,IntVect(0)};
                if (ilev > 0) {
                    coarse_bc.first = &solution[ilev-1];
                    coarse_bc.second = IntVect(ref_ratio);
                }

                hypre_mlabeclap.setup(ascalar, bscalar,
                                      {&acoef[ilev]},
                                      {GetArrOfConstPtrs(face_bcoef[ilev])},
                                      {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                    LinOpBCType::Neumann,
                                                    LinOpBCType::Neumann)},
                                      {AMREX_D_DECL(LinOpBCType::Neumann,
                                                    LinOpBCType::Dirichlet,
                                                    LinOpBCType::Neumann)},
                                      {&solution[ilev]},
                                      coarse_bc);

                hypre_mlabeclap.solve({&solution[ilev]}, {&rhs[ilev]}, tol_rel, tol_abs);
            }
        }
    } else {
        amrex::Abort("Unsupported prob_type: " + std::to_string(prob_type));
    }
}
#endif
