#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_WriteEBSurface.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace amrex;

// 3 rows of 7 boxes, each [x0,x0+0.1] x [y0,y0+0.15] x [-0.125,0.125], given
// either by cubes.stl (with eb2.stl_scale = 1e-3) or by an implicit function.
// Several box faces lie on grid planes.
namespace {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void box_bounds (int ib, int jb, Real lo[3], Real hi[3])
    {
        lo[0] = -0.85_rt + static_cast<Real>(ib)*0.2_rt;
        lo[1] = -0.325_rt + static_cast<Real>(jb)*0.25_rt;
        lo[2] = -0.125_rt;
        hi[0] = lo[0] + 0.1_rt;
        hi[1] = lo[1] + 0.15_rt;
        hi[2] = 0.125_rt;
    }

    // Signed distance to the boxes, positive inside (the EB level set convention)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real box_signed_distance (Real x, Real y, Real z)
    {
        Real const p[3] = {x, y, z};
        Real dmin = std::numeric_limits<Real>::max();
        for (int jb = 0; jb < 3; ++jb) {
            for (int ib = 0; ib < 7; ++ib) {
                Real lo[3], hi[3];
                box_bounds(ib, jb, lo, hi);
                Real dout2 = 0.0_rt;
                Real din = std::numeric_limits<Real>::max();
                for (int d = 0; d < 3; ++d) {
                    Real const q = amrex::max(lo[d]-p[d], p[d]-hi[d]);
                    if (q > 0.0_rt) { dout2 += q*q; }
                    din = amrex::min(din, -q);
                }
                if (dout2 == 0.0_rt) { return din; } // inside this box
                dmin = amrex::min(dmin, std::sqrt(dout2));
            }
        }
        return -dmin;
    }

    // The boxes as an implicit function, positive inside
    struct BoxGridIF : GPUable
    {
        AMREX_GPU_HOST_DEVICE
        Real operator() (AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
        {
            return box_signed_distance(x, y, z);
        }

        Real operator() (RealArray const& p) const noexcept
        {
            return box_signed_distance(p[0], p[1], p[2]);
        }
    };

    // Exact fluid volume fraction of the cell [lo,hi]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real exact_volfrac (Real const clo[3], Real const chi[3])
    {
        Real covered = 0.0_rt;
        for (int jb = 0; jb < 3; ++jb) {
            for (int ib = 0; ib < 7; ++ib) {
                Real lo[3], hi[3];
                box_bounds(ib, jb, lo, hi);
                Real v = 1.0_rt;
                for (int d = 0; d < 3; ++d) {
                    v *= amrex::max(0.0_rt, amrex::min(chi[d],hi[d]) - amrex::max(clo[d],lo[d]))
                        / (chi[d]-clo[d]);
                }
                covered += v;
            }
        }
        return 1.0_rt - covered;
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        Array<int,3> n_cell{256, 128, 64};
        pp.query("n_cell", n_cell);
        int max_grid_size = 32;
        pp.query("max_grid_size", max_grid_size);
        int write_surface = 0;
        pp.query("write_surface", write_surface);
        int verbose = 0;
        pp.query("verbose", verbose);
        std::string geometry = "stl"; // or "boxes" for the implicit function
        pp.query("geometry", geometry);
        Real max_vf_err_tol = 0.75_rt;
        pp.query("max_volfrac_error", max_vf_err_tol);

        Box const domain(IntVect(0), IntVect(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
        Geometry const geom(domain); // reads geometry.prob_lo/hi
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping const dm(ba);

        if (geometry == "boxes") {
            EB2::Build(EB2::makeShop(BoxGridIF{}), geom, 0, 0);
        } else {
            EB2::Build(geom, 0, 0); // eb2.geom_type = stl
        }

        auto factory = makeEBFabFactory(geom, ba, dm, {1,1,1}, EBSupport::full);
        auto const& flags = factory->getMultiEBCellFlagFab();
        auto const& bcent = factory->getBndryCent();
        auto const& vfrac = factory->getVolFrac();
        auto const& levset = factory->getLevelSet();

        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();
        auto const fa = flags.const_arrays();
        auto const bc_a = bcent.const_arrays();
        auto const vf_a = vfrac.const_arrays();
        auto const ls_a = levset.const_arrays();

        // Largest distance from an EB face centroid to the true surface, and
        // largest error in the volume fraction.
        auto r = ParReduce(TypeList<ReduceOpMax,ReduceOpMax,ReduceOpSum>{},
                           TypeList<Real,Real,Long>{}, flags, IntVect(0),
            [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) -> GpuTuple<Real,Real,Long>
        {
            Real const clo[3] = {problo[0] + static_cast<Real>(i)*dx[0],
                                 problo[1] + static_cast<Real>(j)*dx[1],
                                 problo[2] + static_cast<Real>(k)*dx[2]};
            Real const chi[3] = {clo[0]+dx[0], clo[1]+dx[1], clo[2]+dx[2]};
            Real const vf_err = std::abs(vf_a[b](i,j,k) - exact_volfrac(clo, chi));
            if (vf_err > max_vf_err_tol && verbose) {
                AMREX_DEVICE_PRINTF("bad cell (%d,%d,%d) volfrac %g error %g\n",
                                    i, j, k, double(vf_a[b](i,j,k)), double(vf_err));
            }
            Real dist = 0.0_rt;
            if (fa[b](i,j,k).isSingleValued()) {
                Real const x = clo[0] + (0.5_rt+bc_a[b](i,j,k,0))*dx[0];
                Real const y = clo[1] + (0.5_rt+bc_a[b](i,j,k,1))*dx[1];
                Real const z = clo[2] + (0.5_rt+bc_a[b](i,j,k,2))*dx[2];
                dist = std::abs(box_signed_distance(x,y,z)) / dx[0];
            }
            return {dist, vf_err, Long(fa[b](i,j,k).isSingleValued())};
        });
        Real max_dist = amrex::get<0>(r);
        Real max_vf_err = amrex::get<1>(r);
        Long ncut = amrex::get<2>(r);
        ParallelDescriptor::ReduceRealMax({max_dist, max_vf_err});
        ParallelDescriptor::ReduceLongSum(ncut);

        // Nodes clearly off the surface must be on the correct side.
        Long nbad_nodes = ParReduce(TypeList<ReduceOpSum>{}, TypeList<Long>{}, levset, IntVect(0),
            [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) -> GpuTuple<Long>
        {
            Real const sd = box_signed_distance(problo[0] + static_cast<Real>(i)*dx[0],
                                                problo[1] + static_cast<Real>(j)*dx[1],
                                                problo[2] + static_cast<Real>(k)*dx[2]);
            bool const bad = std::abs(sd) > 0.1_rt*dx[0] && (sd > 0.0_rt) != (ls_a[b](i,j,k) > 0.0_rt);
            if (bad && verbose) {
                AMREX_DEVICE_PRINTF("bad node (%d,%d,%d) levelset %g distance %g\n",
                                    i, j, k, double(ls_a[b](i,j,k)), double(sd));
            }
            return {Long(bad)};
        });
        ParallelDescriptor::ReduceLongSum(nbad_nodes);

        amrex::Print() << "  number of cut cells:   " << ncut << "\n"
                       << "  max distance / dx:     " << max_dist << "\n"
                       << "  max volfrac error:     " << max_vf_err << "\n"
                       << "  misclassified nodes:   " << nbad_nodes << "\n";

        if (write_surface) {
            WriteEBSurface(ba, dm, geom, factory.get());
        }

        if (max_dist > 1.0_rt || max_vf_err > max_vf_err_tol || nbad_nodes > 0) {
            amrex::Abort("GridAlignedBoxes: EB does not match the geometry");
        }
    }
    amrex::Finalize();
}
