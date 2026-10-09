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

using namespace amrex;

// cubes.stl is 3 rows of 7 boxes, each [x0,x0+0.1] x [y0,y0+0.15] x [-0.125,0.125]
// after eb2.stl_scale = 1e-3.  Several box faces lie on grid planes.
namespace {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real box_surface_distance (Real x, Real y, Real z)
    {
        Real dmin = std::numeric_limits<Real>::max();
        for (int jb = 0; jb < 3; ++jb) {
            Real const ylo = -0.325_rt + static_cast<Real>(jb)*0.25_rt;
            for (int ib = 0; ib < 7; ++ib) {
                Real const xlo = -0.85_rt + static_cast<Real>(ib)*0.2_rt;
                Real const lo[3] = {xlo, ylo, -0.125_rt};
                Real const hi[3] = {xlo+0.1_rt, ylo+0.15_rt, 0.125_rt};
                Real const p[3] = {x, y, z};
                Real dout2 = 0.0_rt;
                Real din = std::numeric_limits<Real>::max();
                for (int d = 0; d < 3; ++d) {
                    Real const q = amrex::max(lo[d]-p[d], p[d]-hi[d]);
                    if (q > 0.0_rt) { dout2 += q*q; }
                    din = amrex::min(din, -q);
                }
                Real const dist = (dout2 > 0.0_rt) ? std::sqrt(dout2) : amrex::max(din,0.0_rt);
                dmin = amrex::min(dmin, dist);
            }
        }
        return dmin;
    }

    // Exact aperture of the face normal to dir at node coordinate xn, with
    // transverse extents [a0,a1] x [b0,b1] (dirs (dir+1)%3, (dir+2)%3).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real exact_aperture (int dir, Real xn, Real a0, Real a1, Real b0, Real b1)
    {
        Real covered = 0.0_rt;
        Real const tol = 1.e-5_rt;
        for (int jb = 0; jb < 3; ++jb) {
            Real const ylo = -0.325_rt + static_cast<Real>(jb)*0.25_rt;
            for (int ib = 0; ib < 7; ++ib) {
                Real const xlo = -0.85_rt + static_cast<Real>(ib)*0.2_rt;
                Real const lo[3] = {xlo, ylo, -0.125_rt};
                Real const hi[3] = {xlo+0.1_rt, ylo+0.15_rt, 0.125_rt};
                int const da = (dir+1)%3;
                int const db = (dir+2)%3;
                if (xn >= lo[dir]-tol && xn <= hi[dir]+tol) {
                    Real const la = amrex::max(0.0_rt, amrex::min(a1,hi[da])-amrex::max(a0,lo[da]));
                    Real const lb = amrex::max(0.0_rt, amrex::min(b1,hi[db])-amrex::max(b0,lo[db]));
                    covered += la*lb;
                }
            }
        }
        return 1.0_rt - covered/((a1-a0)*(b1-b0));
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

        Box const domain(IntVect(0), IntVect(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
        Geometry const geom(domain); // reads geometry.prob_lo/hi
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping const dm(ba);

        EB2::Build(geom, 0, 0);

        auto factory = makeEBFabFactory(geom, ba, dm, {1,1,1}, EBSupport::full);
        auto const& flags = factory->getMultiEBCellFlagFab();
        auto const& bcent = factory->getBndryCent();
        auto const& vfrac = factory->getVolFrac();

        auto const dx = geom.CellSizeArray();
        auto const problo = geom.ProbLoArray();
        auto const fa = flags.const_arrays();
        auto const bc_a = bcent.const_arrays();

        // Largest distance from an EB face centroid to the true surface, and
        // number of cut cells whose EB face is more than dx away.
        auto r = ParReduce(TypeList<ReduceOpMax,ReduceOpSum,ReduceOpSum>{},
                           TypeList<Real,Long,Long>{}, flags, IntVect(0),
            [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) -> GpuTuple<Real,Long,Long>
        {
            if (fa[b](i,j,k).isSingleValued()) {
                Real const x = problo[0] + (static_cast<Real>(i)+0.5_rt+bc_a[b](i,j,k,0))*dx[0];
                Real const y = problo[1] + (static_cast<Real>(j)+0.5_rt+bc_a[b](i,j,k,1))*dx[1];
                Real const z = problo[2] + (static_cast<Real>(k)+0.5_rt+bc_a[b](i,j,k,2))*dx[2];
                Real const dist = box_surface_distance(x,y,z) / dx[0];
                return {dist, Long(1), Long(dist > 1.0_rt)};
            } else {
                return {0.0_rt, Long(0), Long(0)};
            }
        });
        Real max_dist = amrex::get<0>(r);
        Long ncut = amrex::get<1>(r);
        Long nbad = amrex::get<2>(r);
        ParallelDescriptor::ReduceRealMax(max_dist);
        ParallelDescriptor::ReduceLongSum(ncut);
        ParallelDescriptor::ReduceLongSum(nbad);

        // Faces whose aperture is far from the exact one.  A fin (a zero-thickness
        // wall in the fluid) has aperture 0 where the exact one is 1.
        Real max_aperture_error = 0.0_rt;
        Long nbad_faces = 0;
        for (int dir = 0; dir < 3; ++dir) {
            MultiFab const apmf = factory->getAreaFrac()[dir]->ToMultiFab(1.0_rt, 0.0_rt);
            auto const ap = apmf.const_arrays();
            auto rf = ParReduce(TypeList<ReduceOpMax,ReduceOpSum>{}, TypeList<Real,Long>{},
                                apmf, IntVect(0),
                [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) -> GpuTuple<Real,Long>
            {
                IntVect const iv(i,j,k);
                Real const xn = problo[dir] + static_cast<Real>(iv[dir])*dx[dir];
                int const da = (dir+1)%3;
                int const db = (dir+2)%3;
                Real const a0 = problo[da] + static_cast<Real>(iv[da])*dx[da];
                Real const b0 = problo[db] + static_cast<Real>(iv[db])*dx[db];
                Real const err = std::abs(ap[b](i,j,k) - exact_aperture(dir, xn, a0, a0+dx[da],
                                                                        b0, b0+dx[db]));
                if (err > 0.75_rt && verbose) {
                    AMREX_DEVICE_PRINTF("bad face dir %d (%d,%d,%d) aperture %g error %g\n",
                                        dir, i, j, k, ap[b](i,j,k), err);
                }
                return {err, Long(err > 0.75_rt)};
            });
            max_aperture_error = amrex::max(max_aperture_error, amrex::get<0>(rf));
            nbad_faces += amrex::get<1>(rf);
        }
        ParallelDescriptor::ReduceRealMax(max_aperture_error);
        ParallelDescriptor::ReduceLongSum(nbad_faces);

        Real const covered_volume = geom.ProbDomain().volume() - vfrac.sum()*dx[0]*dx[1]*dx[2];

        // 21 boxes of 0.1 x 0.15 x 0.125 above z = 0
        Real const exact_volume = 21.0_rt*0.1_rt*0.15_rt*0.125_rt;

        amrex::Print() << "  number of cut cells:      " << ncut << "\n"
                       << "  EB faces off the surface: " << nbad << "\n"
                       << "  max distance / dx:        " << max_dist << "\n"
                       << "  bad faces:                " << nbad_faces << "\n"
                       << "  max aperture error:       " << max_aperture_error << "\n"
                       << "  covered volume:           " << covered_volume
                       << " (exact " << exact_volume << ")\n";

        if (write_surface) {
            WriteEBSurface(ba, dm, geom, factory.get());
        }

        if (nbad > 0 || nbad_faces > 0) {
            amrex::Abort("STLCubes: EB faces found away from the STL surface");
        }
    }
    amrex::Finalize();
}
