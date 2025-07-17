#include <AMReX_EB2_Level_STL.H>

namespace amrex::EB2 {

STLLevel::STLLevel (IndexSpace const* is, STLtools const& stl_tools, const Geometry& geom,
                    int max_grid_size, int ngrow, bool extend_domain_face, int num_crse_opt)
    : GShopLevel<STLtools>(is, geom)
{
    BL_PROFILE("EB2::STLLevel()-fine");

    define_fine(stl_tools, geom, max_grid_size, ngrow, extend_domain_face, num_crse_opt);


    // xxxxx STLLevel debug
    BoxArray ba(geom.Domain());
    ba.maxSize(32);
    DistributionMapping dm(ba);
    MultiFab mf(ba, dm, 1, 0);
    stl_tools.fillSignedDistance(mf, mf.nGrowVect(), geom);
}

STLLevel::STLLevel (IndexSpace const* is, int ilev, int max_grid_size, int ngrow,
                    const Geometry& geom, STLLevel& fineLevel)
    : GShopLevel<STLtools>(is, ilev, max_grid_size, ngrow, geom, fineLevel)
{}

}
