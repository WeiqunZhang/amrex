#include <AMReX_FillPatchUtil.H>

namespace amrex
{
    void FillPatchInterp (MultiFab& mf_fine_patch, int fcomp,
                          MultiFab const& mf_crse_patch, int ccomp,
                          int ncomp, IntVect const& ng,
                          const Geometry& cgeom, const Geometry& fgeom,
                          Box const& dest_domain, const IntVect& ratio,
                          MFInterpolater* mapper, const Vector<BCRec>& bcs, int bcscomp)
    {
        BL_PROFILE("FillPatchInterp(MF)");
        mapper->interp(mf_crse_patch, ccomp, mf_fine_patch, fcomp, ncomp, ng, cgeom, fgeom,
                       dest_domain, ratio, bcs, bcscomp);
    }
}
