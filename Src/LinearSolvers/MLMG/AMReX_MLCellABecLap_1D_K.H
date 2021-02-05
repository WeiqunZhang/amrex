#ifndef AMREX_MLCELLABECLAP_1D_K_H_
#define AMREX_MLCELLABECLAP_1D_K_H_
#include <AMReX_Config.H>

namespace amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuTuple<unsigned long long, unsigned long long>
coarsen_overset_mask (Box const& bx, Array4<int> const& cmsk, Array4<int const> const& fmsk)
    noexcept
{
    unsigned long long nmixed = 0;
    unsigned long long nvalid = 0;
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    for (int i = lo.x; i <= hi.x; ++i) {
        int ii = 2*i;
        cmsk(i,0,0) = fmsk(ii,0,0) + fmsk(ii+1,0,0);
        if (cmsk(i,0,0) == 2) {
            cmsk(i,0,0) = 1;
            ++nvalid;
        } else if (cmsk(i,0,0) != 0) {
            cmsk(i,0,0) = 0;
            ++nmixed;
        }
    }
    return GpuTuple<unsigned long long, unsigned long long>{nmixed,nvalid};
}

}

#endif
