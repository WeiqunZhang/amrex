#include <AMReX_FilFC_C.H>

namespace amrex {

void fab_filfc (Box const& bx, Array4<Real> const& qn, int ncomp,
                Box const& domain, Real const* dx, Real const* xlo,
                BCRec const* bcn)
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    const auto domlo = amrex::lbound(domain);
    const auto domhi = amrex::ubound(domain);

    for (int n = 0; n < ncomp; ++n)
    {
        Array4<Real> q(qn,n);
        BCRec const& bc = bcn[n];

        if (lo.x < domlo.x) {
            const int imin = lo.x;
            const int imax = domlo.x-1;
            if (bc.lo(0) == BCType::reflect_odd || bc.lo(0) == BCType::reflect_even) {
                int ii = (bx.type(0) == IndexType::NODE) ? 2*domlo.x : 2*domlo.x-1;
                Real s = (bc.lo(0) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = s*q(ii-i,j,k);
                }}}
            } else if (bc.lo(0) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = q(domlo.x,j,k);
                }}}
            }
        }
        if (hi.x > domhi.x) {
            const int imin = domhi.x+1;
            const int imax = hi.x;
            if (bc.hi(0) == BCType::reflect_odd || bc.hi(0) == BCType::reflect_even) {
                int ii = (bx.type(0) == IndexType::NODE) ? 2*domhi.x : 2*domhi.x+1;
                Real s = (bc.hi(0) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = s*q(ii-i,j,k);
                }}}
            } else if (bc.hi(0) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    q(i,j,k) = q(domhi.x,j,k);
                }}}
            }
        }

#if (AMREX_SPACEDIM >= 2)
        if (lo.y < domlo.y) {
            const int jmin = lo.y;
            const int jmax = domlo.y-1;
            if (bc.lo(1) == BCType::reflect_odd || bc.lo(1) == BCType::reflect_even) {
                int jj = (bx.type(1) == IndexType::NODE) ? 2*domlo.y : 2*domlo.y-1;
                Real s = (bc.lo(1) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = s*q(i,jj-j,k);
                }}}
            } else if (bc.lo(1) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,domlo.y,k);
                }}}
            }
        }
        if (hi.y > domhi.y) {
            const int jmin = domhi.y+1;
            const int jmax = hi.y;
            if (bc.hi(1) == BCType::reflect_odd || bc.hi(1) == BCType::reflect_even) {
                int jj = (bx.type(1) == IndexType::NODE) ? 2*domhi.y : 2*domhi.y+1;
                Real s = (bc.hi(1) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = s*q(i,jj-j,k);
                }}}
            } else if (bc.hi(1) == BCType::foextrap) {
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = jmin; j <= jmax; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,domhi.y,k);
                }}}
            }
        }
#endif

#if (AMREX_SPACEDIM == 3)
        if (lo.z < domlo.z) {
            const int kmin = lo.z;
            const int kmax = domlo.z-1;
            if (bc.lo(2) == BCType::reflect_odd || bc.lo(2) == BCType::reflect_even) {
                int kk = (bx.type(2) == IndexType::NODE) ? 2*domlo.z : 2*domlo.z-1;
                Real s = (bc.lo(2) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = kmin; k <= kmax; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = s*q(i,j,kk-k);
                }}}
            } else if (bc.lo(2) == BCType::foextrap) {
                for (int k = kmin; k <= kmax; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,j,domlo.z);
                }}}
            }
        }
        if (hi.z > domhi.z) {
            const int kmin = domhi.z+1;
            const int kmax = hi.z;
            if (bc.hi(2) == BCType::reflect_odd || bc.hi(2) == BCType::reflect_even) {
                int kk = (bx.type(2) == IndexType::NODE) ? 2*domhi.z : 2*domhi.z+1;
                Real s = (bc.hi(2) == BCType::reflect_odd) ? -1.0 : 1.0;
                for (int k = kmin; k <= kmax; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = s*q(i,j,kk-k);
                }}}
            } else if (bc.hi(2) == BCType::foextrap) {
                for (int k = kmin; k <= kmax; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    q(i,j,k) = q(i,j,domhi.z);
                }}}
            }
        }
#endif
    }
}

}
