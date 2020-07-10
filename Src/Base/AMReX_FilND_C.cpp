#include <AMReX_FilND_C.H>

namespace amrex {

void fab_filnd (Box const& bx, Array4<Real> const& qn, int ncomp,
                Box const& domain, Real const* /*dx*/, Real const* /*xlo*/,
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

        if (lo.x < domlo.x && (bc.lo(0) != BCType::int_dir)) {
           const int imin = lo.x;
           const int imax = domlo.x-1;
           for (int k = lo.z; k <= hi.z; ++k) {
           for (int j = lo.y; j <= hi.y; ++j) {
           for (int i = imin; i <= imax; ++i) {
               q(i,j,k) = q(domlo.x,j,k);
           }}}
        }

        if (hi.x > domhi.x && (bc.hi(0) != BCType::int_dir)) {
            const int imin = domhi.x+1;
            const int imax = hi.x;
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = imin; i <= imax; ++i) {
                q(i,j,k) = q(domhi.x,j,k);
            }}}
        }

#if AMREX_SPACEDIM >= 2

        if (lo.y < domlo.y && (bc.lo(1) != BCType::int_dir)) {
            const int jmin = lo.y;
            const int jmax = domlo.y-1;
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = jmin; j <= jmax; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                q(i,j,k) = q(i,domlo.y,k);
            }}}
        }

        if (hi.y > domhi.y && (bc.hi(1) != BCType::int_dir)) {
            const int jmin = domhi.y+1;
            const int jmax = hi.y;
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = jmin; j <= jmax; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                q(i,j,k) = q(i,domhi.y,k);
            }}}
        }
#endif

#if AMREX_SPACEDIM == 3

        if (lo.z < domlo.z && (bc.lo(2) != BCType::int_dir)) {
            const int kmin = lo.z;
            const int kmax = domlo.z-1;
            for (int k = kmin; k <= kmax; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                q(i,j,k) = q(i,j,domlo.z);
            }}}
        }

        if (hi.z > domhi.z && (bc.hi(2) != BCType::int_dir)) {
            const int kmin = domhi.z+1;
            const int kmax = hi.z;
            for (int k = kmin; k <= kmax; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                q(i,j,k) = q(i,j,domhi.z);
            }}}
        }
#endif
    }
}

}
