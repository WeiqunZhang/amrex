#ifndef AMREX_GMRES_H_
#define AMREX_GMRES_H_

#include <AMReX_MultiFab.H>

namespace amrex {

class MLMG;

class GMRES
{
public:
    GMRES (MLMG& a_mlmg) : m_mlmg(a_mlmg) {}

    void solve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs,
                Real a_tol_rel, Real a_tol_abs);

    void setVerbose (int v) { m_verbose = v; }
    void setMaxOutIter (int n) { m_max_outer_iters = n; }
    void setMaxInnerIter (int n) { m_max_inner_iters = n; }
    void setPrecondIter (int n) { m_precond_iters = n; }

private:
    MLMG& m_mlmg;
    int m_max_inner_iters = 5;
    int m_max_outer_iters = 200;
    int m_precond_iters = 3;
    int m_verbose = 0;
};

}

#endif
