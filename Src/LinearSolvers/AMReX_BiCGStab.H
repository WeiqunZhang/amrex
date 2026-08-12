#ifndef AMREX_BICGSTAB_H_
#define AMREX_BICGSTAB_H_
#include <AMReX_Config.H>

#include <AMReX_BLProfiler.H>

#include <limits>

namespace amrex {

template <typename V, typename M>
class BiCGStab
{
public:

    using RT = typename M::RT; // double or float

    BiCGStab ();

    //! Defines with a reference to M. It's the user's responsibility to
    //! keep the M object alive for BiCStab to be functional. This function
    //! must be called before solve() can be called.
    void define (M& linop);

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    void solve (V& a_sol, V const& a_rhs, RT a_tol_rel, RT a_tol_abs, int a_its=-1);

    //! Sets verbosity.
    void setVerbose (int v) { m_verbose = v; }

    //! Sets the max number of iterations
    void setMaxIters (int niters) { m_maxiter = niters; }

    //! Gets the number of iterations.
    [[nodiscard]] int getNumIters () const { return m_its; }

    //! Gets the solver status.
    [[nodiscard]] int getStatus () const { return m_status; }

    //! Gets the 2-norm of the residual.
    [[nodiscard]] RT getResidualNorm () const { return m_res; }

private:

    int m_verbose = 0;
    int m_maxiter = 2000;
    int m_its = 0;
    int m_status = -1;
    RT m_res = std::numeric_limits<RT>::max();
    RT m_rtol = RT(0);
    RT m_atol = RT(0);

    M* m_linop = nullptr;
};

}

#endif
