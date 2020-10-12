#include <AMReX_GMRES.H>
#include <AMReX_MLMG.H>

namespace amrex {

namespace {

void RotMat(Real a, Real b, Real& cs, Real& sn)
{
    if (b == 0.) {
        cs = 1.;
        sn = 0.;
    }
    else if (amrex::Math::abs(b) > amrex::Math::abs(a)) {
        Real temp = a/b;
        sn = 1./std::sqrt(1.+temp*temp);
        cs = temp*sn;
    }
    else {
        Real temp = b/a;
        cs = 1./std::sqrt(1.+temp*temp);
        sn = temp*cs;
    }
}

void LeastSquares(int i, Vector<Vector<Real> >& H, Vector<Real>& cs,
                  Vector<Real>& sn, Vector<Real>& s)
{
    // apply Givens rotation
    for (int k=0; k<=i-1; ++k) {
        Real temp =  cs[k]*H[k][i] + sn[k]*H[k+1][i];
        H[k+1][i] = -sn[k]*H[k][i] + cs[k]*H[k+1][i];
        H[k][i] = temp;
    }

    // form i-th rotation matrix
    RotMat(H[i][i], H[i+1][i], cs[i], sn[i]);

    // approximate residual norm
    Real temp = cs[i]*s[i];
    s[i+1] = -sn[i]*s[i];
    s[i] = temp;
    H[i][i] = cs[i]*H[i][i] + sn[i]*H[i+1][i];
    H[i+1][i] = 0.;
}

void SolveUTriangular(int k, Vector<Vector<Real> >& H, Vector<Real>& s, Vector<Real>& y)
{
    y[k+1] = s[k+1]/H[k+1][k+1];
    for (int i=k; i>=0; --i) {
        Real dot = 0.;
        for (int j=i+1; j<= k+1; ++j) {
            dot += H[i][j]*y[j];
        }
        y[i] = (s[i] - dot) / H[i][i];
    }
}

}

void
GMRES::solve (const Vector<MultiFab*>& a_sol, const Vector<MultiFab const*>& a_rhs,
              Real a_tol_rel, Real a_tol_abs)
{
    // TODO: fine level solve w/ coarse fine bc
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_sol.size() == 1, "GMRES: multilevel not supported");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_sol[0]->is_cell_centered(), "GMRES: nodal not supported");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mlmg.linop.getNComp() == 1,
                                     "GMRES: mulitcomponent not supported");
#ifdef AMREX_USE_EB
    amrex::Abort("EB is not support in GMRES");
#endif

    MultiFab& x = *a_sol[0];
    MultiFab const& b = *a_rhs[0];
    auto& linop = m_mlmg.linop;

    std::string old_print_identation = m_mlmg.print_identation;
    m_mlmg.print_identation = "   ";

    int old_fixed_iter = m_mlmg.do_fixed_number_of_iters;
    m_mlmg.do_fixed_number_of_iters = m_precond_iters;

    Vector<MultiFab> v(m_max_inner_iters+1);
    for (auto& mf : v) {
        mf.define(x.boxArray(), x.DistributionMap(), 1, 1);
    }
    MultiFab r(x.boxArray(), x.DistributionMap(), 1, 0);

    Vector<Vector<Real> > h(m_max_inner_iters+1, Vector<Real>(m_max_inner_iters));
    Vector<Real> cs(m_max_inner_iters  );
    Vector<Real> sn(m_max_inner_iters  );
    Vector<Real>  y(m_max_inner_iters  );
    Vector<Real>  s(m_max_inner_iters+1);

    m_mlmg.compResidual({&r}, {&x}, {&b});
    Real r_infnorm_init = r.norminf(0, 0, true);
    Real b_infnorm_init = b.norminf(0, 0, true);
    ParallelAllReduce::Max<Real>({r_infnorm_init, b_infnorm_init},
                                 ParallelContext::CommunicatorSub());

    Real max_infnorm_init;
    std::string norm_name;
    if (m_mlmg.always_use_bnorm or b_infnorm_init >= r_infnorm_init) {
        norm_name = "rhs";
        max_infnorm_init = b_infnorm_init;
    } else {
        norm_name = "resid0";
        max_infnorm_init = r_infnorm_init;
    }
    const Real r_infnorm_target = std::max(a_tol_abs,
                                           std::max(a_tol_rel,Real(1.e-16))*max_infnorm_init);

    if (m_verbose >= 1) amrex::Print() << "\n";

    if (r_infnorm_init <= r_infnorm_target) {
        if (m_verbose >= 1) {
            amrex::Print() << "GMRES: No iterations needed\n\n";
        }
        return;
    }

    Real max_2norm_init;
    if (norm_name == "rhs") {
        max_2norm_init = std::sqrt(linop.xdoty(0, 0, b, b, false));
        if (m_verbose >= 1) {
            amrex::Print() << "GMRES: Initial 2-norm of rhs = " << max_2norm_init << "\n\n";
        }
    } else {
        max_2norm_init = std::sqrt(linop.xdoty(0, 0, r, r, false));
        if (m_verbose >= 1) {
            amrex::Print() << "GMRES: Initial 2-norm of residual = " << max_2norm_init << "\n\n";
        }
    }

    for (int o_iter = 0; o_iter < m_max_outer_iters; ++o_iter)
    {
        if (m_verbose >= 2) amrex::Print() << "GMRES Iter. " << o_iter+1 << " ...\n";

        linop.switchToCorrectionMode();
        v[0].setVal(0.0);
        m_mlmg.solve({&v[0]}, {&r}, a_tol_rel, r_infnorm_target);

        s[0] = std::sqrt(linop.xdoty(0, 0, v[0], v[0], false));
        v[0].mult(1./s[0], 0, 1);

        int i_copy = 0;
        for (int i = 0; i < m_max_inner_iters; ++i)
        {
            i_copy = i;
            m_mlmg.apply({&r}, {&v[i]});
            v[i+1].setVal(0.0);
            m_mlmg.solve({&v[i+1]}, {&r}, a_tol_rel, r_infnorm_target);
            for (int k = 0; k <= i; ++k) {
                h[k][i] = linop.xdoty(0, 0, v[i+1], v[k], false);
                MultiFab::Saxpy(v[i+1], -h[k][i], v[k], 0, 0, 1, 0);
            }
            h[i+1][i] = std::sqrt(linop.xdoty(0, 0, v[i+1], v[i+1], false));
            v[i+1].mult(1./h[i+1][i]);

            // solve least squre problem
            LeastSquares(i, h, cs, sn, s);
            Real error = std::abs(s[i+1]);

            if (m_verbose >=2 ) {
                amrex::Print() << "GMRES Iter. (" << o_iter+1 << ", " << i+1 << "): 2-norm eror/"
                               << norm_name << " = " << error/max_2norm_init << "\n";
            }
            if (error <= a_tol_rel * max_2norm_init) {
                break;
            }
        }

        // Solve for y
        SolveUTriangular(i_copy-1, h, s, y);

        // x += v * y
        for (int k = 0; k <= i_copy; ++k) {
            MultiFab::Saxpy(x, y[k], v[k], 0, 0, 1, 0);
        }

        linop.switchToOriginalMode();
        m_mlmg.compResidual({&r}, {&x}, {&b});
        Real r_infnorm = r.norminf(0, 0, false);
        if (m_verbose >= 2) {
            amrex::Print() << "GMRES Iter. " << o_iter+1 << ": inf-norm resid, resid/"
                           << norm_name << " = " << r_infnorm << ", "
                           << r_infnorm/r_infnorm_init << "\n";
        }
        if (m_verbose >= 3) {
            Real r_2norm = std::sqrt(linop.xdoty(0, 0, r, r, false));
            amrex::Print() << "GMRES Iter. " << o_iter+1 << ":   2-norm resid, resid/"
                           << norm_name << " = " << r_2norm << ", "
                           << r_2norm/max_2norm_init << "\n";
        }
        if (m_verbose >= 2) amrex::Print() << "\n";
        if (r_infnorm <= r_infnorm_target) {
            if (m_verbose >= 1) {
                amrex::Print() << "GMRES Final Iter. " << o_iter+1 << ": "
                               << " inf-norm resid, resid/"
                               << norm_name << " = " << r_infnorm << ", "
                               << r_infnorm/r_infnorm_init << "\n\n";
            }
            break;
        }
    }

    m_mlmg.print_identation = old_print_identation;
    m_mlmg.do_fixed_number_of_iters = old_fixed_iter;
}

}
