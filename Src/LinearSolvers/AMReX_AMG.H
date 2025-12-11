#ifndef AMREX_AMG_H_
#define AMREX_AMG_H_
#include <AMReX_Config.H>

#include <AMReX_Algebra.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Smoother_MV.H>
#include <AMReX_Vector.H>
#include <AMReX_VisMF.H>

#include <exception>
#include <limits>
#include <utility>

namespace amrex {

template <typename T>
class AMG
{
public:
    explicit AMG (SpMatrix<T>& a_mat);

    void define (SpMatrix<T>& a_mat);

    void solve (AlgVector<T>& a_sol, AlgVector<T> const& a_rhs);

private:

    bool m_prepared = false;
    bool m_throw_exception = false;
    int m_verbose = 2; // xxxxx change it to 0 in the future
    int m_maxiter = 200;
    int m_fixediter = 1; // xxxxx change it to 0 in the future

    // Number of smoothing
    int m_nu1 = 4;
    int m_nu2 = 4;

    // AMG parameter
    T m_strong_threshold = T(0.25);

    T m_reltol = std::is_same_v<T,float> ? T(1.e-4) : T(1.e-8);
    T m_init_rnorm = std::numeric_limits<T>::max();
    T m_rnorm = std::numeric_limits<T>::max();
    T m_bnorm = std::numeric_limits<T>::max();

    std::string m_print_ident;

    SpMatrix<T>* m_mat = nullptr;
    Vector<SpMatrix<T>> m_crsemat; // coarse MG level matrix
    Vector<SpMatrix<short>> m_soc;   // strength of connection (influencee)
    Vector<SpMatrix<short>> m_soc_t; // soc transposed (influencer)

    Vector<AlgVector<T>> m_res;    // rhs - L(sol)
    Vector<AlgVector<T>> m_cor;    // cor = L^-1 (res)
    Vector<AlgVector<T>> m_rescor; // res - L(cor)

    enum class PointType { undecided, fine, coarse };

    void vcycle ();

    void smooth (int mglev, AlgVector<T>& xvec, AlgVector<T> const& bvec, int nu,
                 bool with_initial_guess);

    static std::pair<T,T> compute_rbnorms (SpMatrix<T> const& mat,
                                           AlgVector<T> const& xvec,
                                           AlgVector<T> const& bvec,
                                           AlgVector<T>& rvec);

    static T compute_rnorm (SpMatrix<T> const& mat, AlgVector<T> const& xvec,
                            AlgVector<T> const& bvec, AlgVector<T>& rvec);

//Public for CUDA
public: // NOLINT

    void setup ();

    SpMatrix<short> create_soc (SpMatrix<T> const& mat);
};

template <typename T>
AMG<T>::AMG (SpMatrix<T>& a_mat)
{
    define(a_mat);
}

template <typename T>
void AMG<T>::define (SpMatrix<T>& a_mat)
{
    AMREX_ALWAYS_ASSERT(m_mat == nullptr);
    m_mat = &a_mat;
}

template <typename T>
void AMG<T>::solve (AlgVector<T>& a_sol, AlgVector<T> const& a_rhs)
{
    if (!m_prepared) { setup(); }

    auto norms = compute_rbnorms(*m_mat, a_sol, a_rhs, m_res[0]);
    m_bnorm = norms.first;
    m_init_rnorm = m_rnorm = norms.second;

    if (m_bnorm == 0) {
        if (m_rnorm != 0) { a_sol.setVal(0); }
        return;
    }

    if (m_verbose >= 1) {
        amrex::Print() << m_print_ident << "AMG: Initial rhs      2-norm = " << m_bnorm << "\n"
                       << m_print_ident << "AMG: Initial residual 2-norm = " << m_rnorm << "\n";
    }

    T res_target = m_bnorm * m_reltol;
    if (m_rnorm <= res_target) {
        if (m_verbose >= 1) {
            amrex::Print() << m_print_ident << "AMG: No iterations needed\n";
        }
        return;
    }

    bool converged = false;

    int const niters = (m_fixediter > 0) ? m_fixediter :  m_maxiter;
    for (int iter = 0; iter < niters; ++iter)
    {
        vcycle();
        a_sol.plusAsync(m_cor[0]);

        m_rnorm = compute_rnorm(*m_mat, a_sol, a_rhs, m_res[0]);

        if (m_verbose >= 2) {
            amrex::Print() << m_print_ident << "AMG: Iteration " << std::setw(3)
                           << iter+1 << ": rnorm, rnorm/bnorm = "
                           << m_rnorm << ", " << m_rnorm/m_bnorm << "\n";
        }

        converged = m_rnorm <= res_target;
        if (converged) {
            if (m_verbose >= 1) {
                amrex::Print() << m_print_ident << "AMG: Final Iter. " << iter+1
                               << ": rnorm, rnorm/bnorm = "
                               << m_rnorm << ", " << m_rnorm/m_bnorm << "\n";
            }
            break;
        } else {
            if (m_rnorm > T(1.e20) * m_init_rnorm) {
                if (m_verbose > 0) {
                    amrex::Print() << m_print_ident << "AMG: Failing to converge after " << iter+1 << " iterations."
                                   << " rnorm, rnorm/bnorm = "
                                   << m_rnorm << ", " << m_rnorm/m_bnorm << "\n";
                }
                if (m_throw_exception ) {
                    throw std::runtime_error("AMG blew up.");
                } else {
                    amrex::Abort("AMG failing so lets stop here");
                }
            }
        }
    }
}

template <typename T>
std::pair<T,T> AMG<T>::compute_rbnorms (SpMatrix<T> const& mat,
                                        AlgVector<T> const& xvec,
                                        AlgVector<T> const& bvec,
                                        AlgVector<T>& rvec)
{
    auto bnorm = bvec.norm2(true);

    amrex::computeResidual(rvec, mat, xvec, bvec);
    auto rnorm = rvec.norm2(true);

    if (ParallelContext::NProcsSub() > 1) {
        bnorm = Math::powi<2>(bnorm);
        rnorm = Math::powi<2>(rnorm);
        ParallelAllReduce::Sum<T>({bnorm,rnorm}, ParallelContext::CommunicatorSub());
        bnorm = std::sqrt(bnorm);
        rnorm = std::sqrt(rnorm);
    }

    return {bnorm, rnorm};
}

template <typename T>
T AMG<T>::compute_rnorm (SpMatrix<T> const& mat, AlgVector<T> const& xvec,
                         AlgVector<T> const& bvec, AlgVector<T>& rvec)
{
    amrex::computeResidual(rvec, mat, xvec, bvec);
    auto rnorm = rvec.norm2();
    return rnorm;
}

template <typename T>
void AMG<T>::vcycle () // Solve L(m_cor[0]) = m_res[0] with v-cycle
{
    smooth(0, m_cor[0], m_res[0], m_nu1, false); // cor = L^-1 (res) w/o initial guess
    computeResidual(m_rescor[0], *m_mat, m_cor[0], m_res[0]); // rescor = res - L(cor)

    // res[1] = R(rescor[0])

    // smooth on level 1

    // cor[0] += P(cor[1]);

    smooth(0, m_cor[0], m_res[0], m_nu2, true); // cor = L^-1 (res) w/ initial guess
}

template <typename T>
void AMG<T>::smooth (int mglev, AlgVector<T>& xvec, AlgVector<T> const& bvec, int nu,
                     bool with_initial_guess)
{
    JacobiSmoother<T> jacobi_smoother(m_mat);
    jacobi_smoother.setNumIters(nu);
    jacobi_smoother(xvec, bvec, with_initial_guess);
}

template <typename T>
void AMG<T>::setup ()
{
    m_res.emplace_back(m_mat->partition());
    m_cor.emplace_back(m_mat->partition());
    m_rescor.emplace_back(m_mat->partition()); // xxxxx we don't need to build m_rescor at the bottom

    // Reference: Reducing Complexity in Parallel Algebraic Multigrid
    // Preconditioners, Sterck, Yanng, & Heys, SIAM Journal on Matrix
    // Analysis and Applications, vol. 27, iss. 4, 2006.
    // https://doi.org/10.1137/040615729

    m_soc.emplace_back(create_soc(*m_mat));
    m_soc_t.emplace_back(amrex::transpose(m_soc[0]));

    auto wi = m_soc_t[0].rowSum();
    auto const* AMREX_RESTRICT pwi = wi.data();

    auto nrows = m_mat->numLocalRows();
    Gpu::DeviceVector<PointType> state(nrows);
    Gpu::DeviceVector<float> weight(nrows);

    auto* AMREX_RESTRICT pw = weight.data();
    {
        auto* AMREX_RESTRICT ps = state.data();
        ParallelForRNG(nrows, [=] AMREX_GPU_DEVICE (Long i, RandomEngine const& eng)
        {
            if (pwi[i] == 0) {
                pw[i] = 0;
                ps[i] = PointType::fine;
            } else {
                pw[i] = float(pwi[i]) + float(Random(eng));
                ps[i] = PointType::undecided;
            }
        });
    }

#ifdef AMREX_USE_MPI
    amrex::Abort("xxxxx TODO: AMG::setup");
    // use m_mat to communicate weight vector
#endif

    // auto const& S = m_soc[0].const_parcsr();
    auto const& St = m_soc_t[0].const_parcsr();

    Gpu::DeviceVector<PointType> state2(nrows);

    while (true) {
        auto const* ps = state.data();
        auto* ps2 = state2.data();

        Gpu::copyAsync(Gpu::deviceToDevice, state.begin(), state.end(), state2.begin());

        // Select C-points that have strong influence on others
        for (Long i = 0; i < nrows; ++i) {
            if (ps[i] == PointType::undecided) {
                bool strong = true;
                for (auto idx = St.csr0.row_offset[i];
                          idx < St.csr0.row_offset[i+1]; ++idx) {
                    Long j = St.csr0.col_index[idx];
                    if (i != j && ps[j] == PointType::undecided &&
                        (pw[i] < pw[j] || (pw[i] == pw[j] && i < j)))
                    {
                        strong = false;
                        break;
                    }
                }
                if (strong) {
                    ps2[i] = PointType::coarse;
                    for (auto idx = St.csr0.row_offset[i];
                              idx < St.csr0.row_offset[i+1]; ++idx) {
                        Long j = St.csr0.col_index[idx];
                        if (i != j && ps[j] == PointType::undecided &&
                            St.csr0.mat[idx])
                        {
                            ps2[j] = PointType::fine;
                        }
                    }
                }
            }
        }

        auto nund2 = Reduce::Sum<Long>(nrows, [=] AMREX_GPU_DEVICE (Long i) {
            return ps2[i] == PointType::undecided;
        });

        std::swap(state, state2);

        if (nund2 == 0) { break; }
    }

//    for (int irow = 0; irow < nrows; ++irow) {
//        amrex::Print() << "state[" << irow << "] = " << int(state[irow])
//                       << ", weight = " << weight[irow] << "\n";
//    }

    AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 2);
    int ncells = int(std::sqrt(nrows));
    AMREX_ALWAYS_ASSERT(nrows == ncells*ncells);
    Box bx(IntVect(0),IntVect(ncells-1));
    BoxArray ba(bx);
    DistributionMapping dm{ba};
    Gpu::DeviceVector<Real> dstate(nrows);
    for (int irow = 0; irow < nrows; ++irow) {
        dstate[irow] = Real(state[irow]);
    }
    MultiFab mf(ba, dm, 1, 0, MFInfo{}.SetAlloc(false));
    mf.setFab(0, FArrayBox(bx, 1, dstate.data()));
    VisMF::Write(mf, "state");

    m_prepared = true;
}

template <typename T>
SpMatrix<short> AMG<T>::create_soc (SpMatrix<T> const& A)
{
    SpMatrix<short>::CSRT S_csr;
    Long nrows = A.numLocalRows();
    Long nnz   = A.numLocalNonZeros();
    Long row_begin = A.globalRowBegin();
    S_csr.resize(nrows, nnz);
    auto* AMREX_RESTRICT s_mat = S_csr.mat.data();
    auto* AMREX_RESTRICT s_col = S_csr.col_index.data();
    auto* AMREX_RESTRICT s_row = S_csr.row_offset.data();

    auto const& A_csr = A.m_csr;
    auto* AMREX_RESTRICT A_mat = A_csr.mat.data();
    auto* AMREX_RESTRICT A_col = A_csr.col_index.data();
    auto* AMREX_RESTRICT A_row = A_csr.row_offset.data();
    auto A_col_os = A.m_split ? Long(0) : row_begin;

    auto const& Ad = A.diagonalVector();
    auto* AMREX_RESTRICT A_ii = Ad.data();

    auto threshold = m_strong_threshold;

#ifdef AMREX_USE_MPI
    auto const& A_csr_remote = A.m_csr_remote;
    if (A_csr_remote.nnz > 0) {
        amrex::Abort("xxxxx remote nnz > 0 todo");
    } else
#endif
    {
#if !defined(AMREX_USE_GPU)

        s_row[0] = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
        for (Long i = 0; i < nrows; ++i) {
            s_row[i+1] = A_row[i+1];
            T const sgn = std::copysign(T(1), A_ii[i]);
            T amin = std::numeric_limits<T>::max();
            for (Long j = A_row[i]; j < A_row[i+1]; ++j) {
                auto lcol = A_col[j] - A_col_os;
                if (i != lcol) {
                    amin = std::min(amin, sgn*A_mat[j]);
                }
                s_col[j] = lcol + row_begin;
            }
            amin *= threshold;
            for (Long j = A_row[i]; j < A_row[i+1]; ++j) {
                auto lcol = A_col[j] - A_col_os;
                s_mat[j] = (sgn*A_mat[j] < amin) && (i != lcol) ? 1 : 0;
            }
        }

#elif defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

        // One warp per row.

#elif defined(AMREX_USE_SYCL)
        amrex::Abort("xxxxx todo");
#endif
    }

    return SpMatrix<short>(A.partition(), std::move(S_csr));
}

}

#endif
