#ifndef AMREX_COMPENSATEDSUM_H_
#define AMREX_COMPENSATEDSUM_H_
#include <AMReX_Config.H>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

namespace amrex
{

// Kahan compensation relies on (t - sum) - y evaluating to the rounding
// error lost by sum + y, rather than to the exact-arithmetic value of 0.
// Optimizations that assume floating-point addition is associative (e.g.
// -ffast-math, -fassociative-math) or that fuse operations across the
// subtractions (aggressive FP contraction) are free to simplify that
// expression to 0, which silently turns compensatedAdd into a plain,
// uncompensated sum. Most compilers only enable such optimizations when a
// user opts in (e.g. AMReX_FASTMATH, off by default -- see
// Tools/CMake/AMReXOptions.cmake). Since this function's correctness must not
// depend on the flags used by its translation unit, its body is pinned to
// strict floating-point semantics regardless.
//
// This function is deliberately NOT AMREX_FORCE_INLINE. Forcing
// inlining across an optimization-setting boundary like the one below
// causes both compilers to inline the -ffast-math-optimized call-site
// version instead of honoring this function's own pragma-scoped settings,
// silently defeating the protection below.
//
// The pragmas below were verified empirically (Compiler Explorer) for GCC,
// Clang, Intel classic icc, Intel icx (clang-based, covered by the clang
// branch), MSVC, clang-cl (defines both _MSC_VER and __clang__; the clang
// branch takes priority and is what actually applies), NVCC host and
// device code, AMD's HIP/ROCm clang host and device code (also covered by
// the clang branch), and NVHPC. NVCC's device compiler was confirmed not to
// need protection: neither --use_fast_math nor --fmad=true collapses this
// expression, unlike host-side -ffast-math. NVHPC nvc++ 26.5 also preserves
// the expression under -fast, and it does not support GCC's optimize pragma
// even though it defines __GNUC__. The nv_diag_suppress pair
// exists only to silence NVCC's front end complaining about the (correct,
// but nvcc-unrecognized) GCC/MSVC/Intel pragmas when it parses this header
// while selecting a host-side branch; it has no effect on codegen.

/**
 * \brief Add a value to a running sum with Kahan compensation.
 *
 * Callers own both the accumulated sum and the compensation storage. Both
 * should be initialized before the first call, usually to zero.
 *
 * \param[in,out] sum Running sum updated by \p value.
 * \param[in,out] compensation Kahan compensation term updated to track the
 *                             rounding error from the addition.
 * \param[in] value Value to add to \p sum.
 */
template <typename T, typename U>
AMREX_GPU_HOST_DEVICE
void compensatedAdd (T& sum, T& compensation, U value) noexcept
{
    T const y = static_cast<T>(value) - compensation;
    T volatile t = sum + y;
    T volatile c = t - sum;
    compensation = c - y;
    sum = t;
}

}

#endif
