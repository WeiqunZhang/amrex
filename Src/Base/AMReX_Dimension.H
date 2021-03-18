#ifndef AMREX_DIMENSION_H_
#define AMREX_DIMENSION_H_
#include <AMReX_Config.H>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include <iosfwd>
#include <cstdint>

namespace amrex {

enum struct Dimension : short { one = 1, two = 2, three = 3, four = 4, five = 5, six = 6 };

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr Dimension operator+ (Dimension d, short i) {
    return static_cast<Dimension>(static_cast<short>(d)+i);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr Dimension operator- (Dimension d, short i) {
    return static_cast<Dimension>(static_cast<short>(d)-i);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Dimension& operator++ (Dimension& d)
{
    d = static_cast<Dimension>(static_cast<short>(d)+1);
    return d;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Dimension& operator-- (Dimension& d)
{
    d = static_cast<Dimension>(static_cast<short>(d)-1);
    return d;
}

#if (AMREX_SPACEDIM == 1)
static constexpr Dimension maxDim = Dimension::one;
#elif (AMREX_SPACEDIM == 2)
static constexpr Dimension maxDim = Dimension::two;
#elif (AMREX_SPACEDIM == 3)
static constexpr Dimension maxDim = Dimension::three;
#elif (AMREX_SPACEDIM == 4)
static constexpr Dimension maxDim = Dimension::four;
#elif (AMREX_SPACEDIM == 5)
static constexpr Dimension maxDim = Dimension::five;
#elif (AMREX_SPACEDIM == 6)
static constexpr Dimension maxDim = Dimension::six;
#endif

enum struct Direction : int { x = 0, y = 1, z = 2, u = 3, v = 4, w = 5 };

std::ostream& operator<< (std::ostream& os, Dimension const& dimen);

}

#endif
