#ifndef AMREX_DIM3_H_
#define AMREX_DIM3_H_
#include <AMReX_Config.H>

#include <AMReX_REAL.H>
#include <AMReX_INT.H>
#include <iosfwd>

namespace amrex {

struct Dim3 { int x; int y; int z; };
struct LDim3 { Long x; Long y; Long z; };
struct XDim3 { Real x; Real y; Real z; };

#if (AMREX_SPACEDIM == 1)

struct DimN { int i0; };
struct LDimN { Long i0; };
struct XDimN { Real x0; };

#elif (AMREX_SPACEDIM == 2)

struct DimN { int i0; int i1; };
struct LDimN { Long i0; Long i1; };
struct XDimN { Real x0; Real x1; };

#elif (AMREX_SPACEDIM == 3)

struct DimN { int i0; int i1; int i2; };
struct LDimN { Long i0; Long i1; Long i2; };
struct XDimN { Real x0; Real x1; Real x2; };

#elif (AMREX_SPACEDIM == 4)

struct DimN { int i0; int i1; int i2; int i3; };
struct LDimN { Long i0; Long i1; Long i2; Long i3; };
struct XDimN { Real x0; Real x1; Real x2; Real x3; };

#elif (AMREX_SPACEDIM == 5)

struct DimN { int i0; int i1; int i2; int i3; int i4; };
struct LDimN { Long i0; Long i1; Long i2; Long i3; Long i4; };
struct XDimN { Real x0; Real x1; Real x2; Real x3; Real x4; };

#elif (AMREX_SPACEDIM == 6)

struct DimN { int i0; int i1; int i2; int i3; int i4; int i5; };
struct LDimN { Long i0; Long i1; Long i2; Long i3; Long i4; Long i5; };
struct XDimN { Real x0; Real x1; Real x2; Real x3; Real x4; Real x5; };

#endif

std::ostream& operator<< (std::ostream& os, const Dim3& d);
std::ostream& operator<< (std::ostream& os, const DimN& d);

}

#endif
