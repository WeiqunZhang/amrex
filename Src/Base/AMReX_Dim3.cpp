
#include <AMReX_Dim3.H>
#include <iostream>

namespace amrex {

std::ostream& operator<< (std::ostream& os, const Dim3& d)
{
    os << '(' << d.x << ',' << d.y << ',' << d.z << ')';
    return os;
}

#if (AMREX_SPACEDIM == 1)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 <<')';
    return os;
}

#elif (AMREX_SPACEDIM == 2)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 << ',' << d.i1 <<')';
    return os;
}

#elif (AMREX_SPACEDIM == 3)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 << ',' << d.i1 << ',' << d.i2 <<')';
    return os;
}

#elif (AMREX_SPACEDIM == 4)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 << ',' << d.i1 << ',' << d.i2 << ',' << d.i3 <<')';
    return os;
}

#elif (AMREX_SPACEDIM == 5)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 << ',' << d.i1 << ',' << d.i2
       << ',' << d.i3 << ',' << d.i4 <<')';
    return os;
}

#elif (AMREX_SPACEDIM == 6)

std::ostream& operator<< (std::ostream& os, const DimN& d)
{
    os << '(' << d.i0 << ',' << d.i1 << ',' << d.i2
       << ',' << d.i3 << ',' << d.i4 << ',' << d.i5 <<')';
    return os;
}

#endif

}
