
#include <AMReX_Dimension.H>
#include <iostream>

namespace amrex {

std::ostream& operator<< (std::ostream& os, Dimension const& dimen)
{
    os << static_cast<short>(dimen);
    return os;
}

}
