#ifndef AMREX_MF_INTERPOLATER_H_
#define AMREX_MF_INTERPOLATER_H_
#include <AMReX_Config.H>

#include <AMReX_InterpBase.H>

namespace amrex {

class MFInterpolater
{
public:
    virtual ~MFInterpolater () = default;
};

}

#endif
