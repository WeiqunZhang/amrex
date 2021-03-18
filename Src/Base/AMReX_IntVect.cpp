
#include <AMReX_IntVect.H>
#include <AMReX_BLassert.H>
#include <AMReX.H>
#include <AMReX_Utility.H>
#include <AMReX_IndexType.H>

#include <iostream>

namespace amrex {

const IntVect IntVect::Zero = IntVect::TheZeroVector();
const IntVect IntVect::Unit = IntVect::TheUnitVector();

std::ostream&
operator<< (std::ostream& os, const IntVect& p)
{
    os << AMREX_D6_TERM( '(' << p[0] , <<
                         ',' << p[1] , <<
                         ',' << p[2] , <<
                         ',' << p[3] , <<
                         ',' << p[4] , <<
                         ',' << p[5])  << ')';
    if (os.fail()) {
        amrex::Error("operator<<(ostream&,IntVect&) failed");
    }
    return os;
}

#define BL_IGNORE_MAX 100000

std::istream&
operator>> (std::istream& is, IntVect& iv)
{
    is >> std::ws;
    char c;
    is >> c;

    AMREX_D_TERM(iv[0]=0;, iv[1]=0;, iv[2]=0);

    if (c == '(')
    {
        is >> iv[0];
#if (AMREX_SPACEDIM >= 2)
        is >> std::ws;
        int ic = is.peek();
        if (ic == static_cast<int>(',')) {
            is.ignore(BL_IGNORE_MAX, ',');
            is >> iv[1];
#if (AMREX_SPACEDIM >= 3)
            is >> std::ws;
            ic = is.peek();
            if (ic == static_cast<int>(',')) {
                is.ignore(BL_IGNORE_MAX, ',');
                is >> iv[2];
#if (AMREX_SPACEDIM >= 4)
                is >> std::ws;
                ic = is.peek();
                if (ic == static_cast<int>(',')) {
                    is.ignore(BL_IGNORE_MAX, ',');
                    is >> iv[3];
#if (AMREX_SPACEDIM >= 5)
                    is >> std::ws;
                    ic = is.peek();
                    if (ic == static_cast<int>(',')) {
                        is.ignore(BL_IGNORE_MAX, ',');
                        is >> iv[4];
#if (AMREX_SPACEDIM == 6)
                        is >> std::ws;
                        ic = is.peek();
                        if (ic == static_cast<int>(',')) {
                            is.ignore(BL_IGNORE_MAX, ',');
                            is >> iv[5];
                        }
#endif
                    }
#endif
                }
#endif
            }
#endif
        }
#endif
        is.ignore(BL_IGNORE_MAX, ')');
    }
    else
    {
        amrex::Error("operator>>(istream&,IntVect&): expected \'(\'");
    }

    if (is.fail()) {
        amrex::Error("operator>>(istream&,IntVect&) failed");
    }

    return is;
}

std::ostream&
operator<< (std::ostream& os, const IntVectDim& ivd)
{
    IntVect const& iv = ivd.intVect();
    Dimension d = ivd.dimension();
    os << '(' << iv[0];
    if (d > Dimension::one  ) os << ',' << iv[1];
    if (d > Dimension::two  ) os << ',' << iv[2];
    if (d > Dimension::three) os << ',' << iv[3];
    if (d > Dimension::four ) os << ',' << iv[4];
    if (d > Dimension::five ) os << ',' << iv[5];
    os << ')';
    if (os.fail()) {
        amrex::Error("operator<<(ostream&,IntVect&) failed");
    }
    return os;
}

std::istream&
operator>> (std::istream& is, IntVectDim& ivd)
{
    is >> std::ws;
    char c;
    is >> c;

    IntVect iv;

    int d = 0;
    if (c == '(')
    {
        is >> iv[0];
        ++d;
#if (AMREX_SPACEDIM >= 2)
        is >> std::ws;
        int ic = is.peek();
        if (ic == static_cast<int>(',')) {
            is.ignore(BL_IGNORE_MAX, ',');
            is >> iv[1];
            ++d;
#if (AMREX_SPACEDIM >= 3)
            is >> std::ws;
            ic = is.peek();
            if (ic == static_cast<int>(',')) {
                is.ignore(BL_IGNORE_MAX, ',');
                is >> iv[2];
                ++d;
#if (AMREX_SPACEDIM >= 4)
                is >> std::ws;
                ic = is.peek();
                if (ic == static_cast<int>(',')) {
                    is.ignore(BL_IGNORE_MAX, ',');
                    is >> iv[3];
                    ++d;
#if (AMREX_SPACEDIM >= 5)
                    is >> std::ws;
                    ic = is.peek();
                    if (ic == static_cast<int>(',')) {
                        is.ignore(BL_IGNORE_MAX, ',');
                        is >> iv[4];
                        ++d;
#if (AMREX_SPACEDIM == 6)
                        is >> std::ws;
                        ic = is.peek();
                        if (ic == static_cast<int>(',')) {
                            is.ignore(BL_IGNORE_MAX, ',');
                            is >> iv[5];
                            ++d;
                        }
#endif
                    }
#endif
                }
#endif
            }
#endif
        }
#endif
        is.ignore(BL_IGNORE_MAX, ')');
    }
    else
    {
        amrex::Error("operator>>(istream&,IntVect&): expected \'(\'");
    }

    if (is.fail()) {
        amrex::Error("operator>>(istream&,IntVect&) failed");
    }

    ivd = IntVectDim(iv,static_cast<Dimension>(d));

    return is;
}

}
