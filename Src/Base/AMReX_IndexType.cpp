
#include <AMReX_IndexType.H>

#include <iostream>
#include <iomanip>

namespace amrex {

std::ostream&
operator<< (std::ostream& os, const IndexType& it)
{
    os << '('
       << AMREX_D6_TERM( (it.test(0)?'N':'C'),
               << ',' << (it.test(1)?'N':'C'),
               << ',' << (it.test(2)?'N':'C'),
               << ',' << (it.test(3)?'N':'C'),
               << ',' << (it.test(4)?'N':'C'),
               << ',' << (it.test(5)?'N':'C')) << ')';

    if (os.fail()) {
        amrex::Error("operator<<(ostream&,IndexType&) failed");
    }

    return os;
}

//
// Copied from <Utility.H>
//
#define BL_IGNORE_MAX 100000

std::istream&
operator>> (std::istream& is, IndexType& it)
{
    char AMREX_D6_DECL(t0,t1,t2,t3,t4,t5);

    AMREX_D6_EXPR( is.ignore(BL_IGNORE_MAX, '(') >> t0,
                   is.ignore(BL_IGNORE_MAX, ',') >> t1,
                   is.ignore(BL_IGNORE_MAX, ',') >> t2,
                   is.ignore(BL_IGNORE_MAX, ',') >> t3,
                   is.ignore(BL_IGNORE_MAX, ',') >> t4,
                   is.ignore(BL_IGNORE_MAX, ',') >> t5 );
    is.ignore(BL_IGNORE_MAX, ')');
    AMREX_D6_TERM(
        BL_ASSERT(t0 == 'C' || t0 == 'N'); t0=='N'?it.set(0):it.unset(0); ,
        BL_ASSERT(t1 == 'C' || t1 == 'N'); t1=='N'?it.set(1):it.unset(1); ,
        BL_ASSERT(t2 == 'C' || t2 == 'N'); t2=='N'?it.set(2):it.unset(2); ,
        BL_ASSERT(t3 == 'C' || t3 == 'N'); t3=='N'?it.set(3):it.unset(3); ,
        BL_ASSERT(t4 == 'C' || t4 == 'N'); t4=='N'?it.set(4):it.unset(4); ,
        BL_ASSERT(t5 == 'C' || t5 == 'N'); t5=='N'?it.set(5):it.unset(5); );

    if (is.fail()) {
        amrex::Error("operator>>(ostream&,IndexType&) failed");
    }

    return is;
}

}
