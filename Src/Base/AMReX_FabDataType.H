#ifndef AMREX_FAB_DATA_TYPE_H_
#define AMREX_FAB_DATA_TYPE_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_FabDataType.H
 * \brief Type trait that exposes the FAB and value types of MultiFab-like containers.
 */

#include <AMReX_TypeTraits.H>

namespace amrex {

/// \cond DOXYGEN_IGNORE
template <typename T, class Enable = void> struct FabDataType {};
//
template <typename T>
struct FabDataType <T, std::enable_if_t<IsMultiFabLike_v<T> > >
{
    //! Alias to the FAB type stored directly inside \p T.
    using   fab_type = typename T::fab_type;
    //! Scalar value type held by the FAB.
    using value_type = typename T::value_type;
};

template <typename T>
struct FabDataType <T, std::enable_if_t<IsMultiFabLike_v<typename T::value_type> > >
{
    //! Alias to the FAB type stored inside the container element.
    using   fab_type = typename T::value_type::fab_type;
    //! Scalar value type held by that FAB.
    using value_type = typename T::value_type::value_type;
};
/// \endcond

}

#endif
