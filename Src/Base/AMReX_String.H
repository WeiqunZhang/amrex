#ifndef AMREX_STRING_H_
#define AMREX_STRING_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_String.H
 * \brief String utility functions.
 */

#include <string>
#include <vector>

namespace amrex {

    /// \ingroup amrex_utilities
    //! Convert \p s to lowercase characters using the current locale.
    std::string toLower (std::string s);

    /// \ingroup amrex_utilities
    //! Convert \p s to uppercase characters using the current locale.
    std::string toUpper (std::string s);

    /// \ingroup amrex_utilities
    //! Trim leading/trailing characters from \p s that appear in \p space.
    std::string trim (std::string s, std::string const& space = " \t");

    /// \ingroup amrex_utilities
    //! Return `rootNNNN`, zero-padding \p num out to \p mindigits digits.
    std::string Concatenate (const std::string& root,
                             int                num,
                             int                mindigits = 5);

    /// \ingroup amrex_utilities
    //! Split \p s using any character in \p sep as a delimiter.
    std::vector<std::string> split (std::string const& s,
                                    std::string const& sep = " \t");

    /// \ingroup amrex_utilities
    //! Join the entries in \p sv using the delimiter character \p sep.
    std::string join (std::vector<std::string> const& sv, char sep);

    /// \ingroup amrex_utilities
    //! Join the entries in \p sv without any delimiter.
    std::string join (std::vector<std::string> const& sv);
}

#endif
