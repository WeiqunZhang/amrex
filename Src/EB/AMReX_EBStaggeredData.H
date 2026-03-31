#ifndef AMREX_EBSTAGGEREDDATA_H_
#define AMREX_EBSTAGGEREDDATA_H_
#include <AMReX_Config.H>

#include <memory>

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_EBSupport.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace amrex
{

/**
 * \brief Lightweight EB data container defined on a single staggered (face-centered)
 *        index type.  It mirrors the data products exposed by EBDataCollection but
 *        keeps allocations scoped to one staggered direction so downstream codes
 *        can cache exactly what they need (flags, volume fraction, centroids, etc.).
 */
class EBStaggeredData
{
public:

    EBStaggeredData () = default;
    EBStaggeredData (int direction,
                     const Geometry& geom,
                     const BoxArray& ba,
                     const DistributionMapping& dm,
                     Vector<int> ngrow,
                     EBSupport support);

    EBStaggeredData (const EBStaggeredData&) = delete;
    EBStaggeredData (EBStaggeredData&&) noexcept = default;
    ~EBStaggeredData () = default;

    EBStaggeredData& operator= (const EBStaggeredData&) = delete;
    EBStaggeredData& operator= (EBStaggeredData&&) noexcept = default;

    void define (int direction,
                 const Geometry& geom,
                 const BoxArray& ba,
                 const DistributionMapping& dm,
                 Vector<int> ngrow,
                 EBSupport support);

    [[nodiscard]] bool isDefined () const noexcept { return m_cellflags != nullptr; }

    [[nodiscard]] int direction () const noexcept { return m_direction; }
    [[nodiscard]] const Geometry& geom () const noexcept { return m_geom; }
    [[nodiscard]] EBSupport supportLevel () const noexcept { return m_support; }

    [[nodiscard]] FabArray<EBCellFlagFab>& getMultiEBCellFlagFab () const noexcept
        { return *m_cellflags; }

    [[nodiscard]] MultiFab& getVolFrac () const noexcept { return *m_volfrac; }
    [[nodiscard]] MultiFab& getCentroid () const noexcept { return *m_centroid; }

    [[nodiscard]] MultiFab& getBndryCent () const noexcept { return *m_bndrycent; }
    [[nodiscard]] MultiFab& getBndryNormal () const noexcept { return *m_bndrynorm; }
    [[nodiscard]] MultiFab& getBndryArea () const noexcept { return *m_bndryarea; }

    [[nodiscard]] Array<MultiFab*,AMREX_SPACEDIM> getAreaFrac () const noexcept;
    [[nodiscard]] Array<MultiFab*,AMREX_SPACEDIM> getFaceCent () const noexcept;
    [[nodiscard]] Array<MultiFab*,AMREX_SPACEDIM> getAreaFracAligned () const noexcept;
    [[nodiscard]] Array<MultiFab*,AMREX_SPACEDIM> getFaceCentAligned () const noexcept;

private:

    void reset ();

    int m_direction = -1;
    Vector<int> m_ngrow;
    Geometry m_geom;
    EBSupport m_support = EBSupport::none;

    std::unique_ptr<FabArray<EBCellFlagFab>> m_cellflags;
    std::unique_ptr<MultiFab> m_volfrac;
    std::unique_ptr<MultiFab> m_centroid;

    std::unique_ptr<MultiFab> m_bndrycent;
    std::unique_ptr<MultiFab> m_bndrynorm;
    std::unique_ptr<MultiFab> m_bndryarea;
    Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM> m_areafrac;
    Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM> m_facecent;
    Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM> m_areafrac_aligned;
    Array<std::unique_ptr<MultiFab>,AMREX_SPACEDIM> m_facecent_aligned;
};

}

#endif
