#include <AMReX_EBStaggeredData.H>

namespace amrex
{

EBStaggeredData::EBStaggeredData (int direction,
                                  const Geometry& geom,
                                  const BoxArray& ba,
                                  const DistributionMapping& dm,
                                  Vector<int> ngrow,
                                  EBSupport support)
{
    define(direction, geom, ba, dm, std::move(ngrow), support);
}

void
EBStaggeredData::define (int direction,
                         const Geometry& geom,
                         const BoxArray& ba,
                         const DistributionMapping& dm,
                         Vector<int> ngrow,
                         EBSupport support)
{
    reset();

    m_direction = direction;
    m_geom = geom;
    m_ngrow = std::move(ngrow);
    m_support = support;

    if (m_direction < 0 || m_direction >= AMREX_SPACEDIM) {
        return;
    }

    AMREX_ALWAYS_ASSERT(!m_ngrow.empty());

    const BoxArray stag_ba = amrex::convert(ba, IntVect::TheDimensionVector(m_direction));

    m_cellflags = std::make_unique<FabArray<EBCellFlagFab>>(stag_ba, dm, 1, m_ngrow[0],
                                                           MFInfo(),
                                                           DefaultFabFactory<EBCellFlagFab>());

    if (m_support >= EBSupport::volume) {
        int ng_vol = (m_ngrow.size() >= 2) ? m_ngrow[1] : m_ngrow[0];
        m_volfrac = std::make_unique<MultiFab>(stag_ba, dm, 1, ng_vol,
                                               MFInfo(), FArrayBoxFactory());
        m_centroid = std::make_unique<MultiFab>(stag_ba, dm, AMREX_SPACEDIM,
                                                ng_vol, MFInfo(), FArrayBoxFactory());
    }

    if (m_support == EBSupport::full) {
        int ng_full = (m_ngrow.size() >= 3) ? m_ngrow[2] : m_ngrow.back();

        m_bndrycent = std::make_unique<MultiFab>(stag_ba, dm, AMREX_SPACEDIM,
                                                 ng_full, MFInfo(), FArrayBoxFactory());
        m_bndrynorm = std::make_unique<MultiFab>(stag_ba, dm, AMREX_SPACEDIM,
                                                 ng_full, MFInfo(), FArrayBoxFactory());
        m_bndryarea = std::make_unique<MultiFab>(stag_ba, dm, 1,
                                                 ng_full, MFInfo(), FArrayBoxFactory());

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            IntVect face_type = IntVect::TheZeroVector();
            face_type[m_direction] = 1;
            face_type[idim] = 1;
            BoxArray face_ba = amrex::convert(ba, face_type);
            m_areafrac[idim] = std::make_unique<MultiFab>(face_ba, dm, 1,
                                                          ng_full+1, MFInfo(), FArrayBoxFactory());
            m_facecent[idim] = std::make_unique<MultiFab>(face_ba, dm,
                                                          AMREX_SPACEDIM-1,
                                                          ng_full, MFInfo(), FArrayBoxFactory());
            // The aligned caches repack face-based staggered data into the
            // legacy base BoxArray layout expected by ERF's eb_aux_ side data.
            m_areafrac_aligned[idim] = std::make_unique<MultiFab>(ba, dm, 1,
                                                                  ng_full+1, MFInfo(), FArrayBoxFactory());
            m_facecent_aligned[idim] = std::make_unique<MultiFab>(ba, dm,
                                                                  AMREX_SPACEDIM-1,
                                                                  ng_full, MFInfo(), FArrayBoxFactory());
        }
    }
}

Array<MultiFab*,AMREX_SPACEDIM>
EBStaggeredData::getAreaFrac () const noexcept
{
    Array<MultiFab*,AMREX_SPACEDIM> r
        {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        r[d] = m_areafrac[d].get();
    }
    return r;
}

Array<MultiFab*,AMREX_SPACEDIM>
EBStaggeredData::getFaceCent () const noexcept
{
    Array<MultiFab*,AMREX_SPACEDIM> r
        {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        r[d] = m_facecent[d].get();
    }
    return r;
}

Array<MultiFab*,AMREX_SPACEDIM>
EBStaggeredData::getAreaFracAligned () const noexcept
{
    Array<MultiFab*,AMREX_SPACEDIM> r
        {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        r[d] = m_areafrac_aligned[d].get();
    }
    return r;
}

Array<MultiFab*,AMREX_SPACEDIM>
EBStaggeredData::getFaceCentAligned () const noexcept
{
    Array<MultiFab*,AMREX_SPACEDIM> r
        {AMREX_D_DECL(nullptr,nullptr,nullptr)};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        r[d] = m_facecent_aligned[d].get();
    }
    return r;
}

void
EBStaggeredData::reset ()
{
    m_direction = -1;
    m_cellflags.reset();
    m_volfrac.reset();
    m_centroid.reset();
    m_bndrycent.reset();
    m_bndrynorm.reset();
    m_bndryarea.reset();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_areafrac[idim].reset();
        m_facecent[idim].reset();
        m_areafrac_aligned[idim].reset();
        m_facecent_aligned[idim].reset();
    }
}

}
