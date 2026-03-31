
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBStaggeredData.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_FabArray.H>

#include <AMReX_EB2_Level.H>
#include <AMReX_EB2.H>

namespace amrex
{

EBFArrayBoxFactory::EBFArrayBoxFactory (const EB2::Level& a_level,
                                        const Geometry& a_geom,
                                        const BoxArray& a_ba,
                                        const DistributionMapping& a_dm,
                                        const Vector<int>& a_ngrow, EBSupport a_support)
    : m_support(a_support),
      m_geom(a_geom),
      m_ebdc(std::make_shared<EBDataCollection>(a_level,a_geom,a_ba,a_dm,a_ngrow,a_support)),
      m_parent(&a_level),
      m_ngrow(a_ngrow)
{
    auto const& ebflags = getMultiEBCellFlagFab();
    {
        // If we do not do this here, there would a race condition when
        // calling const_array() in getEBData, because const_arrays() is not
        // thread safe.
        auto const& ma = ebflags.const_arrays();
        amrex::ignore_unused(ma);
    }
#ifdef AMREX_USE_GPU
    m_eb_data.resize(EBData::real_data_size*ebflags.local_size());
    Gpu::PinnedVector<Array4<Real const>> eb_data_hv;
#else
    auto& eb_data_hv = m_eb_data;
#endif

    eb_data_hv.reserve(EBData::real_data_size*ebflags.local_size());

    for (MFIter mfi(ebflags,MFItInfo{}.DisableDeviceSync()); mfi.isValid(); ++mfi) {
        Array4<Real const> a{};

        bool cutfab_is_ok = ebflags[mfi].getType() == FabType::singlevalued;

        a = ( m_ebdc->m_levelset )
            ? m_ebdc->m_levelset->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        a = ( m_ebdc->m_volfrac )
            ? m_ebdc->m_volfrac->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        a = ( m_ebdc->m_centroid && cutfab_is_ok )
            ? m_ebdc->m_centroid->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        a = ( m_ebdc->m_bndrycent && cutfab_is_ok )
            ? m_ebdc->m_bndrycent->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        a = ( m_ebdc->m_bndrynorm && cutfab_is_ok )
            ? m_ebdc->m_bndrynorm->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        a = ( m_ebdc->m_bndryarea && cutfab_is_ok )
            ? m_ebdc->m_bndryarea->const_array(mfi) : Array4<Real const>{};
        eb_data_hv.push_back(a);

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            a = ( m_ebdc->m_areafrac[idim] && cutfab_is_ok )
                ? m_ebdc->m_areafrac[idim]->const_array(mfi) : Array4<Real const>{};
            eb_data_hv.push_back(a);
        }

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            a = ( m_ebdc->m_facecent[idim] && cutfab_is_ok )
                ? m_ebdc->m_facecent[idim]->const_array(mfi) : Array4<Real const>{};
            eb_data_hv.push_back(a);
        }

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            a = ( m_ebdc->m_edgecent[idim] && cutfab_is_ok )
                ? m_ebdc->m_edgecent[idim]->const_array(mfi) : Array4<Real const>{};
            eb_data_hv.push_back(a);
        }
    }

#ifdef AMREX_USE_GPU
    Gpu::copyAsync(Gpu::hostToDevice, eb_data_hv.begin(), eb_data_hv.end(), m_eb_data.begin());
    Gpu::streamSynchronize();
#endif
}

EBFArrayBoxFactory::~EBFArrayBoxFactory () = default;

EBFArrayBoxFactory::EBFArrayBoxFactory (const EBFArrayBoxFactory& rhs)
    : m_support(rhs.m_support),
      m_geom(rhs.m_geom),
      m_ebdc(rhs.m_ebdc),
      m_parent(rhs.m_parent),
      m_eb_data(rhs.m_eb_data),
      m_ngrow(rhs.m_ngrow),
      m_stag_refined_geom(rhs.m_stag_refined_geom),
      m_has_stag_refined_geom(rhs.m_has_stag_refined_geom)
{
    for (auto& ptr : m_stag_data) { ptr.reset(); }
    m_stag_refined_data.reset();
}

EBFArrayBoxFactory&
EBFArrayBoxFactory::operator= (const EBFArrayBoxFactory& rhs)
{
    if (this != &rhs) {
        m_support = rhs.m_support;
        m_geom = rhs.m_geom;
        m_ebdc = rhs.m_ebdc;
        m_parent = rhs.m_parent;
        m_eb_data = rhs.m_eb_data;
        m_ngrow = rhs.m_ngrow;
        m_stag_refined_geom = rhs.m_stag_refined_geom;
        m_has_stag_refined_geom = rhs.m_has_stag_refined_geom;
        for (auto& ptr : m_stag_data) { ptr.reset(); }
        m_stag_refined_data.reset();
    }
    return *this;
}

AMREX_NODISCARD
FArrayBox*
EBFArrayBoxFactory::create (const Box& box, int ncomps,
                            const FabInfo& info, int box_index) const
{
    if (m_support == EBSupport::none)
    {
        return new FArrayBox(box, ncomps, info.alloc, info.shared, info.arena);
    }
    else
    {
        const EBCellFlagFab& ebcellflag = m_ebdc->getMultiEBCellFlagFab()[box_index];
        return new EBFArrayBox(ebcellflag, box, ncomps, info.arena, this, box_index);
    }
}

AMREX_NODISCARD
FArrayBox*
EBFArrayBoxFactory::create_alias (FArrayBox const& rhs, int scomp, int ncomp) const
{
    if (m_support == EBSupport::none)
    {
        return new FArrayBox(rhs, amrex::make_alias, scomp, ncomp);
    }
    else
    {
        auto const& ebrhs = static_cast<EBFArrayBox const&>(rhs);
        return new EBFArrayBox(ebrhs, amrex::make_alias, scomp, ncomp);
    }
}

void
EBFArrayBoxFactory::destroy (FArrayBox* fab) const
{
    if (m_support == EBSupport::none)
    {
        delete fab;
    }
    else
    {
        auto* p = static_cast<EBFArrayBox*>(fab);
        delete p;
    }
}

AMREX_NODISCARD
EBFArrayBoxFactory*
EBFArrayBoxFactory::clone () const
{
    return new EBFArrayBoxFactory(*this);
}

bool
EBFArrayBoxFactory::isAllRegular () const noexcept
{
    return m_parent->isAllRegular();
}

EB2::IndexSpace const*
EBFArrayBoxFactory::getEBIndexSpace () const noexcept
{
    return (m_parent) ? m_parent->getEBIndexSpace() : nullptr;
}

int
EBFArrayBoxFactory::maxCoarseningLevel () const noexcept
{
    if (m_parent) {
        EB2::IndexSpace const* ebis = m_parent->getEBIndexSpace();
        return EB2::maxCoarseningLevel(ebis, m_geom);
    } else {
        return EB2::maxCoarseningLevel(m_geom);
    }
}

const DistributionMapping&
EBFArrayBoxFactory::DistributionMap () const noexcept
{
    return m_ebdc->getMultiEBCellFlagFab().DistributionMap();
}

const BoxArray&
EBFArrayBoxFactory::boxArray () const noexcept
{
    return m_ebdc->getMultiEBCellFlagFab().boxArray();
}

bool
EBFArrayBoxFactory::hasEBInfo () const noexcept
{
    return m_parent->hasEBInfo();
}

EBData
EBFArrayBoxFactory::getEBData (MFIter const& mfi) const noexcept
{
    int const li = mfi.LocalIndex();
    auto const& ebflags_ma = this->getMultiEBCellFlagFab().const_arrays();
#ifdef AMREX_USE_GPU
    auto const* pebflag = ebflags_ma.dp + li;
#else
    auto const* pebflag = ebflags_ma.hp + li;
#endif
    return EBData{.m_cell_flag = pebflag,
                  .m_real_data = m_eb_data.data()+EBData::real_data_size*li};
}

const EB2::Level*
EBFArrayBoxFactory::getRefinedLevel (int refinement_ratio) const
{
    if (m_parent == nullptr || refinement_ratio <= 1) {
        return nullptr;
    }

    EB2::IndexSpace const* ebis = m_parent->getEBIndexSpace();
    if (ebis == nullptr) {
        return nullptr;
    }

    Geometry refined_geom = amrex::refine(m_geom, refinement_ratio);

    if (!ebis->hasGeometry(refined_geom.Domain())) {
        int rr = refinement_ratio;
        int num_new_levels = 0;
        while (rr > 1) {
            AMREX_ALWAYS_ASSERT(rr % 2 == 0);
            rr /= 2;
            ++num_new_levels;
        }

        auto* non_const_ebis = const_cast<EB2::IndexSpace*>(ebis);
        non_const_ebis->addFineLevels(num_new_levels);

        ebis = m_parent->getEBIndexSpace();
        AMREX_ALWAYS_ASSERT(ebis != nullptr);
        m_parent = &(ebis->getLevel(m_geom));
    }

    return &(ebis->getLevel(refined_geom));
}

EBDataArrays
EBFArrayBoxFactory::getEBDataArrays () const noexcept
{
    auto const& ebflags_ma = this->getMultiEBCellFlagFab().const_arrays();
#ifdef AMREX_USE_GPU
    auto const* pebflag = ebflags_ma.dp;
#else
    auto const* pebflag = ebflags_ma.hp;
#endif
    return EBDataArrays{.m_cell_flag = pebflag,
                        .m_real_data = m_eb_data.data()};
}

const EBStaggeredData*
EBFArrayBoxFactory::getStaggeredData (int direction) const noexcept
{
    if (direction < 0 || direction >= AMREX_SPACEDIM) { return nullptr; }
    if (!m_stag_data[direction]) {
        const_cast<EBFArrayBoxFactory*>(this)->buildStaggeredData(direction);
    }
    return m_stag_data[direction].get();
}

namespace {

void
fillStaggeredVolFracAndFlags (EBStaggeredData& stag,
                              const EBDataCollection& fine_data,
                              const Geometry& fine_geom,
                              int direction)
{
    auto& coarse_flags = stag.getMultiEBCellFlagFab();
    auto& coarse_vol = stag.getVolFrac();

    const MultiFab& fine_vol = fine_data.getVolFrac();
    Box const& fine_domain = fine_geom.Domain();
    Dim3 const fine_lo_dim = lbound(fine_domain);
    Dim3 const fine_hi_dim = ubound(fine_domain);
    GpuArray<int,3> fine_lo {AMREX_D_DECL(fine_lo_dim.x, fine_lo_dim.y, fine_lo_dim.z)};
    GpuArray<int,3> fine_hi {AMREX_D_DECL(fine_hi_dim.x, fine_hi_dim.y, fine_hi_dim.z)};

    GpuArray<int,AMREX_SPACEDIM> ratio {AMREX_D_DECL(2,2,2)};
    GpuArray<int,AMREX_SPACEDIM> offset {AMREX_D_DECL(0,0,0)};
    offset[direction] = -ratio[direction]/2;
    constexpr Real covered_tol = Real(1.e-8);

    coarse_flags.setVal(EBCellFlag::TheDefaultCell());

    for (MFIter mfi(coarse_vol, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& fine_arr = fine_vol.const_array(mfi);
        auto const& coarse_arr = coarse_vol.array(mfi);
        auto const& flag_arr = coarse_flags.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int ilo_raw = ratio[0]*i + offset[0];
            int jlo_raw = ratio[1]*j + offset[1];
            int klo_raw = ratio[2]*k + offset[2];
            int ihi_raw = ilo_raw + ratio[0] - 1;
            int jhi_raw = jlo_raw + ratio[1] - 1;
            int khi_raw = klo_raw + ratio[2] - 1;

            Real sum = Real(0.0);
            int cnt = 0;
            for (int kk_raw = klo_raw; kk_raw <= khi_raw; ++kk_raw) {
                int kk = kk_raw;
                if (direction == 2) {
                    kk = (kk < fine_lo[2]) ? (2*fine_lo[2] - kk)
                       : (kk > fine_hi[2]) ? (2*fine_hi[2] - kk) : kk;
                } else if (kk < fine_lo[2] || kk > fine_hi[2]) {
                    continue;
                }
                for (int jj_raw = jlo_raw; jj_raw <= jhi_raw; ++jj_raw) {
                    int jj = jj_raw;
                    if (direction == 1) {
                        jj = (jj < fine_lo[1]) ? (2*fine_lo[1] - jj)
                           : (jj > fine_hi[1]) ? (2*fine_hi[1] - jj) : jj;
                    } else if (jj < fine_lo[1] || jj > fine_hi[1]) {
                        continue;
                    }
                    for (int ii_raw = ilo_raw; ii_raw <= ihi_raw; ++ii_raw) {
                        int ii = ii_raw;
                        if (direction == 0) {
                            ii = (ii < fine_lo[0]) ? (2*fine_lo[0] - ii)
                               : (ii > fine_hi[0]) ? (2*fine_hi[0] - ii) : ii;
                        } else if (ii < fine_lo[0] || ii > fine_hi[0]) {
                            continue;
                        }
                        sum += fine_arr(ii,jj,kk);
                        ++cnt;
                    }
                }
            }

            Real avg = (cnt > 0) ? sum / static_cast<Real>(cnt) : Real(0.0);
            coarse_arr(i,j,k) = avg;

            if (avg <= covered_tol) {
                flag_arr(i,j,k).setCovered();
            } else if (avg >= Real(1.0)-covered_tol) {
                flag_arr(i,j,k).setRegular();
            } else {
                flag_arr(i,j,k).setSingleValued();
            }
        });
    }
}

void
fillStaggeredCentroid (EBStaggeredData& stag,
                       const EBDataCollection& fine_data,
                       const Geometry& fine_geom,
                       int direction)
{
    auto& coarse_centroid = stag.getCentroid();
    const MultiFab& fine_vol = fine_data.getVolFrac();
    const MultiCutFab& fine_centroid = fine_data.getCentroid();

    GpuArray<int,AMREX_SPACEDIM> ratio {AMREX_D_DECL(2,2,2)};
    GpuArray<int,AMREX_SPACEDIM> offset {AMREX_D_DECL(0,0,0)};
    offset[direction] = -ratio[direction]/2;

    Box const& fine_domain = fine_geom.Domain();
    Dim3 const fine_lo_dim = lbound(fine_domain);
    Dim3 const fine_hi_dim = ubound(fine_domain);
    GpuArray<int,3> fine_lo {AMREX_D_DECL(fine_lo_dim.x, fine_lo_dim.y, fine_lo_dim.z)};
    GpuArray<int,3> fine_hi {AMREX_D_DECL(fine_hi_dim.x, fine_hi_dim.y, fine_hi_dim.z)};

    for (MFIter mfi(coarse_centroid, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& fine_vol_arr = fine_vol.const_array(mfi);
        auto const& fine_cent_arr = fine_centroid.const_array(mfi);
        auto const& coarse_cent_arr = coarse_centroid.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int ilo_raw = ratio[0]*i + offset[0];
            int jlo_raw = ratio[1]*j + offset[1];
            int klo_raw = ratio[2]*k + offset[2];
            int ihi_raw = ilo_raw + ratio[0] - 1;
            int jhi_raw = jlo_raw + ratio[1] - 1;
            int khi_raw = klo_raw + ratio[2] - 1;

            Real weight = Real(0.0);
            GpuArray<Real,AMREX_SPACEDIM> accum {AMREX_D_DECL(Real(0.0), Real(0.0), Real(0.0))};

            for (int kk_raw = klo_raw; kk_raw <= khi_raw; ++kk_raw) {
                int kk = kk_raw;
                if (direction == 2) {
                    kk = (kk < fine_lo[2]) ? (2*fine_lo[2] - kk)
                       : (kk > fine_hi[2]) ? (2*fine_hi[2] - kk) : kk;
                } else if (kk < fine_lo[2] || kk > fine_hi[2]) {
                    continue;
                }
                for (int jj_raw = jlo_raw; jj_raw <= jhi_raw; ++jj_raw) {
                    int jj = jj_raw;
                    if (direction == 1) {
                        jj = (jj < fine_lo[1]) ? (2*fine_lo[1] - jj)
                           : (jj > fine_hi[1]) ? (2*fine_hi[1] - jj) : jj;
                    } else if (jj < fine_lo[1] || jj > fine_hi[1]) {
                        continue;
                    }
                    for (int ii_raw = ilo_raw; ii_raw <= ihi_raw; ++ii_raw) {
                        int ii = ii_raw;
                        if (direction == 0) {
                            ii = (ii < fine_lo[0]) ? (2*fine_lo[0] - ii)
                               : (ii > fine_hi[0]) ? (2*fine_hi[0] - ii) : ii;
                        } else if (ii < fine_lo[0] || ii > fine_hi[0]) {
                            continue;
                        }
                        Real vol = fine_vol_arr(ii,jj,kk);
                        weight += vol;
                        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                            accum[n] += vol * fine_cent_arr(ii,jj,kk,n);
                        }
                    }
                }
            }

            if (weight > Real(0.0)) {
                Real inv_w = Real(1.0) / weight;
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    coarse_cent_arr(i,j,k,n) = accum[n] * inv_w;
                }
            } else {
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    coarse_cent_arr(i,j,k,n) = Real(0.0);
                }
            }
        });
    }
}

void
fillStaggeredBoundaryData (EBStaggeredData& stag,
                           const EBDataCollection& fine_data,
                           const Geometry& fine_geom,
                           int direction)
{
    auto& coarse_bndry_cent = stag.getBndryCent();
    auto& coarse_bndry_norm = stag.getBndryNormal();
    auto& coarse_bndry_area = stag.getBndryArea();

    const MultiCutFab& fine_bndry_cent = fine_data.getBndryCent();
    const MultiCutFab& fine_bndry_norm = fine_data.getBndryNormal();
    const MultiCutFab& fine_bndry_area = fine_data.getBndryArea();

    GpuArray<int,AMREX_SPACEDIM> ratio {AMREX_D_DECL(2,2,2)};
    GpuArray<int,AMREX_SPACEDIM> offset {AMREX_D_DECL(0,0,0)};
    offset[direction] = -ratio[direction]/2;

    Box const& fine_domain = fine_geom.Domain();
    Dim3 const fine_lo_dim = lbound(fine_domain);
    Dim3 const fine_hi_dim = ubound(fine_domain);
    GpuArray<int,3> fine_lo {AMREX_D_DECL(fine_lo_dim.x, fine_lo_dim.y, fine_lo_dim.z)};
    GpuArray<int,3> fine_hi {AMREX_D_DECL(fine_hi_dim.x, fine_hi_dim.y, fine_hi_dim.z)};

    for (MFIter mfi(coarse_bndry_area, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& coarse_area_arr = coarse_bndry_area.array(mfi);
        auto const& coarse_cent_arr = coarse_bndry_cent.array(mfi);
        auto const& coarse_norm_arr = coarse_bndry_norm.array(mfi);
        auto const& fine_area_arr = fine_bndry_area.const_array(mfi);
        auto const& fine_cent_arr = fine_bndry_cent.const_array(mfi);
        auto const& fine_norm_arr = fine_bndry_norm.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int idx_all[3] = {i,j,k};
            int raw_lo[3];
            int raw_hi[3];
            int bounds_lo[3];
            int bounds_hi[3];

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                raw_lo[d] = ratio[d]*idx_all[d] + offset[d];
                raw_hi[d] = raw_lo[d] + ratio[d] - 1;
                bounds_lo[d] = amrex::max(raw_lo[d], fine_lo[d]);
                bounds_hi[d] = amrex::min(raw_hi[d], fine_hi[d]);
                if (d == direction) {
                    bounds_hi[d] = bounds_lo[d];
                }
            }

            if (bounds_lo[0] > bounds_hi[0] ||
                bounds_lo[1] > bounds_hi[1] ||
                bounds_lo[2] > bounds_hi[2]) {
                coarse_area_arr(i,j,k) = Real(0.0);
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    coarse_cent_arr(i,j,k,n) = Real(0.0);
                    coarse_norm_arr(i,j,k,n) = Real(0.0);
                }
                return;
            }

            Real area_sum = Real(0.0);
            GpuArray<Real,AMREX_SPACEDIM> cent_accum {AMREX_D_DECL(Real(0.0), Real(0.0), Real(0.0))};
            GpuArray<Real,AMREX_SPACEDIM> norm_accum {AMREX_D_DECL(Real(0.0), Real(0.0), Real(0.0))};

            for (int kk_raw = raw_lo[2]; kk_raw <= raw_hi[2]; ++kk_raw) {
                int kk = kk_raw;
                if (direction == 2) {
                    kk = (kk < fine_lo[2]) ? (2*fine_lo[2] - kk)
                       : (kk > fine_hi[2]) ? (2*fine_hi[2] - kk) : kk;
                } else if (kk < fine_lo[2] || kk > fine_hi[2]) {
                    continue;
                }
                for (int jj_raw = raw_lo[1]; jj_raw <= raw_hi[1]; ++jj_raw) {
                    int jj = jj_raw;
                    if (direction == 1) {
                        jj = (jj < fine_lo[1]) ? (2*fine_lo[1] - jj)
                           : (jj > fine_hi[1]) ? (2*fine_hi[1] - jj) : jj;
                    } else if (jj < fine_lo[1] || jj > fine_hi[1]) {
                        continue;
                    }
                    for (int ii_raw = raw_lo[0]; ii_raw <= raw_hi[0]; ++ii_raw) {
                        int ii = ii_raw;
                        if (direction == 0) {
                            ii = (ii < fine_lo[0]) ? (2*fine_lo[0] - ii)
                               : (ii > fine_hi[0]) ? (2*fine_hi[0] - ii) : ii;
                        } else if (ii < fine_lo[0] || ii > fine_hi[0]) {
                            continue;
                        }
                        Real area = fine_area_arr(ii,jj,kk);
                        area_sum += area;
                        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                            cent_accum[n] += area * fine_cent_arr(ii,jj,kk,n);
                            norm_accum[n] += area * fine_norm_arr(ii,jj,kk,n);
                        }
                    }
                }
            }
            int const spans_i = amrex::max(0, bounds_hi[0] - bounds_lo[0] + 1);
            int const spans_j = amrex::max(0, bounds_hi[1] - bounds_lo[1] + 1);
            int const spans_k = amrex::max(0, bounds_hi[2] - bounds_lo[2] + 1);
            GpuArray<int,3> spans {AMREX_D_DECL(spans_i, spans_j, spans_k)};
            int norm_count = 1;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (d == direction) { continue; }
                norm_count *= spans[d];
            }
            Real const boundary_norm = static_cast<Real>(norm_count);
            Real const coarse_fraction = (boundary_norm > Real(0.0))
                                         ? area_sum / boundary_norm : Real(0.0);
            coarse_area_arr(i,j,k) = coarse_fraction;

            if (area_sum > Real(0.0)) {
                Real inv = Real(1.0) / area_sum;
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    coarse_cent_arr(i,j,k,n) = cent_accum[n] * inv;
                    coarse_norm_arr(i,j,k,n) = norm_accum[n] * inv;
                }
            } else {
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    coarse_cent_arr(i,j,k,n) = Real(0.0);
                    coarse_norm_arr(i,j,k,n) = Real(0.0);
                }
            }
        });
    }
}

void
fillStaggeredAreaFracAndFaceCent (EBStaggeredData& stag,
                                  const EBDataCollection& fine_data,
                                  const Geometry& fine_geom,
                                  int direction)
{
    auto coarse_area = stag.getAreaFrac();
    auto coarse_face = stag.getFaceCent();
    auto coarse_area_aligned = stag.getAreaFracAligned();
    auto coarse_face_aligned = stag.getFaceCentAligned();
    auto fine_area = fine_data.getAreaFrac();
    auto fine_face = fine_data.getFaceCent();
    GpuArray<int,AMREX_SPACEDIM> ratio {AMREX_D_DECL(2,2,2)};
    GpuArray<int,AMREX_SPACEDIM> offset {AMREX_D_DECL(0,0,0)};
    offset[direction] = -ratio[direction]/2;

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        MultiFab* carea = coarse_area[idir];
        MultiFab* carea_aligned = coarse_area_aligned[idir];
        const MultiCutFab* farea = fine_area[idir];
        if (carea == nullptr || farea == nullptr) { continue; }

        MultiFab* cface = coarse_face[idir];
        MultiFab* cface_aligned = coarse_face_aligned[idir];
        const MultiCutFab* fface = fine_face[idir];
        GpuArray<int,AMREX_SPACEDIM-1> face_axes;
        {
            int n = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (d == idir) { continue; }
                face_axes[n++] = d;
            }
        }

        IntVect face_type = IntVect::TheZeroVector();
        face_type[direction] = 1;
        face_type[idir] = 1;
        Box face_domain = amrex::convert(fine_geom.Domain(), face_type);
        Dim3 const face_lo_dim = lbound(face_domain);
        Dim3 const face_hi_dim = ubound(face_domain);
        GpuArray<int,3> face_lo {AMREX_D_DECL(face_lo_dim.x, face_lo_dim.y, face_lo_dim.z)};
        GpuArray<int,3> face_hi {AMREX_D_DECL(face_hi_dim.x, face_hi_dim.y, face_hi_dim.z)};

        for (MFIter mfi(*carea, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox(face_type);
            auto const& fine_area_arr = farea->const_array(mfi);
            auto const& coarse_area_arr = carea->array(mfi);
            Array4<Real const> fine_face_arr;
            Array4<Real> coarse_face_arr;
            bool const fill_face = (cface != nullptr) && (fface != nullptr);
            if (fill_face) {
                fine_face_arr = fface->const_array(mfi);
                coarse_face_arr = cface->array(mfi);
            }

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                int idx_all[3] = {i,j,k};
                int raw_lo[3];
                int raw_hi[3];
                GpuArray<int,3> dom_lo = face_lo;
                GpuArray<int,3> dom_hi = face_hi;
                GpuArray<int,3> bounds_lo = dom_lo;
                GpuArray<int,3> bounds_hi = dom_hi;

                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    raw_lo[d] = ratio[d]*idx_all[d] + offset[d];
                    raw_hi[d] = raw_lo[d] + ratio[d] - 1;
                    if (d == idir) {
                        raw_hi[d] = raw_lo[d];
                    }
                    bounds_lo[d] = amrex::max(raw_lo[d], dom_lo[d]);
                    bounds_hi[d] = amrex::min(raw_hi[d], dom_hi[d]);
                }

#if (AMREX_SPACEDIM == 2)
                bounds_lo[2] = dom_lo[2];
                bounds_hi[2] = dom_hi[2];
#endif

                if (bounds_lo[0] > bounds_hi[0] ||
                    bounds_lo[1] > bounds_hi[1] ||
                    bounds_lo[2] > bounds_hi[2]) {
                    coarse_area_arr(i,j,k) = Real(0.0);
                    if (fill_face) {
                        for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                            coarse_face_arr(i,j,k,n) = Real(0.0);
                        }
                    }
                    return;
                }

                Real sum_area = Real(0.0);
                Real face_w = Real(0.0);
                GpuArray<Real,3> face_accum {Real(0.0), Real(0.0), Real(0.0)};

                for (int kk_raw = raw_lo[2]; kk_raw <= raw_hi[2]; ++kk_raw) {
                    int kk = kk_raw;
                    if (direction == 2) {
                        kk = (kk < dom_lo[2]) ? (2*dom_lo[2] - kk)
                           : (kk > dom_hi[2]) ? (2*dom_hi[2] - kk) : kk;
                    } else if (kk < dom_lo[2] || kk > dom_hi[2]) {
                        continue;
                    }
                    for (int jj_raw = raw_lo[1]; jj_raw <= raw_hi[1]; ++jj_raw) {
                        int jj = jj_raw;
                        if (direction == 1) {
                            jj = (jj < dom_lo[1]) ? (2*dom_lo[1] - jj)
                               : (jj > dom_hi[1]) ? (2*dom_hi[1] - jj) : jj;
                        } else if (jj < dom_lo[1] || jj > dom_hi[1]) {
                            continue;
                        }
                        for (int ii_raw = raw_lo[0]; ii_raw <= raw_hi[0]; ++ii_raw) {
                            int ii = ii_raw;
                            if (direction == 0) {
                                ii = (ii < dom_lo[0]) ? (2*dom_lo[0] - ii)
                                   : (ii > dom_hi[0]) ? (2*dom_hi[0] - ii) : ii;
                            } else if (ii < dom_lo[0] || ii > dom_hi[0]) {
                                continue;
                            }
                            Real aval = fine_area_arr(ii,jj,kk);
                            sum_area += aval;
                            if (fill_face) {
                                for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                                    face_accum[n] += aval * fine_face_arr(ii,jj,kk,n);
                                }
                                face_w += aval;
                            }
                        }
                    }
                }

                int count_perp = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (d == idir) { continue; }
                    int span = 0;
                    if (d == direction) {
                        span = raw_hi[d] - raw_lo[d] + 1;
                    } else {
                        span = amrex::max(0, bounds_hi[d] - bounds_lo[d] + 1);
                    }
                    count_perp *= span;
                }
                Real const norm = static_cast<Real>(count_perp);
                Real avg = (norm > Real(0.0)) ? sum_area / norm : Real(0.0);
                coarse_area_arr(i,j,k) = avg;

                if (fill_face) {
                    if (face_w > Real(0.0)) {
                        Real invw = Real(1.0) / face_w;
                        for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                            Real const face_val = face_accum[n] * invw;
                            coarse_face_arr(i,j,k,n) = face_val;
                        }
                    } else {
                        for (int n = 0; n < AMREX_SPACEDIM-1; ++n) {
                            coarse_face_arr(i,j,k,n) = Real(0.0);
                        }
                    }
                }
            });
        }

        for (MFIter mfi(*carea_aligned, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& dst_bx = (*carea_aligned)[mfi].box();
            int const src_k = mfi.index();
            const Box src_area_bx = (*carea)[src_k].box();
            auto const& src_area_arr = carea->const_array(src_k);
            auto const& dst_area_arr = carea_aligned->array(mfi);

            ParallelFor(dst_bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect iv(AMREX_D_DECL(i,j,k));
                IntVect iv_hi = iv;
                iv_hi[direction] += 1;

                Real area_lo = src_area_bx.contains(iv) ? src_area_arr(iv) : Real(0.0);
                if (idir == direction) {
                    dst_area_arr(i,j,k) = area_lo;
                } else {
                    Real area_hi = src_area_bx.contains(iv_hi) ? src_area_arr(iv_hi) : Real(0.0);
                    dst_area_arr(i,j,k) = area_lo + area_hi;
                }
            });
        }

        if (cface != nullptr && cface_aligned != nullptr) {
            for (MFIter mfi(*cface_aligned, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& dst_bx = (*cface_aligned)[mfi].box();
                int const src_k = mfi.index();
                const Box src_area_bx = (*carea)[src_k].box();
                const Box src_face_bx = (*cface)[src_k].box();
                auto const& src_area_arr = carea->const_array(src_k);
                auto const& src_face_arr = cface->const_array(src_k);
                auto const& dst_face_arr = cface_aligned->array(mfi);

                ParallelFor(dst_bx, AMREX_SPACEDIM-1,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    IntVect iv(AMREX_D_DECL(i,j,k));
                    IntVect iv_hi = iv;
                    iv_hi[direction] += 1;

                    if (idir == direction) {
                        if (src_face_bx.contains(iv)) {
                            dst_face_arr(i,j,k,n) = src_face_arr(iv,n);
                        } else {
                            dst_face_arr(i,j,k,n) = Real(0.0);
                        }
                    } else {
                        int const axis = face_axes[n];
                        Real area_lo = src_area_bx.contains(iv) ? src_area_arr(iv) : Real(0.0);
                        Real area_hi = src_area_bx.contains(iv_hi) ? src_area_arr(iv_hi) : Real(0.0);
                        Real numer = Real(0.0);
                        if (src_face_bx.contains(iv)) {
                            Real val = src_face_arr(iv,n);
                            if (axis == direction) { val -= Real(0.5); }
                            numer += area_lo * val;
                        }
                        if (src_face_bx.contains(iv_hi)) {
                            Real val = src_face_arr(iv_hi,n);
                            if (axis == direction) { val += Real(0.5); }
                            numer += area_hi * val;
                        }
                        Real denom = area_lo + area_hi;
                        dst_face_arr(i,j,k,n) = (denom > Real(0.0)) ? numer / denom : Real(0.0);
                    }
                });
            }
        }

        if (!fine_geom.isPeriodic(direction)) {
            for (MFIter mfi(*carea_aligned, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box valid_bx = carea_aligned->boxArray()[mfi];
                const Box fab_bx = (*carea_aligned)[mfi].box();
                auto const& area_arr = carea_aligned->array(mfi);

                Box lo_slab(fab_bx);
                lo_slab.setBig(direction, valid_bx.smallEnd(direction)-1);
                if (lo_slab.ok()) {
                    ParallelFor(lo_slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        IntVect iv(AMREX_D_DECL(i,j,k));
                        IntVect iv_nearest = iv;
                        iv_nearest[direction] = valid_bx.smallEnd(direction);
                        area_arr(iv) = area_arr(iv_nearest);
                    });
                }

                Box hi_slab(fab_bx);
                hi_slab.setSmall(direction, valid_bx.bigEnd(direction)+1);
                if (hi_slab.ok()) {
                    ParallelFor(hi_slab, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        IntVect iv(AMREX_D_DECL(i,j,k));
                        IntVect iv_nearest = iv;
                        iv_nearest[direction] = valid_bx.bigEnd(direction);
                        area_arr(iv) = area_arr(iv_nearest);
                    });
                }
            }

            if (cface_aligned != nullptr) {
                for (MFIter mfi(*cface_aligned, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box valid_bx = cface_aligned->boxArray()[mfi];
                    const Box fab_bx = (*cface_aligned)[mfi].box();
                    auto const& face_arr = cface_aligned->array(mfi);

                    Box lo_slab(fab_bx);
                    lo_slab.setBig(direction, valid_bx.smallEnd(direction)-1);
                    if (lo_slab.ok()) {
                        ParallelFor(lo_slab, AMREX_SPACEDIM-1,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                        {
                            IntVect iv(AMREX_D_DECL(i,j,k));
                            IntVect iv_nearest = iv;
                            iv_nearest[direction] = valid_bx.smallEnd(direction);
                            face_arr(i,j,k,n) = face_arr(iv_nearest,n);
                        });
                    }

                    Box hi_slab(fab_bx);
                    hi_slab.setSmall(direction, valid_bx.bigEnd(direction)+1);
                    if (hi_slab.ok()) {
                        ParallelFor(hi_slab, AMREX_SPACEDIM-1,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                        {
                            IntVect iv(AMREX_D_DECL(i,j,k));
                            IntVect iv_nearest = iv;
                            iv_nearest[direction] = valid_bx.bigEnd(direction);
                            face_arr(i,j,k,n) = face_arr(iv_nearest,n);
                        });
                    }
                }
            }
        }

    }
}

} // namespace

void
EBFArrayBoxFactory::buildStaggeredData (int direction)
{
    if (direction < 0 || direction >= AMREX_SPACEDIM) { return; }
    if (m_stag_data[direction]) { return; }
    if (m_support == EBSupport::none) { return; }

    if (!m_stag_refined_data) {
        const EB2::Level* fine_level = getRefinedLevel(2);
        if (fine_level == nullptr) { return; }

        Geometry fine_geom = amrex::refine(m_geom, 2);
        BoxArray fine_ba = amrex::refine(this->boxArray(), 2);
        const DistributionMapping& fine_dm = this->DistributionMap();

        m_stag_refined_data = std::make_unique<EBDataCollection>(
            *fine_level, fine_geom, fine_ba, fine_dm, m_ngrow, m_support);
        m_stag_refined_geom = fine_geom;
        m_has_stag_refined_geom = true;
    }

    if (!m_has_stag_refined_geom) { return; }

    const EBDataCollection& fine_data = *m_stag_refined_data;
    const Geometry& fine_geom = m_stag_refined_geom;

    auto stag = std::make_unique<EBStaggeredData>(direction, m_geom,
                                                  this->boxArray(),
                                                  this->DistributionMap(),
                                                  m_ngrow, m_support);

    if (m_support >= EBSupport::volume) {
        stag->getCentroid().setVal(Real(0.0));
        fillStaggeredVolFracAndFlags(*stag, fine_data, fine_geom, direction);
        fillStaggeredCentroid(*stag, fine_data, fine_geom, direction);
    }
    if (m_support == EBSupport::full) {
        stag->getBndryCent().setVal(Real(0.0));
        stag->getBndryNormal().setVal(Real(0.0));
        stag->getBndryArea().setVal(Real(0.0));
        auto area = stag->getAreaFrac();
        auto face = stag->getFaceCent();
        auto area_aligned = stag->getAreaFracAligned();
        auto face_aligned = stag->getFaceCentAligned();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (area[idim]) { area[idim]->setVal(Real(0.0)); }
            if (face[idim]) { face[idim]->setVal(Real(0.0)); }
            if (area_aligned[idim]) { area_aligned[idim]->setVal(Real(0.0)); }
            if (face_aligned[idim]) { face_aligned[idim]->setVal(Real(0.0)); }
        }
        fillStaggeredBoundaryData(*stag, fine_data, fine_geom, direction);
        fillStaggeredAreaFracAndFaceCent(*stag, fine_data, fine_geom, direction);
    }

    m_stag_data[direction] = std::move(stag);
}


std::unique_ptr<EBFArrayBoxFactory>
makeEBFabFactory (const Geometry& a_geom,
                  const BoxArray& a_ba,
                  const DistributionMapping& a_dm,
                  const Vector<int>& a_ngrow, EBSupport a_support)
{
    const EB2::IndexSpace& index_space = EB2::IndexSpace::top();
    const EB2::Level& eb_level = index_space.getLevel(a_geom);
    return std::make_unique<EBFArrayBoxFactory>(eb_level, a_geom, a_ba, a_dm, a_ngrow, a_support);
}

std::unique_ptr<EBFArrayBoxFactory>
makeEBFabFactory (const EB2::Level* eb_level,
                  const BoxArray& a_ba,
                  const DistributionMapping& a_dm,
                  const Vector<int>& a_ngrow, EBSupport a_support)
{
    return std::make_unique<EBFArrayBoxFactory>(*eb_level, eb_level->Geom(),
                                                a_ba, a_dm, a_ngrow, a_support);
}

std::unique_ptr<EBFArrayBoxFactory>
makeEBFabFactory (const EB2::IndexSpace* index_space, const Geometry& a_geom,
                  const BoxArray& a_ba,
                  const DistributionMapping& a_dm,
                  const Vector<int>& a_ngrow, EBSupport a_support)
{
    const EB2::Level& eb_level = index_space->getLevel(a_geom);
    return std::make_unique<EBFArrayBoxFactory>(eb_level, a_geom,
                                                a_ba, a_dm, a_ngrow, a_support);
}

}
