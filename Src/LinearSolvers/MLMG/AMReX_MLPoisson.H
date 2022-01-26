#ifndef AMREX_MLPOISSON_H_
#define AMREX_MLPOISSON_H_
#include <AMReX_Config.H>

#include <AMReX_MLCellABecLap.H>
#include <AMReX_Array.H>
#include <AMReX_MLPoisson_K.H>

#include <limits>

namespace amrex {

// del dot grad phi

class MLPoisson
    : public MLCellABecLap
{
public:

    MLPoisson () noexcept {}
    MLPoisson (const Vector<Geometry>& a_geom,
               const Vector<BoxArray>& a_grids,
               const Vector<DistributionMapping>& a_dmap,
               const LPInfo& a_info = LPInfo(),
               const Vector<FabFactory<FArrayBox> const*>& a_factory = {});
    MLPoisson (const Vector<Geometry>& a_geom,
               const Vector<BoxArray>& a_grids,
               const Vector<DistributionMapping>& a_dmap,
               const Vector<iMultiFab const*>& a_overset_mask, // 1: unknown, 0: known
               const LPInfo& a_info = LPInfo(),
               const Vector<FabFactory<FArrayBox> const*>& a_factory = {});
    virtual ~MLPoisson ();

    MLPoisson (const MLPoisson&) = delete;
    MLPoisson (MLPoisson&&) = delete;
    MLPoisson& operator= (const MLPoisson&) = delete;
    MLPoisson& operator= (MLPoisson&&) = delete;

    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info = LPInfo(),
                 const Vector<FabFactory<FArrayBox> const*>& a_factory = {});

    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const Vector<iMultiFab const*>& a_overset_mask,
                 const LPInfo& a_info = LPInfo(),
                 const Vector<FabFactory<FArrayBox> const*>& a_factory = {});

    virtual void prepareForSolve () final override;
    virtual bool isSingular (int amrlev) const final override { return m_is_singular[amrlev]; }
    virtual bool isBottomSingular () const final override { return m_is_singular[0]; }
    virtual void Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const final override;
    virtual void Fapply_s (int amrlev, int mglev, fMultiFab& out, const fMultiFab& in) const final override;
    virtual void Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const final override;
    virtual void Fsmooth_s (int amrlev, int mglev, fMultiFab& sol, const fMultiFab& rhs, int redblack) const final override;
    virtual void FFlux (int amrlev, const MFIter& mfi,
                        const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                        const FArrayBox& sol, Location loc, const int face_only=0) const final override;

    virtual void normalize (int amrlev, int mglev, MultiFab& mf) const final override;

    virtual Real getAScalar () const final override { return  0.0; }
    virtual Real getBScalar () const final override { return -1.0; }
    virtual MultiFab const* getACoeffs (int /*amrlev*/, int /*mglev*/) const final override { return nullptr; }
    virtual Array<MultiFab const*,AMREX_SPACEDIM> getBCoeffs (int /*amrlev*/, int /*mglev*/) const final override
        { return {{ AMREX_D_DECL(nullptr,nullptr,nullptr)}}; }

    virtual std::unique_ptr<MLLinOp> makeNLinOp (int grid_size) const final override;

    virtual bool supportNSolve () const final override;

    virtual void copyNSolveSolution (MultiFab& dst, MultiFab const& src) const final override;

public: // for cuda

    template <typename MF>
    void FsmoothT (int amrlev, int mglev, MF& sol, const MF& rhs, int redblack,
                   BndryRegisterT<MF> const& undrrelxr) const;

    template <typename MF>
    void FapplyT (int amrlev, int mglev, MF& out, const MF& in) const;

private:

    Vector<int> m_is_singular;
};

template <typename MF>
void
MLPoisson::FsmoothT (int amrlev, int mglev, MF& sol, const MF& rhs, int redblack,
                     BndryRegisterT<MF> const& undrrelxr) const
{
    using T = typename MF::value_type;

    const auto& maskvals  = m_maskvals [amrlev][mglev];

    OrientationIter oitr;

    const auto& f0 = undrrelxr[oitr()]; ++oitr;
    const auto& f1 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 1)
    const auto& f2 = undrrelxr[oitr()]; ++oitr;
    const auto& f3 = undrrelxr[oitr()]; ++oitr;
#if (AMREX_SPACEDIM > 2)
    const auto& f4 = undrrelxr[oitr()]; ++oitr;
    const auto& f5 = undrrelxr[oitr()]; ++oitr;
#endif
#endif

    const MultiMask& mm0 = maskvals[0];
    const MultiMask& mm1 = maskvals[1];
#if (AMREX_SPACEDIM > 1)
    const MultiMask& mm2 = maskvals[2];
    const MultiMask& mm3 = maskvals[3];
#if (AMREX_SPACEDIM > 2)
    const MultiMask& mm4 = maskvals[4];
    const MultiMask& mm5 = maskvals[5];
#endif
#endif

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    AMREX_D_TERM(const T dhx = static_cast<T>(dxinv[0]*dxinv[0]);,
                 const T dhy = static_cast<T>(dxinv[1]*dxinv[1]);,
                 const T dhz = static_cast<T>(dxinv[2]*dxinv[2]););

#if (AMREX_SPACEDIM == 3)
    T const dh0 = get_d0(dhx, dhy, dhz);
    T const dh1 = get_d1(dhx, dhy, dhz);
#endif

#if (AMREX_SPACEDIM < 3)
    const T dx = static_cast<T>(m_geom[amrlev][mglev].CellSize(0));
    const T probxlo = static_cast<T>(m_geom[amrlev][mglev].ProbLo(0));
#endif

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.EnableTiling().SetDynamic(true);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(sol,mfi_info); mfi.isValid(); ++mfi)
    {
        const auto& m0 = mm0.array(mfi);
        const auto& m1 = mm1.array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& m2 = mm2.array(mfi);
        const auto& m3 = mm3.array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& m4 = mm4.array(mfi);
        const auto& m5 = mm5.array(mfi);
#endif
#endif

        const Box& tbx = mfi.tilebox();
        const Box& vbx = mfi.validbox();
        const auto& solnfab = sol.array(mfi);
        const auto& rhsfab  = rhs.array(mfi);

        const auto& f0fab = f0.array(mfi);
        const auto& f1fab = f1.array(mfi);
#if (AMREX_SPACEDIM > 1)
        const auto& f2fab = f2.array(mfi);
        const auto& f3fab = f3.array(mfi);
#if (AMREX_SPACEDIM > 2)
        const auto& f4fab = f4.array(mfi);
        const auto& f5fab = f5.array(mfi);
#endif
#endif

#if (AMREX_SPACEDIM == 1)
        if (m_overset_mask[amrlev][mglev]) {
            AMREX_ASSERT(!m_has_metric_term);
            const auto& osm = m_overset_mask[amrlev][mglev]->const_array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb_os(thread_box, solnfab, rhsfab, osm, dhx,
                                  f0fab, m0,
                                  f1fab, m1,
                                  vbx, redblack);
            });
        } else if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb_m(thread_box, solnfab, rhsfab, dhx,
                                 f0fab, m0,
                                 f1fab, m1,
                                 vbx, redblack,
                                 dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb(thread_box, solnfab, rhsfab, dhx,
                               f0fab, m0,
                               f1fab, m1,
                               vbx, redblack);
            });
        }
#endif

#if (AMREX_SPACEDIM == 2)
        if (m_overset_mask[amrlev][mglev]) {
            AMREX_ASSERT(!m_has_metric_term);
            const auto& osm = m_overset_mask[amrlev][mglev]->const_array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb_os(thread_box, solnfab, rhsfab, osm, dhx, dhy,
                                  f0fab, m0,
                                  f1fab, m1,
                                  f2fab, m2,
                                  f3fab, m3,
                                  vbx, redblack);
            });
        } else if (m_has_metric_term) {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb_m(thread_box, solnfab, rhsfab, dhx, dhy,
                                 f0fab, m0,
                                 f1fab, m1,
                                 f2fab, m2,
                                 f3fab, m3,
                                 vbx, redblack,
                                 dx, probxlo);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb(thread_box, solnfab, rhsfab, dhx, dhy,
                               f0fab, m0,
                               f1fab, m1,
                               f2fab, m2,
                               f3fab, m3,
                               vbx, redblack);
            });
        }

#endif

#if (AMREX_SPACEDIM == 3)
        if (m_overset_mask[amrlev][mglev]) {
            AMREX_ASSERT(!m_has_metric_term);
            const auto& osm = m_overset_mask[amrlev][mglev]->const_array(mfi);
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb_os(thread_box, solnfab, rhsfab, osm, dhx, dhy, dhz,
                                  f0fab, m0,
                                  f1fab, m1,
                                  f2fab, m2,
                                  f3fab, m3,
                                  f4fab, m4,
                                  f5fab, m5,
                                  vbx, redblack);
            });
        } else if (hasHiddenDimension()) {
            Box const& tbx_2d = compactify(tbx);
            Box const& vbx_2d = compactify(vbx);
            const auto& solnfab_2d = compactify(solnfab);
            const auto& rhsfab_2d = compactify(rhsfab);
            const auto& f0fab_2d = compactify(get_d0(f0fab,f1fab,f2fab));
            const auto& f1fab_2d = compactify(get_d1(f0fab,f1fab,f2fab));
            const auto& f2fab_2d = compactify(get_d0(f3fab,f4fab,f5fab));
            const auto& f3fab_2d = compactify(get_d1(f3fab,f4fab,f5fab));
            const auto& m0_2d = compactify(get_d0(m0,m1,m2));
            const auto& m1_2d = compactify(get_d1(m0,m1,m2));
            const auto& m2_2d = compactify(get_d0(m3,m4,m5));
            const auto& m3_2d = compactify(get_d1(m3,m4,m5));
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx_2d, thread_box,
            {
                TwoD::mlpoisson_gsrb(thread_box, solnfab_2d, rhsfab_2d, dh0, dh1,
                                     f0fab_2d, m0_2d,
                                     f1fab_2d, m1_2d,
                                     f2fab_2d, m2_2d,
                                     f3fab_2d, m3_2d,
                                     vbx_2d, redblack);
            });
        } else {
            AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
            {
                mlpoisson_gsrb(thread_box, solnfab, rhsfab, dhx, dhy, dhz,
                               f0fab, m0,
                               f1fab, m1,
                               f2fab, m2,
                               f3fab, m3,
                               f4fab, m4,
                               f5fab, m5,
                               vbx, redblack);
            });
        }
#endif
    }
}

template <typename MF>
void
MLPoisson::FapplyT (int amrlev, int mglev, MF& out, const MF& in) const
{
    using T = typename MF::value_type;

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();

    AMREX_D_TERM(const T dhx = static_cast<T>(dxinv[0]*dxinv[0]);,
                 const T dhy = static_cast<T>(dxinv[1]*dxinv[1]);,
                 const T dhz = static_cast<T>(dxinv[2]*dxinv[2]););

#if (AMREX_SPACEDIM == 3)
    T const dh0 = get_d0(dhx, dhy, dhz);
    T const dh1 = get_d1(dhx, dhy, dhz);
#endif

#if (AMREX_SPACEDIM < 3)
    const T dx = static_cast<T>(m_geom[amrlev][mglev].CellSize(0));
    const T probxlo = static_cast<T>(m_geom[amrlev][mglev].ProbLo(0));
#endif

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion() && out.isFusingCandidate() && !hasHiddenDimension()) {
        auto const& xma = in.const_arrays();
        auto const& yma = out.arrays();
        if (m_overset_mask[amrlev][mglev]) {
            AMREX_ASSERT(!m_has_metric_term);
            const auto& osmma = m_overset_mask[amrlev][mglev]->const_arrays();
            ParallelFor(out,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                amrex::ignore_unused(j,k);
                mlpoisson_adotx_os(AMREX_D_DECL(i,j,k), yma[box_no], xma[box_no], osmma[box_no],
                                   AMREX_D_DECL(dhx,dhy,dhz));
            });
        } else {
#if (AMREX_SPACEDIM < 3)
            if (m_has_metric_term) {
                ParallelFor(out,
                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                {
                    amrex::ignore_unused(j,k);
                    mlpoisson_adotx_m(AMREX_D_DECL(i,j,k), yma[box_no], xma[box_no],
                                      AMREX_D_DECL(dhx,dhy,dhz), dx, probxlo);
                });
            } else
#endif
            {
                ParallelFor(out,
                [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
                {
                    amrex::ignore_unused(j,k);
                    mlpoisson_adotx(AMREX_D_DECL(i,j,k), yma[box_no], xma[box_no],
                                    AMREX_D_DECL(dhx,dhy,dhz));
                });
            }
        }
        Gpu::streamSynchronize();
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto& xfab = in.array(mfi);
            const auto& yfab = out.array(mfi);

            if (m_overset_mask[amrlev][mglev]) {
                AMREX_ASSERT(!m_has_metric_term);
                const auto& osm = m_overset_mask[amrlev][mglev]->const_array(mfi);
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
                {
                    amrex::ignore_unused(j,k);
                    mlpoisson_adotx_os(AMREX_D_DECL(i,j,k), yfab, xfab, osm,
                                       AMREX_D_DECL(dhx,dhy,dhz));
                });
            } else {
#if (AMREX_SPACEDIM == 3)
                if (hasHiddenDimension()) {
                    Box const& bx2d = compactify(bx);
                    const auto& xfab2d = compactify(xfab);
                    const auto& yfab2d = compactify(yfab);
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx2d, i, j, k,
                    {
                        amrex::ignore_unused(k);
                        TwoD::mlpoisson_adotx(i, j, yfab2d, xfab2d, dh0, dh1);
                    });
                } else {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                    {
                        mlpoisson_adotx(i, j, k, yfab, xfab, dhx, dhy, dhz);
                    });
                }
#elif (AMREX_SPACEDIM == 2)
                if (m_has_metric_term) {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                    {
                        amrex::ignore_unused(k);
                        mlpoisson_adotx_m(i, j, yfab, xfab, dhx, dhy, dx, probxlo);
                    });
                } else {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                    {
                        amrex::ignore_unused(k);
                        mlpoisson_adotx(i, j, yfab, xfab, dhx, dhy);
                    });
                }
#elif (AMREX_SPACEDIM == 1)
                if (m_has_metric_term) {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                    {
                        amrex::ignore_unused(j,k);
                        mlpoisson_adotx_m(i, yfab, xfab, dhx, dx, probxlo);
                    });
                } else {
                    AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k,
                    {
                        amrex::ignore_unused(j,k);
                        mlpoisson_adotx(i, yfab, xfab, dhx);
                    });
                }
#endif
            }
        }
    }
}

}

#endif
