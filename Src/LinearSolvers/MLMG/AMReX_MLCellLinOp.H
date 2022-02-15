#ifndef AMREX_ML_CELL_LINOP_H_H
#define AMREX_ML_CELL_LINOP_H_H
#include <AMReX_Config.H>

#include <AMReX_MLLinOp.H>

namespace amrex {

namespace detail {
    template <typename T>
    struct ABCTag {
        Array4<T> fab;
        Array4<Real const> bcval_lo;
        Array4<Real const> bcval_hi;
        Array4<int const> mask_lo;
        Array4<int const> mask_hi;
        T bcloc_lo;
        T bcloc_hi;
        Box bx;
        BoundCond bctype_lo;
        BoundCond bctype_hi;
        int blen;
        int comp;
        int dir;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Box const& box() const noexcept { return bx; }
    };

    template <typename T>
    struct PSTag {
        Array4<T> flo;
        Array4<T> fhi;
        Array4<int const> mlo;
        Array4<int const> mhi;
        Real bcllo;
        Real bclhi;
        Box bx;
        BoundCond bctlo;
        BoundCond bcthi;
        int blen;
        int comp;
        int dir;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Box const& box() const noexcept { return bx; }
    };

#ifdef AMREX_USE_EB
    template <typename T>
    struct PSEBTag {
        Array4<T> flo;
        Array4<T> fhi;
        Array4<Real const> ap;
        Array4<int const> mlo;
        Array4<int const> mhi;
        T bcllo;
        T bclhi;
        Box bx;
        BoundCond bctlo;
        BoundCond bcthi;
        int blen;
        int comp;
        int dir;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        Box const& box() const noexcept { return bx; }
    };
#endif
}

class MLCellLinOp
    : public MLLinOp
{
public:

    friend class MLMG;
    friend class MLCGSolver;

    MLCellLinOp ();
    virtual ~MLCellLinOp ();

    MLCellLinOp (const MLCellLinOp&) = delete;
    MLCellLinOp (MLCellLinOp&&) = delete;
    MLCellLinOp& operator= (const MLCellLinOp&) = delete;
    MLCellLinOp& operator= (MLCellLinOp&&) = delete;

    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info = LPInfo(),
                 const Vector<FabFactory<FArrayBox> const*>& a_factory = {});

    virtual void setLevelBC (int amrlev, const MultiFab* levelbcdata,
                             const MultiFab* robinbc_a = nullptr,
                             const MultiFab* robinbc_b = nullptr,
                             const MultiFab* robinbc_f = nullptr) final override;

    virtual bool needsUpdate () const override {
        return MLLinOp::needsUpdate();
    }
    virtual void update () override;

#ifdef AMREX_SOFT_PERF_COUNTERS
    struct Counters
    {
        size_t smooth_n = 0;
        size_t smooth_cell_n = 0;
        size_t apply_n = 0;
        size_t apply_cell_n = 0;
        size_t restrict_n = 0;
        size_t restrict_cell_n = 0;
        size_t interp_n = 0;
        size_t interp_cell_n = 0;

        void reset () {
            smooth_n = 0;
            smooth_cell_n = 0;
            apply_n = 0;
            apply_cell_n = 0;
            restrict_n = 0;
            restrict_cell_n = 0;
            interp_n = 0;
            interp_cell_n = 0;
        }

        void smooth (const MultiFab & mf) {
            smooth_n++;
            smooth_cell_n += mf.boxArray().numPts();
        }
        void apply (const MultiFab & mf) {
            apply_n++;
            apply_cell_n += mf.boxArray().numPts();
        }
        void restrict (const MultiFab & mf) {
            restrict_n++;
            restrict_cell_n += mf.boxArray().numPts();
        }
        void interpolate (const MultiFab & mf) {
            interp_n++;
            interp_cell_n += mf.boxArray().numPts();
        }

        void print () {
            Print() << ParallelContext::MyProcSub() << ": perf: MLCellLinOp::smooth()        called " << smooth_n
                    << " times on " << smooth_cell_n << " cells total" << std::endl;
            Print() << ParallelContext::MyProcSub() << ": perf: MLCellLinOp::apply()         called " << apply_n
                    << " times on " << apply_cell_n << " cells total" << std::endl;
            Print() << ParallelContext::MyProcSub() << ": perf: MLCellLinOp::restriction()   called " << restrict_n
                    << " times on " << restrict_cell_n << " cells total" << std::endl;
            Print() << ParallelContext::MyProcSub() << ": perf: MLCellLinOp::interpolation() called " << interp_n
                    << " times on " << interp_cell_n << " cells total" << std::endl;
        }
    };

    // software counters for performance analysis
    static Counters perf_counters;
#endif

    virtual bool isCrossStencil () const { return true; }
    virtual bool isTensorOp () const { return false; }

    void updateSolBC (int amrlev, const MultiFab& crse_bcdata) const;
    void updateCorBC (int amrlev, const MultiFab& crse_bcdata) const;

    virtual void applyBC (int amrlev, int mglev, MultiFab& in, BCMode bc_mode, StateMode s_mode,
                          const MLMGBndry* bndry=nullptr, bool skip_fillboundary=false) const;

    virtual void applyBC_s (int amrlev, int mglev, fMultiFab& in, BCMode bc_mode, StateMode s_mode,
                            const MLMGBndry* bndry=nullptr, bool skip_fillboundary=false) const;

    BoxArray makeNGrids (int grid_size) const;

    virtual void restriction (int, int, MultiFab& crse, MultiFab& fine) const override;
    virtual void restriction_d2s (int, int, fMultiFab& crse, MultiFab& fine) const override;
    virtual void restriction_s2d (int, int, MultiFab& crse, fMultiFab& fine) const override;
    virtual void restriction_s2s (int, int, fMultiFab& crse, fMultiFab& fine) const override;

    virtual void interpolation (int amrlev, int fmglev, MultiFab& fine, const MultiFab& crse) const override;
    virtual void interpolation_d2s (int amrlev, int fmglev, fMultiFab& fine, const MultiFab& crse) const override;
    virtual void interpolation_s2d (int amrlev, int fmglev, MultiFab& fine, const fMultiFab& crse) const override;
    virtual void interpolation_s2s (int amrlev, int fmglev, fMultiFab& fine, const fMultiFab& crse) const override;

    virtual void averageDownSolutionRHS (int camrlev, MultiFab& crse_sol, MultiFab& crse_rhs,
                                         const MultiFab& fine_sol, const MultiFab& fine_rhs) override;

    virtual void apply (int amrlev, int mglev, MultiFab& out, MultiFab& in, BCMode bc_mode,
                        StateMode s_mode, const MLMGBndry* bndry=nullptr) const override;
    virtual void apply_s (int amrlev, int mglev, fMultiFab& out, fMultiFab& in, BCMode bc_mode,
                          StateMode s_mode, const MLMGBndry* bndry=nullptr) const override;

    virtual void smooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs,
                         bool skip_fillboundary=false) const final override;
    virtual void smooth_s (int amrlev, int mglev, fMultiFab& sol, const fMultiFab& rhs,
                           bool skip_fillboundary=false) const final override;

    virtual void solutionResidual (int amrlev, MultiFab& resid, MultiFab& x, const MultiFab& b,
                                   const MultiFab* crse_bcdata=nullptr) override;

    virtual void correctionResidual (int amrlev, int mglev, MultiFab& resid, MultiFab& x, const MultiFab& b,
                                     BCMode bc_mode, const MultiFab* crse_bcdata=nullptr) final override;

    virtual void correctionResidual_s (int amrlev, int mglev, fMultiFab& resid, fMultiFab& x,
                                       const fMultiFab& b) const final override;

    // The assumption is crse_sol's boundary has been filled, but not fine_sol.
    virtual void reflux (int crse_amrlev,
                         MultiFab& res, const MultiFab& crse_sol, const MultiFab&,
                         MultiFab&, MultiFab& fine_sol, const MultiFab&) const final override;
    virtual void compFlux (int amrlev, const Array<MultiFab*,AMREX_SPACEDIM>& fluxes,
                           MultiFab& sol, Location loc) const override;
    virtual void compGrad (int amrlev, const Array<MultiFab*,AMREX_SPACEDIM>& grad,
                           MultiFab& sol, Location loc) const override;

    virtual void applyMetricTerm (int amrlev, int mglev, MultiFab& rhs) const final override;
    virtual void unapplyMetricTerm (int amrlev, int mglev, MultiFab& rhs) const final override;
    virtual void fillSolutionBC (int amrlev, MultiFab& sol, const MultiFab* crse_bcdata=nullptr) final override;

    virtual void prepareForSolve () override;

    virtual Real xdoty (int amrlev, int mglev, const MultiFab& x, const MultiFab& y, bool local) const final override;

    virtual void Fapply (int amrlev, int mglev, MultiFab& out, const MultiFab& in) const = 0;
    virtual void Fapply_s (int /*amrlev*/, int /*mglev*/, fMultiFab& /*out*/, const fMultiFab& /*in*/) const {}
    virtual void Fsmooth (int amrlev, int mglev, MultiFab& sol, const MultiFab& rhs, int redblack) const = 0;
    virtual void Fsmooth_s (int /*amrlev*/, int /*mglev*/, fMultiFab& /*sol*/, const fMultiFab& /*rhs*/, int /*redblack*/) const {}
    virtual void FFlux (int amrlev, const MFIter& mfi,
                        const Array<FArrayBox*,AMREX_SPACEDIM>& flux,
                        const FArrayBox& sol, Location loc, const int face_only=0) const = 0;

    struct BCTL {
        BoundCond type;
        Real location;
    };

public: // for CUDA

    template <typename MF>
    void prepareForSolveT (int amrlev, int mglev, BndryRegisterT<MF>& undrrelxr);

    template <typename MF>
    void applyBCT (int amrlev, int mglev, MF& in, BCMode bc_mode, const MLMGBndry* bndry,
                   bool skip_fillboundary) const;

    template <typename FMF, typename CMF>
    void interpolationT (int amrlev, int fmglev, FMF& fine, CMF const& crse) const;

protected:

    bool m_has_metric_term = false;

    Vector<std::unique_ptr<MLMGBndry> >   m_bndry_sol;
    Vector<std::unique_ptr<BndryRegister> > m_crse_sol_br;

    Vector<std::unique_ptr<MLMGBndry> > m_bndry_cor;
    Vector<std::unique_ptr<BndryRegister> > m_crse_cor_br;

    // In case of agglomeration, coarse MG grids on amr level 0 are
    // not simply coarsened from fine MG grids.  So we need to build
    // bcond and bcloc for each MG level.
    using RealTuple = Array<Real,2*BL_SPACEDIM>;
    using BCTuple   = Array<BoundCond,2*BL_SPACEDIM>;
    class BndryCondLoc
    {
    public:
        BndryCondLoc (const BoxArray& ba, const DistributionMapping& dm, int ncomp);

        void setLOBndryConds (const Geometry& geom, const Real* dx,
                              const Vector<Array<BCType,AMREX_SPACEDIM> >& lobc,
                              const Vector<Array<BCType,AMREX_SPACEDIM> >& hibc,
                              int ratio, const RealVect& interior_bloc,
                              const Array<Real,AMREX_SPACEDIM>& domain_bloc_lo,
                              const Array<Real,AMREX_SPACEDIM>& domain_bloc_hi);

        const Vector<BCTuple>& bndryConds (const MFIter& mfi) const noexcept {
            return bcond[mfi];
        }
        const Vector<RealTuple>& bndryLocs (const MFIter& mfi) const noexcept {
            return bcloc[mfi];
        }
        const BCTuple& bndryConds (const MFIter& mfi, int icomp) const noexcept {
            return bcond[mfi][icomp];
        }
        const RealTuple& bndryLocs (const MFIter& mfi, int icomp) const noexcept {
            return bcloc[mfi][icomp];
        }
        GpuArray<BCTL,2*AMREX_SPACEDIM> const* getBCTLPtr (const MFIter& mfi) const noexcept {
            return bctl[mfi];
        }
    private:
        LayoutData<Vector<BCTuple> >   bcond;
        LayoutData<Vector<RealTuple> > bcloc;
        LayoutData<GpuArray<BCTL,2*AMREX_SPACEDIM>*> bctl;
        Gpu::DeviceVector<GpuArray<BCTL,2*AMREX_SPACEDIM> > bctl_dv;
        int m_ncomp;
    };
    Vector<Vector<std::unique_ptr<BndryCondLoc> > > m_bcondloc;

    Vector<std::unique_ptr<MultiFab> > m_robin_bcval;

    // used to save interpolation coefficients of the first interior cells
    mutable Vector<Vector<BndryRegister> > m_undrrelxr;
    mutable Vector<Vector<BndryRegisterT<fMultiFab> > > m_undrrelxr_s;

    // boundary cell flags for covered, not_covered, outside_domain
    Vector<Vector<Array<MultiMask,2*AMREX_SPACEDIM> > > m_maskvals;

    mutable Vector<YAFluxRegister> m_fluxreg;

private:

    void defineAuxData ();
    void defineBC ();
};

template <typename MF>
void
MLCellLinOp::prepareForSolveT (int amrlev, int mglev, BndryRegisterT<MF>& undrrelxr)
{
    const int imaxorder = maxorder;
    const int ncomp = getNComp();
    const int hidden_direction = hiddenDirection();
    const auto& bcondloc = *m_bcondloc[amrlev][mglev];
    const auto& maskvals = m_maskvals[amrlev][mglev];

    const Real dxi = m_geom[amrlev][mglev].InvCellSize(0);
    const Real dyi = (AMREX_SPACEDIM >= 2) ? m_geom[amrlev][mglev].InvCellSize(1) : Real(1.0);
    const Real dzi = (AMREX_SPACEDIM == 3) ? m_geom[amrlev][mglev].InvCellSize(2) : Real(1.0);

    MultiFab foo(m_grids[amrlev][mglev], m_dmap[amrlev][mglev], ncomp, 0, MFInfo().SetAlloc(false));

#ifdef AMREX_USE_EB
    auto factory = dynamic_cast<EBFArrayBoxFactory const*>(m_factory[amrlev][mglev].get());
    const FabArray<EBCellFlagFab>* flags =
        (factory) ? &(factory->getMultiEBCellFlagFab()) : nullptr;
    auto area = (factory) ? factory->getAreaFrac()
        : Array<const MultiCutFab*,AMREX_SPACEDIM>{AMREX_D_DECL(nullptr,nullptr,nullptr)};
#endif

#ifdef AMREX_USE_GPU
    using T = typename MF::value_type;

    if (Gpu::inLaunchRegion()) {
#ifdef AMREX_USE_EB
        if (factory) {
            Vector<detail::PSEBTag<T> > tags;
            tags.reserve(foo.local_size()*AMREX_SPACEDIM*ncomp);

            for (MFIter mfi(foo); mfi.isValid(); ++mfi)
            {
                const Box& vbx = mfi.validbox();

                const auto & bdlv = bcondloc.bndryLocs(mfi);
                const auto & bdcv = bcondloc.bndryConds(mfi);

                auto fabtyp = (flags) ? (*flags)[mfi].getType(vbx) : FabType::regular;

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    if (idim != hidden_direction && fabtyp != FabType::covered) {
                        const Orientation olo(idim,Orientation::low);
                        const Orientation ohi(idim,Orientation::high);
                        auto const& ap = (fabtyp == FabType::singlevalued)
                            ? area[idim]->const_array(mfi) : Array4<Real const>{};
                        for (int icomp = 0; icomp < ncomp; ++icomp) {
                            tags.emplace_back(detail::PSEBTag<T>{
                                                      undrrelxr[olo].array(mfi),
                                                      undrrelxr[ohi].array(mfi),
                                                      ap,
                                                      maskvals[olo].const_array(mfi),
                                                      maskvals[ohi].const_array(mfi),
                                                      static_cast<T>(bdlv[icomp][olo]),
                                                      static_cast<T>(bdlv[icomp][ohi]),
                                                      amrex::adjCell(vbx,olo),
                                                      bdcv[icomp][olo], bdcv[icomp][ohi],
                                                      vbx.length(idim), icomp, idim});
                        }
                    }
                }
            }

            ParallelFor(tags,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, detail::PSEBTag<T> const& tag) noexcept
            {
                if (tag.ap) {
                    if (tag.dir == 0)
                    {
                        mllinop_comp_interp_coef0_x_eb
                            (0, i           , j, k, tag.blen, tag.flo, tag.mlo, tag.ap,
                             tag.bctlo, tag.bcllo, imaxorder, dxi, tag.comp);
                        mllinop_comp_interp_coef0_x_eb
                            (1, i+tag.blen+1, j, k, tag.blen, tag.fhi, tag.mhi, tag.ap,
                             tag.bcthi, tag.bclhi, imaxorder, dxi, tag.comp);
                    }
#if (AMREX_SPACEDIM > 1)
                    else
#if (AMREX_SPACEDIM > 2)
                        if (tag.dir == 1)
#endif
                        {
                            mllinop_comp_interp_coef0_y_eb
                                (0, i, j           , k, tag.blen, tag.flo, tag.mlo, tag.ap,
                                 tag.bctlo, tag.bcllo, imaxorder, dyi, tag.comp);
                            mllinop_comp_interp_coef0_y_eb
                                (1, i, j+tag.blen+1, k, tag.blen, tag.fhi, tag.mhi, tag.ap,
                                 tag.bcthi, tag.bclhi, imaxorder, dyi, tag.comp);
                        }
#if (AMREX_SPACEDIM > 2)
                        else {
                            mllinop_comp_interp_coef0_z_eb
                                (0, i, j, k           , tag.blen, tag.flo, tag.mlo, tag.ap,
                                 tag.bctlo, tag.bcllo, imaxorder, dzi, tag.comp);
                            mllinop_comp_interp_coef0_z_eb
                                (1, i, j, k+tag.blen+1, tag.blen, tag.fhi, tag.mhi, tag.ap,
                                 tag.bcthi, tag.bclhi, imaxorder, dzi, tag.comp);
                        }
#endif
#endif
                } else {
                    if (tag.dir == 0)
                    {
                        mllinop_comp_interp_coef0_x
                            (0, i           , j, k, tag.blen, tag.flo, tag.mlo,
                             tag.bctlo, tag.bcllo, imaxorder, dxi, tag.comp);
                        mllinop_comp_interp_coef0_x
                            (1, i+tag.blen+1, j, k, tag.blen, tag.fhi, tag.mhi,
                             tag.bcthi, tag.bclhi, imaxorder, dxi, tag.comp);
                    }
#if (AMREX_SPACEDIM > 1)
                    else
#if (AMREX_SPACEDIM > 2)
                    if (tag.dir == 1)
#endif
                    {
                        mllinop_comp_interp_coef0_y
                            (0, i, j           , k, tag.blen, tag.flo, tag.mlo,
                             tag.bctlo, tag.bcllo, imaxorder, dyi, tag.comp);
                        mllinop_comp_interp_coef0_y
                            (1, i, j+tag.blen+1, k, tag.blen, tag.fhi, tag.mhi,
                             tag.bcthi, tag.bclhi, imaxorder, dyi, tag.comp);
                    }
#if (AMREX_SPACEDIM > 2)
                    else {
                        mllinop_comp_interp_coef0_z
                            (0, i, j, k           , tag.blen, tag.flo, tag.mlo,
                             tag.bctlo, tag.bcllo, imaxorder, dzi, tag.comp);
                        mllinop_comp_interp_coef0_z
                            (1, i, j, k+tag.blen+1, tag.blen, tag.fhi, tag.mhi,
                             tag.bcthi, tag.bclhi, imaxorder, dzi, tag.comp);
                    }
#endif
#endif
                }
            });
        } else
#endif
        {
            Vector<detail::PSTag<T> > tags;
            tags.reserve(foo.local_size()*AMREX_SPACEDIM*ncomp);

            for (MFIter mfi(foo); mfi.isValid(); ++mfi)
            {
                const Box& vbx = mfi.validbox();

                const auto & bdlv = bcondloc.bndryLocs(mfi);
                const auto & bdcv = bcondloc.bndryConds(mfi);

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    if (idim != hidden_direction) {
                        const Orientation olo(idim,Orientation::low);
                        const Orientation ohi(idim,Orientation::high);
                        for (int icomp = 0; icomp < ncomp; ++icomp) {
                            tags.emplace_back(detail::PSTag<T>{
                                                    undrrelxr[olo].array(mfi),
                                                    undrrelxr[ohi].array(mfi),
                                                    maskvals[olo].const_array(mfi),
                                                    maskvals[ohi].const_array(mfi),
                                                    static_cast<T>(bdlv[icomp][olo]),
                                                    static_cast<T>(bdlv[icomp][ohi]),
                                                    amrex::adjCell(vbx,olo),
                                                    bdcv[icomp][olo], bdcv[icomp][ohi],
                                                    vbx.length(idim), icomp, idim});
                        }
                    }
                }
            }

            ParallelFor(tags,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, detail::PSTag<T> const& tag) noexcept
            {
                if (tag.dir == 0)
                {
                    mllinop_comp_interp_coef0_x
                        (0, i           , j, k, tag.blen, tag.flo, tag.mlo,
                         tag.bctlo, tag.bcllo, imaxorder, dxi, tag.comp);
                    mllinop_comp_interp_coef0_x
                        (1, i+tag.blen+1, j, k, tag.blen, tag.fhi, tag.mhi,
                         tag.bcthi, tag.bclhi, imaxorder, dxi, tag.comp);
                }
#if (AMREX_SPACEDIM > 1)
                else
#if (AMREX_SPACEDIM > 2)
                if (tag.dir == 1)
#endif
                {
                    mllinop_comp_interp_coef0_y
                        (0, i, j           , k, tag.blen, tag.flo, tag.mlo,
                         tag.bctlo, tag.bcllo, imaxorder, dyi, tag.comp);
                    mllinop_comp_interp_coef0_y
                        (1, i, j+tag.blen+1, k, tag.blen, tag.fhi, tag.mhi,
                         tag.bcthi, tag.bclhi, imaxorder, dyi, tag.comp);
                }
#if (AMREX_SPACEDIM > 2)
                else {
                    mllinop_comp_interp_coef0_z
                        (0, i, j, k           , tag.blen, tag.flo, tag.mlo,
                         tag.bctlo, tag.bcllo, imaxorder, dzi, tag.comp);
                    mllinop_comp_interp_coef0_z
                        (1, i, j, k+tag.blen+1, tag.blen, tag.fhi, tag.mhi,
                         tag.bcthi, tag.bclhi, imaxorder, dzi, tag.comp);
                }
#endif
#endif
            });
        }
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(foo, MFItInfo{}.SetDynamic(true)); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();

            const auto & bdlv = bcondloc.bndryLocs(mfi);
            const auto & bdcv = bcondloc.bndryConds(mfi);

#ifdef AMREX_USE_EB
            auto fabtyp = (flags) ? (*flags)[mfi].getType(vbx) : FabType::regular;
#endif
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                if (idim == hidden_direction) continue;
                const Orientation olo(idim,Orientation::low);
                const Orientation ohi(idim,Orientation::high);
                const Box blo = amrex::adjCellLo(vbx, idim);
                const Box bhi = amrex::adjCellHi(vbx, idim);
                const int blen = vbx.length(idim);
                const auto& mlo = maskvals[olo].array(mfi);
                const auto& mhi = maskvals[ohi].array(mfi);
                const auto& flo = undrrelxr[olo].array(mfi);
                const auto& fhi = undrrelxr[ohi].array(mfi);
                for (int icomp = 0; icomp < ncomp; ++icomp) {
                    const BoundCond bctlo = bdcv[icomp][olo];
                    const BoundCond bcthi = bdcv[icomp][ohi];
                    const Real bcllo = bdlv[icomp][olo];
                    const Real bclhi = bdlv[icomp][ohi];
#ifdef AMREX_USE_EB
                    if (fabtyp == FabType::singlevalued) {
                        Array4<Real const> const& ap = area[idim]->const_array(mfi);
                        if (idim == 0) {
                            mllinop_comp_interp_coef0_x_eb
                                (0, blo, blen, flo, mlo, ap, bctlo, bcllo,
                                 imaxorder, dxi, icomp);
                            mllinop_comp_interp_coef0_x_eb
                                (1, bhi, blen, fhi, mhi, ap, bcthi, bclhi,
                                 imaxorder, dxi, icomp);
                        } else if (idim == 1) {
                            mllinop_comp_interp_coef0_y_eb
                                (0, blo, blen, flo, mlo, ap, bctlo, bcllo,
                                 imaxorder, dyi, icomp);
                            mllinop_comp_interp_coef0_y_eb
                                (1, bhi, blen, fhi, mhi, ap, bcthi, bclhi,
                                 imaxorder, dyi, icomp);
                        } else {
                            mllinop_comp_interp_coef0_z_eb
                                (0, blo, blen, flo, mlo, ap, bctlo, bcllo,
                                 imaxorder, dzi, icomp);
                            mllinop_comp_interp_coef0_z_eb
                                (1, bhi, blen, fhi, mhi, ap, bcthi, bclhi,
                                 imaxorder, dzi, icomp);
                        }
                    } else if (fabtyp == FabType::regular)
#endif
                    {
                        if (idim == 0) {
                            mllinop_comp_interp_coef0_x
                                (0, blo, blen, flo, mlo, bctlo, bcllo,
                                 imaxorder, dxi, icomp);
                            mllinop_comp_interp_coef0_x
                                (1, bhi, blen, fhi, mhi, bcthi, bclhi,
                                 imaxorder, dxi, icomp);
                        } else if (idim == 1) {
                            mllinop_comp_interp_coef0_y
                                (0, blo, blen, flo, mlo, bctlo, bcllo,
                                 imaxorder, dyi, icomp);
                            mllinop_comp_interp_coef0_y
                                (1, bhi, blen, fhi, mhi, bcthi, bclhi,
                                 imaxorder, dyi, icomp);
                        } else {
                            mllinop_comp_interp_coef0_z
                                (0, blo, blen, flo, mlo, bctlo, bcllo,
                                 imaxorder, dzi, icomp);
                            mllinop_comp_interp_coef0_z
                                (1, bhi, blen, fhi, mhi, bcthi, bclhi,
                                 imaxorder, dzi, icomp);
                        }
                    }
                }
            }
        }
    }
}

template <typename MF>
void
MLCellLinOp::applyBCT (int amrlev, int mglev, MF& in, BCMode bc_mode, const MLMGBndry* bndry,
                       bool skip_fillboundary) const
{
    BL_PROFILE("MLCellLinOp::applyBC()");
    // No coarsened boundary values, cannot apply inhomog at mglev>0.
    BL_ASSERT(mglev == 0 || bc_mode == BCMode::Homogeneous);
    BL_ASSERT(bndry != nullptr || bc_mode == BCMode::Homogeneous);

    const int ncomp = getNComp();
    const int cross = isCrossStencil();
    if (!skip_fillboundary) {
        in.FillBoundary(0, ncomp, m_geom[amrlev][mglev].periodicity(),cross);
    }

    int flagbc = bc_mode == BCMode::Inhomogeneous;
    const int imaxorder = maxorder;

    using T = typename MF::value_type;

    const Real* dxinv = m_geom[amrlev][mglev].InvCellSize();
    const T dxi = static_cast<T>(dxinv[0]);
    const T dyi = static_cast<T>((AMREX_SPACEDIM >= 2) ? dxinv[1] : Real(1.0));
    const T dzi = static_cast<T>((AMREX_SPACEDIM == 3) ? dxinv[2] : Real(1.0));

    const auto& maskvals = m_maskvals[amrlev][mglev];
    const auto& bcondloc = *m_bcondloc[amrlev][mglev];

    FArrayBox foofab(Box::TheUnitBox(),ncomp);
    const auto& foo = foofab.const_array();

    MFItInfo mfi_info;
    if (Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);

    const int hidden_direction = hiddenDirection();

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion())
    {
        Vector<detail::ABCTag<T> > tags;
        tags.reserve(in.local_size()*AMREX_SPACEDIM*ncomp);

        for (MFIter mfi(in); mfi.isValid(); ++mfi) {
            const Box& vbx = mfi.validbox();
            const auto& iofab = in.array(mfi);
            const auto & bdlv = bcondloc.bndryLocs(mfi);
            const auto & bdcv = bcondloc.bndryConds(mfi);

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (idim != hidden_direction) {
                    const Orientation olo(idim,Orientation::low);
                    const Orientation ohi(idim,Orientation::high);
                    const auto& bvlo = (bndry != nullptr) ?
                        bndry->bndryValues(olo).const_array(mfi) : foo;
                    const auto& bvhi = (bndry != nullptr) ?
                        bndry->bndryValues(ohi).const_array(mfi) : foo;
                    for (int icomp = 0; icomp < ncomp; ++icomp) {
                        tags.emplace_back(detail::ABCTag<T>{iofab, bvlo, bvhi,
                                                 maskvals[olo].const_array(mfi),
                                                 maskvals[ohi].const_array(mfi),
                                                 static_cast<T>(bdlv[icomp][olo]),
                                                 static_cast<T>(bdlv[icomp][ohi]),
                                                 amrex::adjCell(vbx,olo),
                                                 bdcv[icomp][olo], bdcv[icomp][ohi],
                                                 vbx.length(idim), icomp, idim});
                    }
                }
            }
        }

        ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, detail::ABCTag<T> const& tag) noexcept
        {
            if (tag.dir == 0)
            {
                mllinop_apply_bc_x(0, i, j, k, tag.blen, tag.fab,
                                   tag.mask_lo, tag.bctype_lo, tag.bcloc_lo, tag.bcval_lo,
                                   imaxorder, dxi, flagbc, tag.comp);
                mllinop_apply_bc_x(1, i+tag.blen+1, j, k, tag.blen, tag.fab,
                                   tag.mask_hi, tag.bctype_hi, tag.bcloc_hi, tag.bcval_hi,
                                   imaxorder, dxi, flagbc, tag.comp);
            }
#if (AMREX_SPACEDIM > 1)
            else
#if (AMREX_SPACEDIM > 2)
            if (tag.dir == 1)
#endif
            {
                mllinop_apply_bc_y(0, i, j, k, tag.blen, tag.fab,
                                   tag.mask_lo, tag.bctype_lo, tag.bcloc_lo, tag.bcval_lo,
                                   imaxorder, dyi, flagbc, tag.comp);
                mllinop_apply_bc_y(1, i, j+tag.blen+1, k, tag.blen, tag.fab,
                                   tag.mask_hi, tag.bctype_hi, tag.bcloc_hi, tag.bcval_hi,
                                   imaxorder, dyi, flagbc, tag.comp);
            }
#if (AMREX_SPACEDIM > 2)
            else {
                mllinop_apply_bc_z(0, i, j, k, tag.blen, tag.fab,
                                   tag.mask_lo, tag.bctype_lo, tag.bcloc_lo, tag.bcval_lo,
                                   imaxorder, dzi, flagbc, tag.comp);
                mllinop_apply_bc_z(1, i, j, k+tag.blen+1, tag.blen, tag.fab,
                                   tag.mask_hi, tag.bctype_hi, tag.bcloc_hi, tag.bcval_hi,
                                   imaxorder, dzi, flagbc, tag.comp);
            }
#endif
#endif
        });
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(in, mfi_info); mfi.isValid(); ++mfi)
        {
            const Box& vbx   = mfi.validbox();
            const auto& iofab = in.array(mfi);

            const auto & bdlv = bcondloc.bndryLocs(mfi);
            const auto & bdcv = bcondloc.bndryConds(mfi);

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                if (hidden_direction == idim) continue;
                const Orientation olo(idim,Orientation::low);
                const Orientation ohi(idim,Orientation::high);
                const Box blo = amrex::adjCellLo(vbx, idim);
                const Box bhi = amrex::adjCellHi(vbx, idim);
                const int blen = vbx.length(idim);
                const auto& mlo = maskvals[olo].array(mfi);
                const auto& mhi = maskvals[ohi].array(mfi);
                const auto& bvlo = (bndry != nullptr) ? bndry->bndryValues(olo).const_array(mfi) : foo;
                const auto& bvhi = (bndry != nullptr) ? bndry->bndryValues(ohi).const_array(mfi) : foo;
                for (int icomp = 0; icomp < ncomp; ++icomp) {
                    const BoundCond bctlo = bdcv[icomp][olo];
                    const BoundCond bcthi = bdcv[icomp][ohi];
                    const T bcllo = static_cast<T>(bdlv[icomp][olo]);
                    const T bclhi = static_cast<T>(bdlv[icomp][ohi]);
                    if (idim == 0) {
                        mllinop_apply_bc_x(0, blo, blen, iofab, mlo,
                                           bctlo, bcllo, bvlo,
                                           imaxorder, dxi, flagbc, icomp);
                        mllinop_apply_bc_x(1, bhi, blen, iofab, mhi,
                                           bcthi, bclhi, bvhi,
                                           imaxorder, dxi, flagbc, icomp);
                    } else if (idim == 1) {
                        mllinop_apply_bc_y(0, blo, blen, iofab, mlo,
                                           bctlo, bcllo, bvlo,
                                           imaxorder, dyi, flagbc, icomp);
                        mllinop_apply_bc_y(1, bhi, blen, iofab, mhi,
                                           bcthi, bclhi, bvhi,
                                           imaxorder, dyi, flagbc, icomp);
                    } else {
                        mllinop_apply_bc_z(0, blo, blen, iofab, mlo,
                                           bctlo, bcllo, bvlo,
                                           imaxorder, dzi, flagbc, icomp);
                        mllinop_apply_bc_z(1, bhi, blen, iofab, mhi,
                                           bcthi, bclhi, bvhi,
                                           imaxorder, dzi, flagbc, icomp);
                    }
                }
            }
        }
    }
}

template <typename FMF, typename CMF>
void MLCellLinOp::interpolationT (int amrlev, int fmglev, FMF& fine, CMF const& crse) const
{
    using FT = typename FMF::value_type;

    const int ncomp = getNComp();

    Dim3 ratio3 = {2,2,2};
    IntVect ratio = (amrlev > 0) ? IntVect(2) : mg_coarsen_ratio_vec[fmglev];
    AMREX_D_TERM(ratio3.x = ratio[0];,
                 ratio3.y = ratio[1];,
                 ratio3.z = ratio[2];);

#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion() && fine.isFusingCandidate()) {
        auto const& finema = fine.arrays();
        auto const& crsema = crse.const_arrays();
        ParallelFor(fine, IntVect(0), ncomp,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
        {
            int ic = amrex::coarsen(i,ratio3.x);
            int jc = amrex::coarsen(j,ratio3.y);
            int kc = amrex::coarsen(k,ratio3.z);
            finema[box_no](i,j,k,n) += static_cast<FT>(crsema[box_no](ic,jc,kc,n));
        });
        Gpu::streamSynchronize();
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(fine,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx    = mfi.tilebox();
            auto const& cfab = crse.const_array(mfi);
            auto const& ffab = fine.array(mfi);
            AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                int ic = amrex::coarsen(i,ratio3.x);
                int jc = amrex::coarsen(j,ratio3.y);
                int kc = amrex::coarsen(k,ratio3.z);
                ffab(i,j,k,n) += static_cast<FT>(cfab(ic,jc,kc,n));
            });
        }
    }
}

}

#endif
