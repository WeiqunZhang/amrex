#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBStaggeredData.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <string>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real volume_fraction (Real x_lo, Real x_hi, Real plane_loc) noexcept
{
    if (plane_loc <= x_lo) { return Real(1.0); }
    if (plane_loc >= x_hi) { return Real(0.0); }
    return (x_hi - plane_loc) / (x_hi - x_lo);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real boundary_fraction (Real x_lo, Real x_hi, Real plane_loc) noexcept
{
    return (plane_loc > x_lo && plane_loc < x_hi) ? Real(1.0) : Real(0.0);
}

void configure_predefined_geometry (std::string geom_type,
                                    Real plane_loc,
                                    Vector<Real> plane_normal,
                                    Vector<Real> sphere_center,
                                    Real sphere_radius,
                                    Vector<Real> box_lo,
                                    Vector<Real> box_hi)
{
    ParmParse ppeb2("eb2");

    if (geom_type.empty()) {
        geom_type = "plane";
    }

    if (geom_type == "plane") {
        ppeb2.add("geom_type", geom_type);
        Vector<Real> plane_point(AMREX_SPACEDIM, Real(0.0));
        plane_point[0] = plane_loc;
        ppeb2.addarr("plane_point", plane_point);
        if (plane_normal.size() != AMREX_SPACEDIM) {
            plane_normal.assign(AMREX_SPACEDIM, Real(0.0));
            plane_normal[0] = Real(-1.0);
        }
        ppeb2.addarr("plane_normal", plane_normal);
    } else if (geom_type == "sphere") {
        if (sphere_center.size() != AMREX_SPACEDIM) {
            sphere_center.assign(AMREX_SPACEDIM, Real(0.5));
        }
        if (sphere_radius <= Real(0.0)) {
            sphere_radius = Real(0.25);
        }
        ppeb2.add("geom_type", geom_type);
        ppeb2.addarr("sphere_center", sphere_center);
        ppeb2.add("sphere_radius", sphere_radius);
    } else if (geom_type == "box") {
        if (box_lo.size() != AMREX_SPACEDIM) {
            box_lo.assign(AMREX_SPACEDIM, Real(0.25));
        }
        if (box_hi.size() != AMREX_SPACEDIM) {
            box_hi.assign(AMREX_SPACEDIM, Real(0.75));
        }
        ppeb2.add("geom_type", geom_type);
        ppeb2.addarr("box_lo", box_lo);
        ppeb2.addarr("box_hi", box_hi);
    } else {
        amrex::Abort("Unsupported geom_type '" + geom_type + "'. "
                     "Supported values: plane, sphere, box.");
    }
}

template <typename T>
void query_with_fallback (ParmParse& primary, ParmParse& fallback,
                          const std::string& key, T& value)
{
    if (primary.query(key.c_str(), value)) { return; }
    fallback.query(key.c_str(), value);
}

void queryarr_with_fallback (ParmParse& primary, ParmParse& fallback,
                             const std::string& key, Vector<Real>& value)
{
    Vector<Real> tmp;
    if (primary.queryarr(key.c_str(), tmp)) {
        value = tmp;
        return;
    }
    if (fallback.queryarr(key.c_str(), tmp)) {
        value = tmp;
    }
}

}

void run_staggered_test ()
{
    int ncell_x = 4;
#if (AMREX_SPACEDIM >= 2)
    int ncell_y = 1;
#endif
#if (AMREX_SPACEDIM >= 3)
    int ncell_z = 1;
#endif
    Real plane_loc = Real(0.37);
    Real tolerance = Real(1.0e-12);
    std::string geom_type = "plane";
    Vector<Real> plane_normal;
    Vector<Real> sphere_center;
    Real sphere_radius = Real(0.25);
    Vector<Real> box_lo;
    Vector<Real> box_hi;

    ParmParse pp_root;
    ParmParse ppstag("staggered");

    query_with_fallback(ppstag, pp_root, "n_cell_x", ncell_x);
#if (AMREX_SPACEDIM >= 2)
    query_with_fallback(ppstag, pp_root, "n_cell_y", ncell_y);
#endif
#if (AMREX_SPACEDIM >= 3)
    query_with_fallback(ppstag, pp_root, "n_cell_z", ncell_z);
#endif
    query_with_fallback(ppstag, pp_root, "plane_location", plane_loc);
    query_with_fallback(ppstag, pp_root, "tolerance", tolerance);
    query_with_fallback(ppstag, pp_root, "geometry_type", geom_type);
    query_with_fallback(ppstag, pp_root, "sphere_radius", sphere_radius);
    queryarr_with_fallback(ppstag, pp_root, "plane_normal", plane_normal);
    queryarr_with_fallback(ppstag, pp_root, "sphere_center", sphere_center);
    queryarr_with_fallback(ppstag, pp_root, "box_lo", box_lo);
    queryarr_with_fallback(ppstag, pp_root, "box_hi", box_hi);

    configure_predefined_geometry(geom_type, plane_loc,
                                  plane_normal, sphere_center,
                                  sphere_radius, box_lo, box_hi);

#if (AMREX_SPACEDIM == 1)
    const Box domain(IntVect(0), IntVect(ncell_x-1));
#elif (AMREX_SPACEDIM == 2)
    const Box domain(IntVect(0,0), IntVect(ncell_x-1, ncell_y-1));
#else
    const Box domain(IntVect(0,0,0), IntVect(ncell_x-1, ncell_y-1, ncell_z-1));
#endif

    const RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)},
                           {AMREX_D_DECL(1.0, 1.0, 1.0)});
    int is_periodic[] = {AMREX_D_DECL(0,0,0)};
    Geometry geom(domain, &real_box, 0, is_periodic);

    BoxArray ba(geom.Domain());
    DistributionMapping dm(ba);

    EB2::Build(geom, 0, 0, 2);

    Vector<int> ngrow(3, 2);
    auto factory = makeEBFabFactory(&EB2::IndexSpace::top(), geom, ba, dm,
                                    ngrow, EBSupport::full);

    const EBStaggeredData* stag = factory->getStaggeredData(0);
    AMREX_ALWAYS_ASSERT(stag != nullptr);

    const MultiFab& stag_vol = stag->getVolFrac();
    const MultiFab& stag_barea = stag->getBndryArea();

    const Box face_domain = amrex::convert(geom.Domain(), IntVect::TheDimensionVector(0));
    const int face_lo = face_domain.smallEnd(0);
    const int face_hi = face_domain.bigEnd(0);

    const auto dx = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();

    if (geom_type == "plane") {
        ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
        ReduceData<Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(stag_vol); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& varr = stag_vol.const_array(mfi);
            auto const& barr = stag_barea.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (i <= face_lo || i >= face_hi) {
                    return {Real(0.0), Real(0.0)};
                }

                Real const x_face = prob_lo[0]
                                  + static_cast<Real>(i - face_lo) * dx[0];
                Real const x_lo = amrex::max(x_face - Real(0.5)*dx[0], prob_lo[0]);
                Real const x_hi = amrex::min(x_face + Real(0.5)*dx[0], prob_hi[0]);
                Real const span = x_hi - x_lo;
                if (span <= Real(0.0)) {
                    return {Real(0.0), Real(0.0)};
                }

                Real const vol_exact = volume_fraction(x_lo, x_hi, plane_loc);
                Real const bnd_exact = boundary_fraction(x_lo, x_hi, plane_loc);

                Real const vol_err = amrex::Math::abs(varr(i,j,k) - vol_exact);
                Real const bnd_err = amrex::Math::abs(barr(i,j,k) - bnd_exact);
                return {vol_err, bnd_err};
            });
        }

        const ReduceTuple hv = reduce_data.value(reduce_op);
        Real max_vol_err = amrex::get<0>(hv);
        Real max_bnd_err = amrex::get<1>(hv);

        amrex::Print() << "Max staggered vol error      : " << max_vol_err << "\n";
        amrex::Print() << "Max staggered boundary error : " << max_bnd_err << "\n";

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_vol_err <= tolerance,
            "Staggered volume fraction mismatch exceeds tolerance");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_bnd_err <= tolerance,
            "Staggered boundary area mismatch exceeds tolerance");
    } else {
        ReduceOps<ReduceOpMin, ReduceOpMax, ReduceOpMin, ReduceOpMax> reduce_op;
        ReduceData<Real, Real, Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (MFIter mfi(stag_vol); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& varr = stag_vol.const_array(mfi);
            auto const& barr = stag_barea.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                return {varr(i,j,k), varr(i,j,k), barr(i,j,k), barr(i,j,k)};
            });
        }

        const ReduceTuple hv = reduce_data.value(reduce_op);
        Real min_vol = amrex::get<0>(hv);
        Real max_vol = amrex::get<1>(hv);
        Real min_bnd = amrex::get<2>(hv);
        Real max_bnd = amrex::get<3>(hv);

        amrex::Print() << "Selected geom_type = " << geom_type
                       << ". Plane analytic check disabled.\n";
        amrex::Print() << "Volume fraction bounds      : [" << min_vol
                       << ", " << max_vol << "]\n";
        amrex::Print() << "Boundary area bounds        : [" << min_bnd
                       << ", " << max_bnd << "]\n";

        constexpr Real unit_tol = Real(1.0e-12);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(min_vol >= -unit_tol,
            "Staggered volume fractions below zero");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_vol <= Real(1.0) + unit_tol,
            "Staggered volume fractions exceed unity");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(min_bnd >= -unit_tol,
            "Staggered boundary areas below zero");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_bnd <= Real(1.0) + unit_tol,
            "Staggered boundary areas exceed unity");
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        run_staggered_test();
    }
    amrex::Finalize();
}
