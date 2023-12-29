#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <random>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        int ncells = 64;
        Box domain(IntVect(0),IntVect(ncells-1));

        std::array<Real,AMREX_SPACEDIM> rblo{AMREX_D_DECL(Real(-3.703e18),Real(-3.703e18),Real(-3.703e18))};
        std::array<Real,AMREX_SPACEDIM> rbhi{AMREX_D_DECL(Real( 3.703e18),Real( 3.703e18),Real( 3.703e18))};

        RealBox rb(rblo, rbhi);
        Geometry geom(domain, rb, 0, {AMREX_D_DECL(0,0,0)});

        auto rlo = geom.ProbLoArrayInParticleReal();
        auto rhi = geom.ProbHiArrayInParticleReal();

        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            amrex::Print().SetPrecision(17)
                << "xxxxx rlo[" << i << "] = " << rlo[i]
                << ", rhi[" << i << "] = " << rhi[i] << std::endl;
        }
    }
    amrex::Finalize();
}
