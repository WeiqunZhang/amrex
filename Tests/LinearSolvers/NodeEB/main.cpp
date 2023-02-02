
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "MyTest.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        amrex::SetVerbose(0);
        {
            amrex::ParmParse pp("eb2");
            pp.add("geom_type", std::string("cylinder"));
            pp.add("cylinder_direction", 0);
            pp.addarr("cylinder_center", std::vector<amrex::Real>{{0.0,0.7006,0.5521}});
            pp.add("cylinder_radius", 0.125);
            pp.add("cylinder_height", -1.0);
            pp.add("cylinder_has_fluid_inside", 0);
        }
        {
            amrex::ParmParse pp;
            pp.add("max_iter", 15);
        }

        MyTest mytest;
        mytest.solve();
    }

    amrex::Finalize();
}
