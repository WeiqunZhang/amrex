#ifndef MY_TEST_K_H_
#define MY_TEST_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE
inline void init (amrex::Real x, amrex::Real y, amrex::Real R2,
                  amrex::Real& u, amrex::Real& v,
                  amrex::Real& urhs, amrex::Real& vrhs,
                  amrex::Real& eta)
{
    constexpr amrex::Real pi = 3.1415926535897932;
    amrex::Real x2 = x*x;
    amrex::Real y2 = y*y;
    amrex::Real r2 = x2+y2;
    amrex::Real r2R2 = r2/R2;
    amrex::Real R4 = R2*R2;
    amrex::Real sinx = amrex::Math::sin(x);
    amrex::Real cosx = amrex::Math::cos(x);
    amrex::Real siny = amrex::Math::sin(y);
    amrex::Real cosy = amrex::Math::cos(y);

    amrex::Real f1 = amrex::Math::sin(   pi*r2R2);
    amrex::Real f2 = amrex::Math::sin(2.*pi*r2R2);
    amrex::Real f3 = amrex::Math::sin(3.*pi*r2R2);

    u = sinx * (1. + siny) * f1;
    v = (1. + sinx) * siny * f2;
    eta = 2. + sinx * siny;

    amrex::Real df1dx = amrex::Math::cos(   pi*r2R2) * (2.*pi)/R2 * x;
    amrex::Real df1dy = amrex::Math::cos(   pi*r2R2) * (2.*pi)/R2 * y;
    amrex::Real df2dx = amrex::Math::cos(2.*pi*r2R2) * (4.*pi)/R2 * x;
    amrex::Real df2dy = amrex::Math::cos(2.*pi*r2R2) * (4.*pi)/R2 * y;
    amrex::Real df3dx = amrex::Math::cos(3.*pi*r2R2) * (6.*pi)/R2 * x;
    amrex::Real df3dy = amrex::Math::cos(3.*pi*r2R2) * (6.*pi)/R2 * y;

    amrex::Real df1dx2 =  -amrex::Math::sin(pi*r2R2) * (4.*pi*pi)/R4 * x2
                          +amrex::Math::cos(pi*r2R2) * (2.*pi)/R2;
    amrex::Real df1dy2 =  -amrex::Math::sin(pi*r2R2) * (4.*pi*pi)/R4 * y2
                          +amrex::Math::cos(pi*r2R2) * (2.*pi)/R2;
    amrex::Real df1dxdy = -amrex::Math::sin(pi*r2R2) * (4.*pi*pi)/R4 * x * y;

    amrex::Real df2dx2 =  -amrex::Math::sin(2.*pi*r2R2) * (16.*pi*pi)/R4 * x2
                          +amrex::Math::cos(2.*pi*r2R2) * (4.*pi)/R2;
    amrex::Real df2dy2 =  -amrex::Math::sin(2.*pi*r2R2) * (16.*pi*pi)/R4 * y2
                          +amrex::Math::cos(2.*pi*r2R2) * (4.*pi)/R2;
    amrex::Real df2dxdy = -amrex::Math::sin(2.*pi*r2R2) * (16.*pi*pi)/R4 * x * y;

    amrex::Real df3dx2 = -amrex::Math::sin(3.*pi*r2R2) * (36.*pi*pi)/R4 * x2
                         +amrex::Math::cos(3.*pi*r2R2) * (6.*pi)/R2;
    amrex::Real df3dy2 = -amrex::Math::sin(3.*pi*r2R2) * (36.*pi*pi)/R4 * y2
                         +amrex::Math::cos(3.*pi*r2R2) * (6.*pi)/R2;

    amrex::Real detadx = cosx * siny;
    amrex::Real detady = sinx * cosy;

    amrex::Real dudx = cosx * (1.+siny) * f1
                     + sinx * (1.+siny) * df1dx;
    amrex::Real dudy = sinx * cosy * f1
                     + sinx * (1.+siny) * df1dy;

    amrex::Real dvdx = cosx * siny * f2
                     + (1.+sinx) * siny * df2dx;
    amrex::Real dvdy = (1.+sinx) * cosy * f2
                     + (1.+sinx) * siny * df2dy;

    amrex::Real dudx2 = -sinx * (1.+siny) * f1
                       + cosx * (1.+siny) * df1dx
                       + cosx * (1.+siny) * df1dx
                       + sinx * (1.+siny) * df1dx2;
    amrex::Real dudy2 = -sinx * siny * f1
                       + sinx * cosy * df1dy
                       + sinx * cosy * df1dy
                       + sinx * (1.+siny) * df1dy2;
    amrex::Real dudxdy = cosx * cosy * f1
                       + cosx * (1.+siny) * df1dy
                       + sinx * cosy * df1dx
                       + sinx * (1.+siny) * df1dxdy;

    amrex::Real dvdx2 = -sinx * siny * f2
                        +cosx * siny * df2dx
                        +cosx * siny * df2dx
                        +(1.+sinx) * siny * df2dx2;
    amrex::Real dvdy2 = -(1.+sinx) * siny * f2
                        +(1.+sinx) * cosy * df2dy
                        +(1.+sinx) * cosy * df2dy
                        +(1.+sinx) * siny * df2dy2;
    amrex::Real dvdxdy = cosx * cosy * f2
                        +cosx * siny * df2dy
                        +(1.+sinx) * cosy * df2dx
                        +(1.+sinx) * siny * df2dxdy;

    amrex::Real dtauxxdx = detadx*((4./3.)*dudx - (2./3.)*dvdy)
                            + eta*((4./3.)*dudx2 - (2./3.)*dvdxdy);
    amrex::Real dtauyydy = detady*((4./3.)*dvdy - (2./3.)*dudx)
                            + eta*((4./3.)*dvdy2 - (2./3.)*dudxdy);

    amrex::Real dtauxydx = detadx*(dvdx + dudy) + eta*(dvdx2 + dudxdy);
    amrex::Real dtauxydy = detady*(dvdx + dudy) + eta*(dvdxdy + dudy2);

    urhs = -(dtauxxdx + dtauxydy);
    vrhs = -(dtauxydx + dtauyydy);
}

#endif
