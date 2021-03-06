#ifndef AMREX_MLMG_K_H_
#define AMREX_MLMG_K_H_
#include <AMReX_Config.H>

#include <AMReX_FArrayBox.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBCellFlag.H>
#endif

#if (AMREX_SPACEDIM == 1)
#include <AMReX_MLMG_1D_K.H>
#elif (AMREX_SPACEDIM == 2)
#include <AMReX_MLMG_2D_K.H>
#else
#include <AMReX_MLMG_2D_K.H>
#include <AMReX_MLMG_3D_K.H>
#endif

#endif
