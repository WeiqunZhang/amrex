# Amr Class

The `Amr` class is designed to manage parts of the computation which do not belong on a single level, like establishing and updating the hierarchy of levels, global timestepping, and managing the different `AmrLevel`s. Most likely you will not need to derive any classes from `Amr`. Our mature application codes use this base class without any derived classes.

One of the most important data members is an array of `AmrLevel`s - the `Amr` class calls many functions from the `AmrLevel` class to do things like advance the solution on a level, compute a time step to be used for a level, etc.

# AmrLevel Class

Pure virtual functions include:

> - `computeInitialDt` Compute an array of time steps for each level of refinement. Called at the beginning of the simulation.
> - `computeNewDt` Compute an array of time steps for each level of refinement. Called at the end of a coarse level advance.
> - `advance` Advance the grids at a level.
> - `post_timestep` Work after at time step at a given level. In this tutorial we do the AMR synchronization here.
> - `post_regrid` Work after regridding. In this tutorial we redistribute particles.
> - `post_init` Work after initialization. In this tutorial we perform AMR synchronization.
> - `initData` Initialize the data on a given level at the beginning of the simulation.
> - `init` There are two versions of this function used to initialize data on a level during regridding. One version is specifically for the case where the level did not previously exist (a newly created refined level).
> - `errorEst` Perform the tagging at a level for refinement.

## StateData

The most important data managed by the `AmrLevel` is an array of `StateData`, which holds the scalar fields, etc., in the boxes that together make up the level.

`StateData` is a class that essentially holds a pair of MultiFabs: one at the old time and one at the new time. AMReX knows how to interpolate in time between these states to get data at any intermediate point in time. The main data that we care about in our applications codes (such as the fluid state) will be stored as `StateData`. Essentially, data is made `StateData` if we need it to be stored in checkpoints/plotfiles, and/or we want it to be automatically interpolated when we refine. An `AmrLevel` stores an array of `StateData` (in a C++ array called `state`). We index this array using integer keys (defined via an `enum` in, e.g., `AmrLevelAdv.H`):

``` c++
enum StateType { Phi_Type = 0,
                 NUM_STATE_TYPE };
```

In our tutorial code, we use the function `AmrLevelAdv::variableSetUp` to tell our simulation about the `StateData` (e.g., how many variables, ghost cells, nodality, etc.). Note that if you have more than one `StateType`, each of the different `StateData` carried in the state array can have different numbers of components, ghost cells, boundary conditions, etc. This is the main reason we separate all this data into separate `StateData` objects collected together in an indexable array.

# LevelBld Class

The `LevelBld` class is a pure virtual class for defining variable types and attributes. To more easily understand its usage, refer to the derived class, `LevelBldAdv` in the tutorial. The `variableSetUp` and `variableCleanUp` are implemented, and in this tutorial call routines in the `AmrLevelAdv` class, e.g.,

``` c++
void
AmrLevelAdv::variableSetUp ()
{
    BL_ASSERT(desc_lst.size() == 0);

    // Get options, set phys_bc
    read_params();

    desc_lst.addDescriptor(Phi_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,0,NUM_STATE,
                           &cell_cons_interp);

    int lo_bc[BL_SPACEDIM];
    int hi_bc[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; ++i) {
        lo_bc[i] = hi_bc[i] = amrex::BCType::int_dir;   // periodic boundaries
    }

    BCRec bc(lo_bc, hi_bc);

    StateDescriptor::BndryFunc bndryfunc(nullfill);
    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.

    desc_lst.setComponent(Phi_Type, 0, "phi", bc,
                          bndryfunc);
}
```

We see how to define the `StateType`, including nodality, whether or not we want the variable to represent a point in time or an interval over time (useful for returning the time associated with data), the number of ghost cells, number of components, and the interlevel interpolation (See AMReX_Interpolator for various interpolation types). We also see how to specify physical boundary functions by providing a function (in this case, `nullfill` since we are not using physical boundary conditions), where `nullfill` is defined in `Src/bc_nullfill.cpp` in the tutorial source code.

# Example: Advection_AmrLevel

The Advection_AmrLevel example is documented in detail [here](https://amrex-codes.github.io/amrex/tutorials_html/AMR_Tutorial.html#advection-amrlevel) in the AMReX tutorial documentation.

The `Src` subdirectory contains source code that is specific to this example. Most notably is the `AmrLevelAdv` class, which is derived from the base `AmrLevel` class, and the `LevelBldAdv` class, derived from the base `LevelBld` class as described above. The subdirectory `Src/Src_K` contain GPU kernels.

The `Exec` subdirectory contains two examples: `SingleVortex` and `UniformVelocity`. Each subdirectory contains problem-specific source code used for initialization using a Fortran subroutine (`Prob.f90`) and specifying the velocity fields used in this simulation (`face_velocity_2d_K.H` and `face_velocity_3d_K.H` for the 2-D and 3-D problem, respectively). Build the code here by editing the `GNUmakefile` and running `make`.

The pseudocode for the main program is given below.

``` c++
/* Advection_AmrLevel Pseudocode */
main()
  Amr amr;
  amr.init()
  loop {
    amr.coarseTimeStep()
      /* compute dt */
      timeStep()
        amr_level[level]->advance()
        /* call timeStep r times for next-finer level */
        amr_level[level]->post_timestep() // AMR synchronization
      postCoarseTimeStep()
      /* write plotfile and checkpoint */
  }
  /* write final plotfile and checkpoint */
```

# Particles

There is an option to turn on passively advected particles. In the `GNUmakefile`, add the line `USE_PARTICLES = TRUE` and build the code (do a `make realclean first`). In the inputs file, add the line `adv.do_tracers = 1`. When you run the code, within each plotfile directory there will be a subdirectory called "Tracer". These can be visualized using either yt or Paraview (refer to the chapter on `Chap:Visualization`).
