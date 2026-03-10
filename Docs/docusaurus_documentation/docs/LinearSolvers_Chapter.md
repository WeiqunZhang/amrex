# Linear Solvers

AMReX supports both single-level solves and composite solves on multiple AMR levels, with the solution to the linear system defined on cell centers, edges or nodes. AMReX also supports solution of linear systems with embedded boundaries. (See chapter `Chap:EB` for more details on the embedded boundary representation of complex geometry.) In addition to the iterative solvers discussed in this chapter, AMReX also supports solving Poisson equations using Fast Fourier Transform (FFT) (see chapter `Chap:FFT` for more information).

The default solution technique is geometric multigrid. AMReX also provides BiCGStab solvers, GMRES, and interfaces to the HYPRE and PETSc libraries.

In this Chapter we give an overview of the linear solvers in AMReX that solve linear systems in the canonical form

<span label="eqn::abeclap">$$(A \alpha - B \nabla \cdot \beta \nabla ) \phi = f,$$</span>

where $A$ and $B$ are scalar constants, $\alpha$ and $\beta$ are scalar fields, $\phi$ is the unknown, and $f$ is the right-hand side of the equation. Note that Poisson's equation $\nabla^2 \phi = f$ is a special case of the canonical form. The solution $\phi$ is at either cell centers or nodes.

For the cell-centered solver, $\alpha$, $\phi$ and $f$ are represented by cell-centered MultiFabs, and $\beta$ is represented by `AMREX_SPACEDIM` face type MultiFabs, i.e. there are separate MultiFabs for the $\beta$ coefficient in each coordinate direction.

For the nodal solver, $A$ and $\alpha$ are assumed to be zero, $\phi$ and $f$ are nodal, and $\beta$ (which we later refer to as $\sigma$) is cell-centered.

In addition to these solvers, AMReX has support for tensor solves used to calculate the viscous terms that appear in the compressible Navier-Stokes equations. In these solves, all components of the velocity field are solved for simultaneously. The tensor solve functionality is only available for cell-centered velocity. AMReX also supports the curl-curl operator, with solutions defined on cell edges.

The tutorials in [Linear Solvers](https://amrex-codes.github.io/amrex/tutorials_html/LinearSolvers_Tutorial.html) show examples of using the solvers. The tutorial `amrex-tutorials/ExampleCodes/Basic/HeatEquation_EX3_C` shows how to solve the heat equation implicitly using the solver. The tutorials also show how to add linear solvers into the build system.

