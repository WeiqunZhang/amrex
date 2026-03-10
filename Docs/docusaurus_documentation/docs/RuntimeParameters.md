# Runtime Parameters

This chapter contains a list of AMReX `ParmParse` runtime parameters and their **default** values. They can be set by either including them in an inputs file, or specifying them at the command line, or passing a function to `amrex::Initialize` and the function adds parameters to AMReX's `ParmParse`'s parameter database. For more information on `ParmParse`, see `sec:basics:parmparse`.

> [!IMPORTANT]
> AMReX reserves the following prefixes in `ParmParse` parameters: `amr`, `amrex`, `blprofiler`, `device`, `DistributionMapping`, `eb2`, `fab`, `fabarray`, `geometry`, `integration`, `particles`, `tiny_profiler`, and `vismf`.

## AMR

AMReX applications with AMR use either `class AmrCore` or the more specialized `class Amr`. Since `class Amr` is derived from `class AmrCore`, the parameters for the `AmrCore` class also apply to the `Amr` class. Additionally, `class AmrCore` is derived from `class AmrMesh`, so `AmrMesh` member functions are also available to `AmrCore` and `Amr`.

### AmrCore Class

Below are a list of important `ParmParse` parameters. However, AMReX applications can choose to avoid them entirely by use this `AMRCore` constructor `AmrCore(Geometry const& level_0_geom, AmrInfo const&
amr_info)`, where `struct AmrInfo` contains all the information that can be set via `ParmParse`.

> This parameter controls the error buffer size in the z-direction. If the size of the integer array is less than the number of levels, the last integer will be used for the unspecified levels.

### Amr Class

> [!WARNING]
> These parameters are specific to `class Amr` based applications. If your application uses `class AmrCore` directly, they do not apply unless you have provided implementations for them.

#### Subcycling

#### Regrid

#### I/O

## Basic Controls

## Communication

## Distribution Mapping

## Embedded Boundary

## Error Handling

By default AMReX installs a signal handler that will be run when a signal such as segfault is received. You can also enable floating point exception trapping. The signal handler will print out backtraces that can be useful for debugging.

> [!NOTE]
> Floating point exception trapping is not enabled by default, because compilers might generate optimized SIMD code that raises the exceptions.

## Extern

### HYPRE

These parameters are relevant only when HYPRE support is enabled.

## Geometry

All these parameters are optional for constructing a `Geometry `sec:basics:geom`` object. They are only used if the information is not provided via function arguments.

## I/O

## Memory

## Particles

## Tiling

## Time Integration

All these parameters are optional for constructing or configuring a `TimeIntegrator `sec:basics:timeintegration`` object.

### Runge--Kutta Methods

These parameters are relevant only when `integration.type` is "RungeKutta".

#### User-specificed Runge--Kutta Method

When `integration.rk.type` is "User", the following parameters can be used to set a user-specificed explicit Butcher tableau,

$$\begin{aligned}
B \; \equiv \;
\begin{array}{r|c}
  c & A \\
  \hline
    & b \\
    & \tilde{b}
\end{array},
\end{aligned}$$

where, for a method with $s$ stages, $c$, and $b$, $\tilde{b}$ are arrays of $s$ values and $A$ is a lower triangular $s \times s$ matrix.

### SUNDIALS

These parameters are relevant only when support for SUNDIALS time integrators is enabled (see `sec:time_int:sundials`) and `integration.type` is "SUNDIALS".

#### Methods

#### Step Sizes

> [!NOTE]
> The parameter `integration.time_step` is used to set a fixed step size with single rate methods (e.g., ERK) or a fixed slow time scale step size with multirate methods (e.g., EX-MRI).

#### Tolerances

When using adaptive time step sizes or an implicit method (e.g., DIRK), selecting appropriate tolerances for the target application is a critical factor in method performance. For advice on selecting tolerances see the [SUNDIALS documentation](https://sundials.readthedocs.io/en/latest/arkode/Usage/User_callable.html#general-advice-on-the-choice-of-tolerances).

#### Algebraic Solvers

The parameters provide control over the nonlinear and linear solvers utilized with implicit methods (e.g., DIRK).

## Tiny Profiler

These parameters are ignored unless profiling with `TinyProfiler` is enabled.
