# Dual Grid Approach

In AMReX-based applications that have both mesh data and particle data, the mesh work and particle work have very different requirements for load balancing.

Rather than using a combined work estimate to create the same grids for mesh and particle data, we have the option to pursue a "dual grid" approach.

With this approach the mesh (`MultiFab`) and particle (`ParticleContainer`) data are allocated on different `BoxArrays` with different `DistributionMappings`.

This enables separate load balancing strategies to be used for the mesh and particle work.

The cost of this strategy, of course, is the need to copy mesh data onto temporary `MultiFabs` defined on the particle `BoxArrays` when mesh-particle communication is required.
