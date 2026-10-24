# Always the enclosing checkout: the test needs the FillBoundary variants of this branch.
AMREX_HOME := $(abspath ../../..)

DEBUG     = FALSE
DIM       = 3
COMP      = gnu

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = TRUE
CUDA_ARCH = 80

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
