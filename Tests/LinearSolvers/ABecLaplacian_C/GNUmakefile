DEBUG = FALSE

USE_MPI  = TRUE
USE_OMP  = FALSE

USE_CUDA = TRUE

USE_HYPRE = TRUE
USE_PETSC = FALSE

COMP = gnu

DIM = 2

BL_NO_FORT = TRUE

AMREX_HOME = ../../..

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs 	:= Base Boundary LinearSolvers

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

