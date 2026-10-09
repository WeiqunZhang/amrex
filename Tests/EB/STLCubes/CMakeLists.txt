# The legacy STL builder misclassifies some nodes on grid-aligned faces in
# single precision, so this test runs in double precision only.
if (NOT (3 IN_LIST AMReX_SPACEDIM) OR AMReX_PRECISION STREQUAL "SINGLE")
    return()
endif ()

set(_sources  main.cpp)

set(_input_files  inputs cubes.stl)

setup_test(3 _sources _input_files)

unset(_sources)
unset(_input_files)
