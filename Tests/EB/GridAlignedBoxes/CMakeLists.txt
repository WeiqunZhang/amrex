if (NOT (3 IN_LIST AMReX_SPACEDIM))
    return()
endif ()

set(_sources  main.cpp)

set(_input_files  inputs cubes.stl)

setup_test(3 _sources _input_files)

# The same boxes as an implicit function
set(_exe_dir ${CMAKE_CURRENT_BINARY_DIR}/3d)
set(_name EB_GridAlignedBoxes_implicit_3d)
set(_cmd ${_exe_dir}/Test_EB_GridAlignedBoxes_3d inputs geometry=boxes max_volfrac_error=0.75)
if (AMReX_OMP)
    add_test(NAME ${_name} COMMAND ${_cmd} WORKING_DIRECTORY ${_exe_dir})
    set_tests_properties(${_name} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
elseif (AMReX_MPI)
    add_test(NAME ${_name} COMMAND mpiexec -n 2 ${_cmd} WORKING_DIRECTORY ${_exe_dir})
else ()
    add_test(NAME ${_name} COMMAND ${_cmd} WORKING_DIRECTORY ${_exe_dir})
endif ()

unset(_sources)
unset(_input_files)
unset(_exe_dir)
unset(_name)
unset(_cmd)
