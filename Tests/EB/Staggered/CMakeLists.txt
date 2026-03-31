if ((NOT 2 IN_LIST AMReX_SPACEDIM) AND (NOT 3 IN_LIST AMReX_SPACEDIM))
    return()
endif ()

set(_sources main.cpp)
set(_input_files inputs)

if (2 IN_LIST AMReX_SPACEDIM)
    setup_test(2 _sources _input_files BASE_NAME EB_Staggered)
endif ()

if (3 IN_LIST AMReX_SPACEDIM)
    setup_test(3 _sources _input_files BASE_NAME EB_Staggered)
endif ()

unset(_sources)
unset(_input_files)
