# Building with GNU Make

In this build approach, you write your own make files defining a number of variables and rules. Then you invoke `make` to start the building process. This will result in an executable upon successful completion. The temporary files generated in the building process are stored in a temporary directory named `tmp_build_dir`.

## Dissecting a Simple Make File

An example of building with GNU Make can be found in `amrex-tutorials/ExampleCodes/Basic/HelloWorld_C`. `tab:makevars` below shows a list of important variables.

<div id="tab:makevars">

<table>
<caption>Important make variables</caption>
<colgroup>
<col />
<col />
<col />
</colgroup>
<thead>
<tr class="header">
<th>Variable</th>
<th>Value</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>AMREX_HOME</td>
<td>Path to amrex</td>
<td>environment</td>
</tr>
<tr class="even">
<td>COMP</td>
<td>gnu, cray, ibm, intel, intel-llvm, intel-classic, llvm, or pgi</td>
<td>none</td>
</tr>
<tr class="odd">
<td>CXXSTD</td>
<td>C++ standard (<code>c++17</code>, <code>c++20</code>)</td>
<td>compiler default, at least <code>c++17</code></td>
</tr>
<tr class="even">
<td>DEBUG</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="odd">
<td>DIM</td>
<td>1 or 2 or 3</td>
<td>3</td>
</tr>
<tr class="even">
<td>PRECISION</td>
<td>DOUBLE or FLOAT</td>
<td>DOUBLE</td>
</tr>
<tr class="odd">
<td>TEST</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="even">
<td>USE_ASSERTION</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="odd">
<td>USE_MPI</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="even">
<td>USE_OMP</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="odd">
<td>USE_CUDA</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="even">
<td>USE_HIP</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="odd">
<td>USE_SYCL</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="even">
<td>USE_RPATH</td>
<td>TRUE or FALSE</td>
<td>FALSE</td>
</tr>
<tr class="odd">
<td>WARN_ALL</td>
<td>TRUE or FALSE</td>
<td>TRUE for DEBUG FALSE otherwise</td>
</tr>
<tr class="even">
<td><dl>
<dt>AMREX_CUDA_ARCH</dt>
<dd>
<p>or CUDA_ARCH</p>
</dd>
</dl></td>
<td>CUDA arch such as 70</td>
<td>70 if not set or detected</td>
</tr>
<tr class="odd">
<td><dl>
<dt>AMREX_AMD_ARCH</dt>
<dd>
<p>or AMD_ARCH</p>
</dd>
</dl></td>
<td>AMD GPU arch such as gfx908</td>
<td>none if the machine is unknown</td>
</tr>
<tr class="even">
<td>USE_GPU_RDC</td>
<td>TRUE or FALSE</td>
<td>TRUE</td>
</tr>
</tbody>
</table>

Important make variables

</div>

At the beginning of `amrex-tutorials/ExampleCodes/Basic/HelloWorld_C/GNUmakefile`, `AMREX_HOME` is set to the path to the top directory of AMReX. Note that in the example `?=` is a conditional variable assignment operator that only has an effect if `AMREX_HOME` has not been defined (including in the environment). One can also set `AMREX_HOME` as an environment variable. For example in bash, one can set

``` bash
export AMREX_HOME=/path/to/amrex
```

alternatively, in tcsh one can set

``` bash
setenv AMREX_HOME /path/to/amrex
```

Note: when setting `AMREX_HOME` in the `GNUmakefile`, be aware that `~` does not expand, so `AMREX_HOME=~/amrex/` will yield an error.

One must set the `COMP` variable to choose a compiler. Currently the list of supported compilers includes gnu, cray, ibm, intel, llvm, and pgi.

One could set the `DIM` variable to either 1, 2, or 3, depending on the dimensionality of the problem. The default dimensionality is 3. AMReX uses double precision by default. One can change to single precision by setting `PRECISION=FLOAT`. (Particles have an equivalent flag `USE_SINGLE_PRECISION_PARTICLES=TRUE/FALSE`.)

Variables `DEBUG`, `TEST`, `USE_MPI` and `USE_OMP` are optional with default set to FALSE. The meaning of these variables should be obvious. When `DEBUG=TRUE`, aggressive compiler optimization flags are turned off and assertions in source code are turned on. For production runs, `DEBUG` should be set to FALSE. `TEST` and `USE_ASSERTION` are set by default in CI and add slight debugging, e.g., initializing default values in FABs. An advanced variable, `MPI_THREAD_MULTIPLE`, can be set to TRUE to initialize MPI with support for concurrent MPI calls from multiple threads.

Variables `USE_CUDA`, `USE_HIP` and `USE_SYCL` are used for targeting Nvidia, AMD and Intel GPUs, respectively. At most one of the three can be TRUE. For HIP and SYCL builds, we do only test against C++17 builds at the moment.

The variable `USE_RPATH` controls the link mechanism to dependent libraries. If enabled, the library path at link time will be saved as a [rpath hint](https://en.wikipedia.org/wiki/Rpath) in created binaries. When disabled, dynamic library paths could be provided via `export LD_LIBRARY_PATH` hints at runtime.

For GCC and Clang, the variable `WARN_ALL` controls the compiler's warning options. There is also a make variable `WARN_ERROR` (with default of `FALSE`) to turn warnings into errors.

When `USE_CUDA` is `TRUE`, the make system will try to detect what CUDA arch should be used by running `$(CUDA_HOME)/extras/demo_suite/deviceQuery` if your computer is unknown. If it fails to detect the CUDA arch, the default value of 70 will be used. The user could override it by `make USE_CUDA=TRUE CUDA_ARCH=80` or `make USE_CUDA=TRUE AMREX_CUDA_ARCH=80`.

After defining these make variables, a number of files, `Make.defs, Make.package` and `Make.rules`, are included in the GNUmakefile. AMReX-based applications do not need to include all directories in AMReX; an application which does not use particles, for example, does not need to include files from the Particle directory in its build. In this simple example, we only need to include `$(AMREX_HOME)/Src/Base/Make.package`. An application code also has its own Make.package file (e.g., `./Make.package` in this example) to append source files to the build system using operator `+=`. Variables for various source files are shown below.

> CEXE_sources  
> C++ source files. Note that C++ source files are assumed to have a .cpp extension.
>
> CEXE_headers  
> C++ headers with .h, .hpp, or .H extension.
>
> cEXE_sources  
> C source files with .c extension.
>
> cEXE_headers  
> C headers with .h extension.
>
> f90EXE_sources  
> Free format Fortran source with .f90 extension.
>
> F90EXE_sources  
> Free format Fortran source with .F90 extension. Note that these Fortran files will go through preprocessing.

In this simple example, the extra source file, `main.cpp` is in the current directory that is already in the build system's search path. If this example has files in a subdirectory (e.g., `mysrcdir`), you will then need to add the following to `Make.package`.

``` bash
VPATH_LOCATIONS += mysrcdir
INCLUDE_LOCATIONS += mysrcdir
```

Here `VPATH_LOCATIONS` and `INCLUDE_LOCATIONS` are the search path for source and header files, respectively.

## Tweaking the Make System

The GNU Make build system is located at `amrex/Tools/GNUMake`. You can read `README.md` and the make files there for more information. Here we will give a brief overview.

Besides building executable, other common make commands include:

> `make cleanconfig`  
> This removes the executable, .o files, and the temporarily generated files for the given build. Note that one can add additional targets to this rule using the double colon (::)
>
> `make clean` and `make realclean`  
> These remove all files generated by make for all builds.
>
> `make help`  
> This shows the rules for compilation.
>
> `make print-xxx`  
> This shows the value of variable xxx. This is very useful for debugging and tweaking the make system.

Compiler flags are set in `amrex/Tools/GNUMake/comps/`. Note that variables like `CXX` and `CXXFLAGS` are reset in that directory and their values in environment variables are disregarded. However, one could override them with make command line arguments (e.g., `make CXX=/path/to/my/mpicxx`). Site-specific setups (e.g., the MPI installation) are in `amrex/Tools/GNUMake/sites/`, which includes a generic setup in `Make.unknown`. You can override the setup by having your own `sites/Make.$(host_name)` file, where variable `host_name` is your host name in the make system and can be found via `make print-host_name`. You can also have an `amrex/Tools/GNUMake/Make.local` file to override various variables. See `amrex/Tools/GNUMake/Make.local.template` for more examples of how to customize the build process.

If you need to pass macro definitions to the preprocessor, you can add them to your make file as follows,

``` bash
DEFINES += -Dmyname1 -Dmyname2=mydefinition
```

To link to an additional library say `foo` with headers located at `foopath/include` and library at `foopath/lib`, you can add the following to your make file before the line that includes AMReX's `Make.defs`,

``` bash
INCLUDE_LOCATIONS += foopath/include
LIBRARY_LOCATIONS += foopath/lib
LIBRARIES += -lfoo
```

## Specifying your own compiler

The `amrex/Tools/GNUMake/Make.local` file can also specify your own compile commands by setting the variables `CXX`, `CC`, `FC`, and `F90`. This might be necessary if your systems contains non-standard names for compiler commands.

For example, the following `amrex/Tools/GNUMake/Make.local` builds AMReX using a specific compiler (in this case `gcc-8`) without MPI. Whenever `USE_MPI` is true, this configuration defaults to the appropriate `mpixxx` command: :

``` bash
ifeq ($(USE_MPI),TRUE)
  CXX = mpicxx
  CC  = mpicc
  FC  = mpif90
  F90 = mpif90
else
  CXX = g++-8
  CC  = gcc-8
  FC  = gfortran-8
  F90 = gfortran-8
endif
```

For building with MPI, we assume `mpicxx`, `mpif90`, etc. provide access to the correct underlying compilers.

## GCC on macOS

The example configuration above should also run on the latest macOS. On macOS the default cxx compiler is clang, whereas the default Fortran compiler is gfortran. Sometimes it is good to avoid mixing compilers, in that case we can use the `Make.local` to force using GCC. However, macOS' Xcode ships with its own (woefully outdated) version of GCC (4.2.1). It is therefore recommended to install GCC using the [homebrew](https://brew.sh) package manager. Running `brew install gcc` installs gcc with names reflecting the version number. If GCC 8.2 is installed, homebrew installs it as `gcc-8`. AMReX can be built using `gcc-8` (with and without MPI) by using the following `amrex/Tools/GNUMake/Make.local`:

``` bash
CXX = g++-8
CC  = gcc-8
FC  = gfortran-8
F90 = gfortran-8

INCLUDE_LOCATIONS += /usr/local/include
```

The additional `INCLUDE_LOCATIONS` are installed using homebrew also. Note that if you are building AMReX using homebrew's gcc, it is recommended that you use homebrew's mpich. Normally it is fine to simply install its binaries: `brew install mpich`. But if you are experiencing problems, we suggest building mpich using homebrew's gcc: `brew install mpich --cc=gcc-8`.

## Fortran

If your code does not use Fortran, you can add `BL_NO_FORT=TRUE` to your makefile to disable Fortran.

## ccache

If you use ccache, you can add `USE_CCACHE=TRUE` to your makefile.

# Building libamrex

If an application code already has its own elaborated build system and wants to use AMReX, an external AMReX library can be created instead. In this approach, one runs `./configure`, followed by `make` and `make install`. Other make options include `make distclean` and `make uninstall`. In the top AMReX directory, one can run `./configure -h` to show the various options for the configure script. In particular, one can specify the installation path for the AMReX library using:

``` bash
./configure --prefix=[AMReX library path]
```

This approach is built on the AMReX GNU Make system. Thus the section on `sec:build:make` is recommended if any fine tuning is needed. The result of `./configure` is `GNUmakefile` in the AMReX top directory. One can modify the make file for fine tuning.

To compile an application code against the external AMReX library, it is necessary to set appropriate compiler flags and set the library paths for linking. To assist with this, when the AMReX library is built, a configuration file is created in `[AMReX library path]/lib/pkgconfig/amrex.pc`. This file contains the Fortran and C++ flags used to compile the AMReX library as well as the appropriate library and include entries.

The following sample GNU Makefile will compile a `main.cpp` source file against an external AMReX library, using the C++ flags and library paths used to build AMReX:

``` bash
AMREX_LIBRARY_HOME ?= [AMReX library path]

LIBDIR := $(AMREX_LIBRARY_HOME)/lib
INCDIR := $(AMREX_LIBRARY_HOME)/include

COMPILE_CPP_FLAGS ?= $(shell awk '/Cflags:/ {$$1=$$2=""; print $$0}' $(LIBDIR)/pkgconfig/amrex.pc)
COMPILE_LIB_FLAGS ?= $(shell awk '/Libs:/ {$$1=$$2=""; print $$0}' $(LIBDIR)/pkgconfig/amrex.pc)

CFLAGS := -I$(INCDIR) $(COMPILE_CPP_FLAGS)
LFLAGS := -L$(LIBDIR) $(COMPILE_LIB_FLAGS)

all:
        g++ -o main.exe main.cpp $(CFLAGS) $(LFLAGS)
```

# Building with CMake

An alternative to the approach described in the section on `sec:build:lib` is to install AMReX as an external library by using the CMake build system. A CMake build is a two-step process. First `cmake` is invoked to create configuration files and makefiles in a chosen directory (`builddir`). This is roughly equivalent to running `./configure` (see the section on `sec:build:lib`). Next, the actual build and installation are performed by invoking `make install` from within `builddir`. This installs the library files in a chosen installation directory (`installdir`). If no installation path is provided by the user, AMReX will be installed in `/path/to/amrex/installdir`. The CMake build process is summarized as follows:

``` console
mkdir /path/to/builddir
cd    /path/to/builddir
cmake [options] -DCMAKE_BUILD_TYPE=[Debug|Release|RelWithDebInfo|MinSizeRel] -DCMAKE_INSTALL_PREFIX=/path/to/installdir  /path/to/amrex
make  install
make  test_install  # optional step to test if the installation is working
```

In the above snippet, `[options]` indicates one or more options for the customization of the build, as described in the subsection on `sec:build:cmake:options`. If the option `CMAKE_BUILD_TYPE` is omitted, `CMAKE_BUILD_TYPE=Release` is assumed. Although the AMReX source could be used as build directory, we advise against doing so. After the installation is complete, `builddir` can be removed.

## Customization options

AMReX build can be customized by setting the value of suitable configuration variables on the command line via the `-D <var>=<value>` syntax, where `<var>` is the variable to set and `<value>` its desired value. For example, one can enable OpenMP support as follows:

``` console
cmake -DAMReX_OMP=YES -DCMAKE_INSTALL_PREFIX=/path/to/installdir  /path/to/amrex
```

In the example above `<var>=AMReX_OMP` and `<value>=YES`. Configuration variables requiring a boolean value are evaluated to true if they are assigned a value of `1`, `ON`, `YES`, `TRUE`, `Y`. Conversely they are evaluated to false if they are assigned a value of `0`, `OFF`, `NO`, `FALSE`, `N`. Boolean configuration variables are case-insensitive. The list of available options is reported in the `table `tab:cmakevar`` below.

<div id="tab:cmakevar">

<table>
<caption>AMReX build options (refer to section <code class="interpreted-text" role="ref">sec:gpu:build</code> for GPU-related options).</caption>
<colgroup>
<col />
<col />
<col />
<col />
</colgroup>
<thead>
<tr class="header">
<th>Variable Name</th>
<th>Description</th>
<th>Default</th>
<th>Possible values</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>CMAKE_Fortran_COMPILER</td>
<td><blockquote>
<p>User-defined Fortran compiler</p>
</blockquote></td>
<td></td>
<td>user-defined</td>
</tr>
<tr class="even">
<td>CMAKE_CXX_COMPILER</td>
<td><blockquote>
<p>User-defined C++ compiler</p>
</blockquote></td>
<td></td>
<td>user-defined</td>
</tr>
<tr class="odd">
<td>CMAKE_Fortran_FLAGS</td>
<td><blockquote>
<p>User-defined Fortran flags</p>
</blockquote></td>
<td></td>
<td>user-defined</td>
</tr>
<tr class="even">
<td>CMAKE_CXX_FLAGS</td>
<td><blockquote>
<p>User-defined C++ flags</p>
</blockquote></td>
<td></td>
<td>user-defined</td>
</tr>
<tr class="odd">
<td>CMAKE_CXX_STANDARD</td>
<td><blockquote>
<p>C++ standard</p>
</blockquote></td>
<td>compiler/17</td>
<td>17, 20</td>
</tr>
<tr class="even">
<td>AMReX_SPACEDIM</td>
<td><blockquote>
<p>Dimension of AMReX build</p>
</blockquote></td>
<td>3 <code>;</code>-separated list</td>
<td>"1;2;3"</td>
</tr>
<tr class="odd">
<td>USE_XSDK_DEFAULTS</td>
<td><blockquote>
<p>Use xSDK defaults settings</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_BUILD_SHARED_LIBS</td>
<td><blockquote>
<p>Build as shared C++ library</p>
</blockquote></td>
<td>NO (unless xSDK)</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_FASTMATH</td>
<td><blockquote>
<p>Enable fast-math optimizations</p>
</blockquote></td>
<td colspan="2">NO (CUDA is ON) | YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_FORTRAN</td>
<td><blockquote>
<p>Enable Fortran language</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_PRECISION</td>
<td><blockquote>
<p>Set the precision of reals</p>
</blockquote></td>
<td>DOUBLE</td>
<td>DOUBLE, SINGLE</td>
</tr>
<tr class="even">
<td>AMReX_PIC</td>
<td><blockquote>
<p>Build Position Independent Code</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_IPO</td>
<td><blockquote>
<p>Interprocedural optimization (IPO/LTO)</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_MPI</td>
<td><blockquote>
<p>Build with MPI support</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_SIMD</td>
<td><blockquote>
<p>Enable SIMD Primitives (using vir::stdx::simd)</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_OMP</td>
<td><blockquote>
<p>Build with OpenMP support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_GPU_BACKEND</td>
<td><blockquote>
<p>Build with on-node, accelerated GPU backend</p>
</blockquote></td>
<td>NONE</td>
<td>NONE, SYCL, HIP, CUDA</td>
</tr>
<tr class="even">
<td>AMReX_GPU_RDC</td>
<td><blockquote>
<p>Build with Relocatable Device Code support</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_FORTRAN_INTERFACES</td>
<td><blockquote>
<p>Build Fortran API</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_LINEAR_SOLVERS</td>
<td><blockquote>
<p>Build AMReX linear solvers</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_LINEAR_SOLVERS_INCFLO</td>
<td><blockquote>
<p>Build AMReX linear solvers for incompressible flow</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_LINEAR_SOLVERS_EM</td>
<td><blockquote>
<p>Build AMReX linear solvers for electromagnetic solvers</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_AMRDATA</td>
<td><blockquote>
<p>Build data services</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_AMRLEVEL</td>
<td><blockquote>
<p>Build AmrLevel class</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_EB</td>
<td><blockquote>
<p>Build Embedded Boundary support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_FFT</td>
<td><blockquote>
<p>Build FFT support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_PARTICLES</td>
<td><blockquote>
<p>Build particle classes</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_PARTICLES_PRECISION</td>
<td><blockquote>
<p>Set reals precision in particle classes</p>
</blockquote></td>
<td>Same as AMReX_PRECISION</td>
<td>DOUBLE, SINGLE</td>
</tr>
<tr class="odd">
<td>AMReX_BASE_PROFILE</td>
<td><blockquote>
<p>Build with basic profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_TINY_PROFILE</td>
<td><blockquote>
<p>Build with tiny profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_TRACE_PROFILE</td>
<td><blockquote>
<p>Build with trace-profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_COMM_PROFILE</td>
<td><blockquote>
<p>Build with comm-profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_MEM_PROFILE</td>
<td><blockquote>
<p>Build with memory-profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_TP_PROFILE</td>
<td><blockquote>
<p>Third-party profiling options</p>
</blockquote></td>
<td>IGNORE</td>
<td>CRAYPAT,FORGE,VTUNE</td>
</tr>
<tr class="odd">
<td>AMReX_TESTING</td>
<td><blockquote>
<p>Build for testing --sets MultiFab initial data to NaN</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_MPI_THREAD_MULTIPLE</td>
<td><blockquote>
<p>Concurrent MPI calls from multiple threads</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_PROFPARSER</td>
<td><blockquote>
<p>Build with profile parser support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_ROCTX</td>
<td><blockquote>
<p>Build with roctx markup profiling support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_FPE</td>
<td><blockquote>
<p>Build with Floating Point Exceptions checks</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_ASSERTIONS</td>
<td><blockquote>
<p>Build with assertions turned on</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_BOUND_CHECK</td>
<td><blockquote>
<p>Enable bound checking in Array4 class</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_EXPORT_DYNAMIC</td>
<td><blockquote>
<p>Enable backtrace on macOS</p>
</blockquote></td>
<td>NO (unless Darwin)</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_SENSEI</td>
<td><blockquote>
<p>Enable the SENSEI in situ infrastructure</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_NO_SENSEI_AMR_INST</td>
<td><blockquote>
<p>Disables the instrumentation in amrex::Amr</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_CONDUIT</td>
<td><blockquote>
<p>Enable Conduit support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_CATALYST</td>
<td><blockquote>
<p>Enable Catalyst support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_ASCENT</td>
<td><blockquote>
<p>Enable Ascent support</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_HYPRE</td>
<td><blockquote>
<p>Enable HYPRE interfaces</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_PETSC</td>
<td><blockquote>
<p>Enable PETSc interfaces</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_SUNDIALS</td>
<td><blockquote>
<p>Enable SUNDIALS interfaces</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_HDF5</td>
<td><blockquote>
<p>Enable HDF5-based I/O</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_HDF5_ZFP</td>
<td><blockquote>
<p>Enable compression with ZFP in HDF5-based I/O</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_PLOTFILE_TOOLS</td>
<td><blockquote>
<p>Build and install plotfile postprocessing tools</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_ENABLE_TESTS</td>
<td><blockquote>
<p>Enable CTest suite</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_TEST_TYPE</td>
<td><blockquote>
<p>Test type -- affects the number of tests</p>
</blockquote></td>
<td>All</td>
<td>All, Small</td>
</tr>
<tr class="even">
<td>AMReX_DIFFERENT_COMPILER</td>
<td><blockquote>
<p>Allow an app to use a different compiler</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_INSTALL</td>
<td><blockquote>
<p>Generate Install Targets</p>
</blockquote></td>
<td>YES</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_PROBINIT</td>
<td><blockquote>
<p>Enable support for probin file</p>
</blockquote></td>
<td>Platform dependent</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_FLATTEN_FOR</td>
<td><blockquote>
<p>Enable flattening of ParallelFor and similar functions for host code</p>
</blockquote></td>
<td>NO</td>
<td>YES, NO</td>
</tr>
<tr class="even">
<td>AMReX_COMPILER_DEFAULT_INLINE</td>
<td><blockquote>
<p>Use default inline behavior of compiler, so far relevant for GCC Only</p>
</blockquote></td>
<td>NO for GCC YES otherwise</td>
<td>YES, NO</td>
</tr>
<tr class="odd">
<td>AMReX_INLINE_LIMIT</td>
<td><blockquote>
<p>Inline limit. Relevant only when AMReX_COMPILER_DEFAULT_INLINE is NO.</p>
</blockquote></td>
<td>43210</td>
<td>Non-negative number</td>
</tr>
</tbody>
</table>

AMReX build options (refer to section `sec:gpu:build` for GPU-related options).

</div>

The option `CMAKE_BUILD_TYPE=Debug` implies `AMReX_ASSERTIONS=YES`. In order to turn off assertions in debug mode, `AMReX_ASSERTIONS=NO` must be set explicitly while invoking CMake.

The `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, and `CMAKE_Fortran_COMPILER` options are used to tell CMake which compiler to use for the compilation of C, C++, and Fortran sources respectively. If those options are not set by the user, CMake will use the system default compilers.

The options `CMAKE_Fortran_FLAGS` and `CMAKE_CXX_FLAGS` allow the user to set their own compilation flags for Fortran and C++ source files respectively. If `CMAKE_Fortran_FLAGS`/ `CMAKE_CXX_FLAGS` are not set by the user, they will be initialized with the value of the environmental variables `FFLAGS`/ `CXXFLAGS`. If neither `FFLAGS`/ `CXXFLAGS` nor `CMAKE_Fortran_FLAGS`/ `CMAKE_CXX_FLAGS` are defined, AMReX default flags are used.

For a detailed explanation of GPU support in AMReX CMake, refer to section `sec:gpu:build`.

## CMake and macOS

While not strictly necessary when using homebrew on macOS, it is highly recommended that the user specifies `-DCMAKE_C_COMPILER=$(which gcc-X) -DCMAKE_CXX_COMPILER=$(which g++-X)` (where X is the GCC version installed by homebrew) when using gfortran. This is because homebrew's CMake defaults to the Clang C/C++ compiler. Normally Clang plays well with gfortran, but if there are some issues, we recommend telling CMake to use gcc for C/C++ also.

## Importing AMReX into your CMake project

In order to import AMReX into your CMake project, you need to include the following line in the appropriate CMakeLists.txt file:

``` cmake
find_package(AMReX)
```

Calls to `find_package(AMReX)` will find a valid installation of AMReX, if present, and import its settings and targets into your CMake project. Imported AMReX targets can be linked to any of your targets, after they have been made available following a successful call to `find_package(AMReX)`, by including the following line in the appropriate CMakeLists.txt file:

``` cmake
target_link_libraries( <your-target-name> PUBLIC AMReX::<amrex-target-name> )
```

In the above snippet, `<amrex-target-name>` is any of the targets listed in the table below.

<div id="tab:cmaketargets">

<table>
<caption>AMReX targets available for import.</caption>
<colgroup>
<col />
<col />
</colgroup>
<thead>
<tr class="header">
<th>Target name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>amrex_1d</td>
<td><blockquote>
<p>AMReX library in 1D</p>
</blockquote></td>
</tr>
<tr class="even">
<td>amrex_2d</td>
<td><blockquote>
<p>AMReX library in 2D</p>
</blockquote></td>
</tr>
<tr class="odd">
<td>amrex_3d</td>
<td><blockquote>
<p>AMReX library in 3D</p>
</blockquote></td>
</tr>
<tr class="even">
<td>amrex</td>
<td><blockquote>
<p>AMReX library (alias, points to last dim)</p>
</blockquote></td>
</tr>
<tr class="odd">
<td>Flags_CXX</td>
<td><blockquote>
<p>C++ flags preset (interface)</p>
</blockquote></td>
</tr>
<tr class="even">
<td>Flags_Fortran</td>
<td><blockquote>
<p>Fortran flags preset (interface)</p>
</blockquote></td>
</tr>
<tr class="odd">
<td>Flags_FPE</td>
<td><blockquote>
<p>Floating Point Exception flags (interface)</p>
</blockquote></td>
</tr>
</tbody>
</table>

AMReX targets available for import.

</div>

The options used to configure the AMReX build may result in certain parts, or `components`, of the AMReX source code to be excluded from compilation. For example, setting `-DAMReX_LINEAR_SOLVERS=no` at configure time prevents the compilation of AMReX linear solvers code. Your CMake project can check which component is included in the AMReX library via \`find_package\`:

``` cmake
find_package(AMReX REQUIRED <components-list>)
```

The keyword `REQUIRED` in the snippet above will cause a fatal error if AMReX is not found, or if it is found but the components listed in `<components-list>` are not include in the installation. A list of AMReX component names and related configure options are shown in the table below.

<div id="tab:cmakecomponents">

| Option                      | Component        |
|-----------------------------|------------------|
| AMReX_SPACEDIM              | 1D, 2D, 3D       |
| AMReX_PRECISION             | DOUBLE, SINGLE   |
| AMReX_FORTRAN               | FORTRAN          |
| AMReX_PIC                   | PIC              |
| AMReX_MPI                   | MPI              |
| AMReX_SIMD                  | SIMD             |
| AMReX_OMP                   | OMP              |
| AMReX_GPU_BACKEND           | CUDA, HIP, SYCL  |
| AMReX_FORTRAN_INTERFACES    | FINTERFACES      |
| AMReX_LINEAR_SOLVERS        | LSOLVERS         |
| AMReX_LINEAR_SOLVERS_INCFLO | LSOLVERS_INCFLO  |
| AMReX_LINEAR_SOLVERS_EM     | LSOLVERS_EM      |
| AMReX_AMRDATA               | AMRDATA          |
| AMReX_AMRLEVEL              | AMRLEVEL         |
| AMReX_EB                    | EB               |
| AMReX_FFT                   | FFT              |
| AMReX_PARTICLES             | PARTICLES        |
| AMReX_PARTICLES_PRECISION   | PDOUBLE, PSINGLE |
| AMReX_BASE_PROFILE          | BASEP            |
| AMReX_TINY_PROFILE          | TINYP            |
| AMReX_TRACE_PROFILE         | TRACEP           |
| AMReX_COMM_PROFILE          | COMMP            |
| AMReX_MEM_PROFILE           | MEMP             |
| AMReX_PROFPARSER            | PROFPARSER       |
| AMReX_FPE                   | FPE              |
| AMReX_ASSERTIONS            | ASSERTIONS       |
| AMReX_SENSEI                | SENSEI           |
| AMReX_CONDUIT               | CONDUIT          |
| AMReX_ASCENT                | ASCENT           |
| AMReX_HYPRE                 | HYPRE            |
| AMReX_PLOTFILE_TOOLS        | PFTOOLS          |

AMReX components.

</div>

As an example, consider the following CMake code:

``` cmake
find_package(AMReX REQUIRED 3D EB)
target_link_libraries(Foo PUBLIC AMReX::amrex_3d)
```

The code in the snippet above checks whether an AMReX installation with 3D and Embedded Boundary support is available on the system. If so, AMReX is linked to target `Foo` and AMReX flags preset is used to compile `Foo`'s C++ sources. If no AMReX installation is found or if the available one was built without 3D or Embedded Boundary support, a fatal error is issued.

You can tell CMake to look for the AMReX library in non-standard paths by setting the environmental variable `AMReX_ROOT` to point to the AMReX installation directory or by adding `-DAMReX_ROOT=<path/to/amrex/installation/directory>` to the `cmake` invocation. More details on `find_package` can be found [here](https://cmake.org/cmake/help/v3.25/command/find_package.html).

# AMReX on Windows

The AMReX team does development on Linux machines, from laptops to supercomputers. Many people also use AMReX on Macs without issues.

We do not officially support AMReX on Windows, and many of us do not have access to any Windows machines. However, we believe there are no fundamental issues for it to work on Windows.

\(1\) AMReX mostly uses standard C++17. We run continuous integration tests on Windows with MSVC and Clang compilers.

\(2\) We use POSIX signal handling when floating point exceptions, segmentation faults, etc. happen. This capability is not supported on Windows.

\(3\) Memory profiling is an optional feature in AMReX that is not enabled by default. It reads memory system information from the OS to give us a summary of our memory usage. This is not supported on Windows.

# Spack

AMReX can be installed using the scientific software package manager Spack. Spack supports multiple versions and configurations of AMReX across a wide variety of platforms and environments. To learn more about Spack visit http://www.spack.io. For system requirements and installation instructions please see https://spack.readthedocs.io/.

Once Spack has been downloaded and the Spack environment enabled, AMReX can be installed with the command,

``` bash
spack install amrex
```

This will install the latest release of AMReX and required dependencies if needed.

AMReX can be built in several combinations of versions and configurations. Available options can be viewed by typing,

``` bash
spack info amrex
```

For example, suppose we want to install the development version of AMReX for a two dimensional simulation with Cuda support for Cuda Architecture `sm_60`. Then we would use the install commands,

``` bash
spack install amrex@develop dimensions=2 +cuda cuda_arch=60
```
