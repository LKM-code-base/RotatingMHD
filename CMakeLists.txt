##
#  CMake script
##
MESSAGE("====================================================")
MESSAGE("=============== Configuring RotatingMHD ============")
MESSAGE("========= An adaptive parallel flow solver =========")
MESSAGE("========= for the rotating magnetohydrodynamics =====")
MESSAGE("====================================================")
##
# Is the cmake version sufficient?
##
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

##
# Is the deal.ii library installed?
##
FIND_PACKAGE(deal.II 9.0.1 REQUIRED QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


##
# Are all dependencies fulfilled?
##

#[[
IF(NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_P4EST)

   MESSAGE(FATAL_ERROR "\n"
   "This library requires a deal.II installation built with support for Trilinos and p4est but one or both of these appears to be missing!"
   "This library requires a deal.II library that was configured with the following options:\n"
   "     DEAL_II_WITH_P4EST = ON\n"
   "     DEAL_II_WITH_TRILINOS = ON\n"
   "However, the deal.II library found at ${DEAL_II_PATH} was configured with these options\n"
   "     DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}\n"
   "     DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}\n"
   "which conflict with the requirements.\n"
   "One or both of the aforementioned combinations of prerequisites are not met by your installation, but at least one is required for\n"
   "this tutorial step.")

   IF(NOT DEAL_II_WITH_TRILINOS)
      MESSAGE(FATAL_ERROR
              "\n-- deal.II was built without support for Trilinos!\n")
   ENDIF()

   IF(NOT DEAL_II_WITH_P4EST)
      MESSAGE(FATAL_ERROR
      "\n-- deal.II was built without support for p4est!\n")
   ENDIF()

ENDIF()
]]

# Set the source files to be compiled
SET(TARGET_SRC
    source/assembly_data.cc
    source/basic_parameters.cc
    source/benchmark_data.cc
    source/boundary_conditions.cc
    source/convergence_test.cc
    source/discrete_time.cc
    source/entities_structs.cc
    source/equation_data.cc
    source/heat_equation.cc
    source/linear_solver_parameters.cc
    source/problem_class.cc
    source/run_time_parameters.cc
    source/time_discretization.cc
    source/utility.cc
    source/navier_stokes_projection.cc
    source/navier_stokes_projection/assemble_advection_matrix.cc
    source/navier_stokes_projection/assemble_diffusion_rhs.cc
    source/navier_stokes_projection/assemble_mass_and_laplace_matrices.cc
    source/navier_stokes_projection/assemble_poisson_prestep_rhs.cc
    source/navier_stokes_projection/assemble_projection_rhs.cc
    source/navier_stokes_projection/assembly_data.cc
    source/navier_stokes_projection/diffusion_step_methods.cc
    source/navier_stokes_projection/poisson_prestep_methods.cc
    source/navier_stokes_projection/projection_step_methods.cc
    source/navier_stokes_projection/setup.cc
    source/navier_stokes_projection/solve.cc
    source/navier_stokes_projection/timestep_update.cc
    source/heat_equation/assemble_advection_matrix.cc
    source/heat_equation/assemble_constant_matrices.cc
    source/heat_equation/assemble_rhs.cc
    source/heat_equation/assembly_data.cc
    source/heat_equation/setup.cc
    source/heat_equation/solve.cc
)

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)

PROJECT(rotatingMHD)

ADD_LIBRARY(rotatingMHD  ${TARGET_SRC})

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(rotatingMHD)

ADD_SUBDIRECTORY(applications)

# Set up unit tests
IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(tests)
ENDIF()

MESSAGE("====================================================")
