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
    source/equation_data.cc
    source/run_time_parameters.cc
    source/assembly_data.cc
    source/projection_solver.cc
    source/projection_solver/make_grid.cc
    source/projection_solver/setup_dofs.cc
    source/projection_solver/problem_setup.cc
    source/projection_solver/assemble_mass_and_laplace_matrices.cc
    source/projection_solver/assemble_advection_matrix.cc
    source/projection_solver/assemble_diffusion_step_rhs.cc
    source/projection_solver/assemble_projection_step_rhs.cc
    source/projection_solver/diffusion_step.cc
    source/projection_solver/projection_step.cc
    source/projection_solver/correction_step.cc
    source/projection_solver/output_results.cc
    source/projection_solver/time_stepping.cc
#     source/assembly_data.cc
#     source/buoyant_fluid_solver.cc
#     source/gravity_field.cc
#     source/grid_factory.cc
#     source/initial_values.cc
#     source/local_temperature_assemble.cc
#     source/local_stokes_assemble.cc
#     source/make_grid.cc
#     source/parameters.cc
#     source/postprocessor.cc
#     source/postprocess_solution.cc
#     source/postprocess_benchmark.cc
#     source/setup.cc
#     source/snapshot_information.cc
#     source/solve_navier_stokes.cc
#     source/solve_temperature.cc
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
