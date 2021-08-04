# Set the source files to be compiled
SET(TARGET_SRC
    basic_parameters.cc
    linear_solver_parameters.cc
    assembly_data.cc
    benchmark_data.cc
    boundary_conditions.cc
    convergence_test.cc
    discrete_time.cc
    entities_structs.cc
    equation_data.cc
    problem_class.cc
    run_time_parameters.cc
    time_discretization.cc
    utility.cc
    # incompressible navier stokes
    navier_stokes_projection.cc
    navier_stokes_projection/assemble_advection_matrix.cc
    navier_stokes_projection/assemble_diffusion_rhs.cc
    navier_stokes_projection/assemble_mass_and_laplace_matrices.cc
    navier_stokes_projection/assemble_poisson_prestep_rhs.cc
    navier_stokes_projection/assemble_projection_rhs.cc
    navier_stokes_projection/assembly_data.cc
    navier_stokes_projection/diffusion_step_methods.cc
    navier_stokes_projection/poisson_prestep_methods.cc
    navier_stokes_projection/projection_step_methods.cc
    navier_stokes_projection/setup.cc
    navier_stokes_projection/solve.cc
    navier_stokes_projection/timestep_update.cc
    # convection diffusion equation
    convection_diffusion_problem.cc
    convection_diffusion_solver.cc
    convection_diffusion_solver/assemble_advection_matrix.cc
    convection_diffusion_solver/assemble_constant_matrices.cc
    convection_diffusion_solver/assemble_rhs.cc
    convection_diffusion_solver/assembly_data.cc
    convection_diffusion_solver/setup.cc
    convection_diffusion_solver/solve.cc
)
ADD_LIBRARY(rotatingMHD SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(rotatingMHD)
