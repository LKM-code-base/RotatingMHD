#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step()
{
  if (parameters.verbose)
    *pcout << "    Navier Stokes: Assembling the diffusion step...";

  /* System matrix setup */

  /* This if scope makes sure that if the time step is 
  constant, the following matrix summation is only done once */
  if (!flag_diffusion_matrix_assembled)
  {
    velocity_mass_plus_laplace_matrix = 0.;

    velocity_mass_plus_laplace_matrix.add
    (time_stepping.get_alpha()[0] / time_stepping.get_next_step_size(),
     velocity_mass_matrix);

    velocity_mass_plus_laplace_matrix.add
    (time_stepping.get_gamma()[0] / parameters.Re,
     velocity_laplace_matrix);

    velocity_mass_plus_laplace_matrix.add
    (parameters.grad_div_parameter,
     grad_div_method_matrix);
    
    if (!parameters.time_stepping_parameters.adaptive_time_stepping)
      flag_diffusion_matrix_assembled = true; 
  }

  /* In case of a semi-implicit scheme, the advection matrix has to be
  assembled and added to the system matrix */
  if (parameters.flag_semi_implicit_convection)
  {
    assemble_velocity_advection_matrix();
    velocity_system_matrix.copy_from(velocity_mass_plus_laplace_matrix);
    velocity_system_matrix.add(1. , velocity_advection_matrix);
  }
  /* Right hand side setup */
  assemble_diffusion_step_rhs();

  if (parameters.verbose)
    *pcout << " done." << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::
solve_diffusion_step(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "    Navier Stokes: Solving the diffusion step...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Diffusion step - Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_velocity(velocity_rhs);
  distributed_velocity = velocity.solution;

  /* The following pointer holds the address to the correct matrix 
  depending on if the semi-implicit scheme is chosen or not */
  const LinearAlgebra::MPI::SparseMatrix  * system_matrix;
  if (parameters.flag_semi_implicit_convection ||
      flag_initializing)
    system_matrix = &velocity_system_matrix;
  else
    system_matrix = &velocity_mass_plus_laplace_matrix;

  if (reinit_prec)
    diffusion_step_preconditioner.initialize(*system_matrix);

  SolverControl solver_control(parameters.n_maximum_iterations,
                               std::max(parameters.relative_tolerance * velocity_rhs.l2_norm(),
                                        absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverGMRES solver(solver_control,
                                      MPI_COMM_WORLD);
  #else
    LinearAlgebra::SolverGMRES solver(solver_control);
  #endif

  try
  {
    solver.solve(*system_matrix,
                 distributed_velocity,
                 velocity_rhs,
                 diffusion_step_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the diffusion step: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception in the solve method of the diffusion step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  velocity.constraints.distribute(distributed_velocity);

  velocity.solution = distributed_velocity;

  if (parameters.verbose)
    *pcout << " done." << std::endl;

  if (parameters.verbose)
    *pcout << "    Number of GMRES iterations: " << solver_control.last_step()
           << ", "
           << "final residual: " << solver_control.last_value() << "."
           << std::endl;
}
}
// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step();

template void RMHD::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_diffusion_step(const bool);
