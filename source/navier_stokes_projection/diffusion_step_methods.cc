#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step()
{
  /* System matrix setup */
  assemble_velocity_advection_matrix();

  if (!flag_diffusion_matrix_assembled)
  {
    velocity_mass_plus_laplace_matrix = 0.;

    velocity_mass_plus_laplace_matrix.add
    (1.0 / parameters.Re,
     velocity_laplace_matrix);

    velocity_mass_plus_laplace_matrix.add
    (time_stepping.get_alpha()[0] / time_stepping.get_next_step_size(),
     velocity_mass_matrix);

    if (!parameters.time_stepping_parameters.adaptive_time_stepping)
      flag_diffusion_matrix_assembled = true; 
  }
  velocity_system_matrix.copy_from(velocity_mass_plus_laplace_matrix);
  velocity_system_matrix.add(1., velocity_advection_matrix);

  /* Right hand side setup */
  assemble_diffusion_step_rhs();
}

template <int dim>
void NavierStokesProjection<dim>::
solve_diffusion_step(const bool reinit_prec)
{
  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_velocity(velocity_rhs);
  distributed_velocity = velocity.solution;

  if (reinit_prec)
    diffusion_step_preconditioner.initialize(velocity_system_matrix);

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
    solver.solve(velocity_system_matrix,
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
}
}
// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step();

template void RMHD::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_diffusion_step(const bool);
