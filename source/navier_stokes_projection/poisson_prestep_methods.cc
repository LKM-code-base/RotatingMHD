#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_poisson_prestep()
{
  // Set external functions to their start time
  if (body_force_ptr != nullptr)
    body_force_ptr->set_time(time_stepping.get_start_time());
  if (gravity_vector_ptr != nullptr)
    gravity_vector_ptr->set_time(time_stepping.get_start_time());

  /* System matrix setup */
  // System matrix is constant and assembled in the
  // in the NavierStokesProjection::setup method.

  /* Right hand side setup */
  assemble_poisson_prestep_rhs();
}

template <int dim>
void
NavierStokesProjection<dim>::
solve_poisson_prestep()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Solving the Poisson pre-step...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Poisson pre-step - Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_old_old_pressure(pressure->distributed_vector);
  distributed_old_old_pressure = pressure->old_old_solution;

  LinearAlgebra::MPI::PreconditionILU::AdditionalData preconditioner_data;
  #ifdef USE_PETSC_LA
    preconditioner_data.level = 1;
  #else
    preconditioner_data.ilu_fill = 2;
    preconditioner_data.overlap = 1;
    preconditioner_data.ilu_rtol = 1.01;
    preconditioner_data.ilu_atol = 1e-5;
  #endif

  poisson_prestep_preconditioner.initialize(pressure_laplace_matrix,
                                            preconditioner_data);

  SolverControl solver_control(
    parameters.poisson_prestep_solver_parameters.n_maximum_iterations,
    std::max(parameters.poisson_prestep_solver_parameters.relative_tolerance *
             poisson_prestep_rhs.l2_norm(),
             parameters.poisson_prestep_solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverCG solver(solver_control,
                                   MPI_COMM_WORLD);
  #else
    LinearAlgebra::SolverCG solver(solver_control);
  #endif

  try
  {
    solver.solve(pressure_laplace_matrix,
                 distributed_old_old_pressure,
                 poisson_prestep_rhs,
                 poisson_prestep_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the poisson pre-step: " << std::endl
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
    std::cerr << "Unknown exception in the solve method of the poisson pre-step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  pressure->constraints.distribute(distributed_old_old_pressure);

  if (flag_normalize_pressure)
    VectorTools::subtract_mean_value(distributed_old_old_pressure);

  pressure->old_old_solution = distributed_old_old_pressure;

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Number of CG iterations: "
           << solver_control.last_step()
           << ", Final residual: " << solver_control.last_value() << "."
           << std::endl;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_poisson_prestep();
template void RMHD::NavierStokesProjection<3>::assemble_poisson_prestep();

template void RMHD::NavierStokesProjection<2>::solve_poisson_prestep();
template void RMHD::NavierStokesProjection<3>::solve_poisson_prestep();
