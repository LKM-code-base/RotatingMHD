#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/utility.h>

#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::assemble_projection_step()
{
  /* System matrix setup */
  // System matrix is constant and assembled in the
  // NavierStokesProjection constructor.

  /* Right hand side setup */
  assemble_projection_step_rhs();
}


template <int dim>
void NavierStokesProjection<dim>::solve_projection_step
(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Solving the projection step...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Projection step - Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_phi(phi->distributed_vector);
  distributed_phi = phi->solution;

  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = parameters.projection_step_solver_parameters;
  if (reinit_prec)
  {
    build_preconditioner(projection_step_preconditioner,
                         phi_laplace_matrix,
                         solver_parameters.preconditioner_parameters_ptr,
                         (phi->get_finite_element().degree > 1? true: false));
  }

  AssertThrow(projection_step_preconditioner != nullptr,
              ExcMessage("The pointer to the projection step's preconditioner has not being initialized."));

  SolverControl solver_control(
    solver_parameters.n_maximum_iterations,
    std::max(solver_parameters.relative_tolerance * projection_step_rhs.l2_norm(),
             solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverCG solver(solver_control,
                                   mpi_communicator);
  #else
    LinearAlgebra::SolverCG solver(solver_control);
  #endif

  try
  {
    solver.solve(phi_laplace_matrix,
                 distributed_phi,
                 projection_step_rhs,
                 *projection_step_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the projection step: " << std::endl
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
    std::cerr << "Unknown exception in the solve method of the projection step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  phi->get_constraints().distribute(distributed_phi);

  phi->solution = distributed_phi;

  if (flag_normalize_pressure)
  {
    const LinearAlgebra::MPI::Vector::value_type mean_value
      = VectorTools::compute_mean_value(phi->get_dof_handler(),
                                        QGauss<dim>(phi->get_finite_element().degree + 1),
                                        phi->solution,
                                        0);

    distributed_phi.add(-mean_value);
    phi->solution = distributed_phi;
  }

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Number of CG iterations: "
           << solver_control.last_step()
           << ", Final residual: " << solver_control.last_value() << "."
           << std::endl;
}

}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step();

template void RMHD::NavierStokesProjection<2>::solve_projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_projection_step(const bool);
