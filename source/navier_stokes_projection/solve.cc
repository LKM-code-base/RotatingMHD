#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::solve()
{
  if (velocity->solution.size() != diffusion_step_rhs.size())
  {
    setup();

    diffusion_step(true);

    projection_step(true);

    pressure_correction(true);
  }
  else
  {
    diffusion_step(time_stepping.get_step_number() %
                   parameters.preconditioner_update_frequency == 0);

    projection_step(false);

    pressure_correction(false);
  }

  phi->update_solution_vectors();
}

template <int dim>
void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_diffusion_step();

  /* Solve linear system */
  solve_diffusion_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_projection_step();

  /* Solve linear system */
  solve_projection_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::pressure_correction(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Pressure correction step...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Pressure correction step");

  switch (parameters.pressure_correction_scheme)
    {
      case RunTimeParameters::PressureCorrectionScheme::standard:
        {
        // In the following scope we create temporal non ghosted copies
        // of the pertinent vectors to be able to perform algebraic
        // operations.
          LinearAlgebra::MPI::Vector distributed_old_pressure(pressure->distributed_vector);
          LinearAlgebra::MPI::Vector distributed_phi(phi->distributed_vector);

          distributed_old_pressure  = pressure->old_solution;
          distributed_phi           = phi->solution;

          distributed_old_pressure  += distributed_phi;

          if (flag_normalize_pressure)
            VectorTools::subtract_mean_value(distributed_old_pressure);

          pressure->solution = distributed_old_pressure;
        }
        break;
      case RunTimeParameters::PressureCorrectionScheme::rotational:
        // In the following scope we create temporal non ghosted copies
        // of the pertinent vectors to be able to perform the solve()
        // operation.
        {
          LinearAlgebra::MPI::Vector distributed_pressure(pressure->distributed_vector);
          LinearAlgebra::MPI::Vector distributed_old_pressure(pressure->distributed_vector);
          LinearAlgebra::MPI::Vector distributed_phi(phi->distributed_vector);

          distributed_pressure      = pressure->solution;
          distributed_old_pressure  = pressure->old_solution;
          distributed_phi           = phi->solution;

          // The divergence of the velocity field is projected into a
          // unconstrained pressure space through the following solve
          // operation.
          SolverControl solver_control(
            parameters.correction_step_solver_parameters.n_maximum_iterations,
            std::max(parameters.correction_step_solver_parameters.relative_tolerance *
                       correction_step_rhs.l2_norm(),
                     parameters.correction_step_solver_parameters.absolute_tolerance));

          if (reinit_prec)
            correction_step_preconditioner.initialize(projection_mass_matrix);

          #ifdef USE_PETSC_LA
            LinearAlgebra::SolverCG solver(solver_control,
                                           MPI_COMM_WORLD);
          #else
            LinearAlgebra::SolverCG solver(solver_control);
          #endif

          try
          {
            solver.solve(projection_mass_matrix,
                         distributed_pressure,
                         correction_step_rhs,
                         correction_step_preconditioner);
          }
          catch (std::exception &exc)
          {
            std::cerr << std::endl << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            std::cerr << "Exception in the solve method of the pressure "
                         "correction step: " << std::endl
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
            std::cerr << "Unknown exception in the solve method of the pressure "
                         "correction step!" << std::endl
                      << "Aborting!" << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            std::abort();
          }

          // The projected divergence is scaled and the old pressure
          // is added to it
          distributed_pressure.sadd(parameters.C2 / parameters.C6,
                                    1.,
                                    distributed_old_pressure);

          // Followed by the addition of pressure correction computed
          // in the projection step
          distributed_pressure += distributed_phi;

          // The pressure's constraints are distributed to the
          // solution vector to consider the case of Dirichlet
          // boundary conditions on the pressure field.
          pressure->constraints.distribute(distributed_pressure);

          // If the pressure field is defined only up to a constant,
          // a zero mean value constraint is enforced
          if (flag_normalize_pressure)
            VectorTools::subtract_mean_value(distributed_pressure);

          // Pass the distributed vector to its ghost counterpart.
          pressure->solution = distributed_pressure;
        }

        break;
      default:
        Assert(false, ExcNotImplemented());
    };

  if (parameters.verbose)
    *pcout << " done!" << std::endl << std::endl;
}


} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::solve();
template void RMHD::NavierStokesProjection<3>::solve();

template void RMHD::NavierStokesProjection<2>::diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::diffusion_step(const bool);

template void RMHD::NavierStokesProjection<2>::projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::projection_step(const bool);

template void RMHD::NavierStokesProjection<2>::pressure_correction(const bool);
template void RMHD::NavierStokesProjection<3>::pressure_correction(const bool);
