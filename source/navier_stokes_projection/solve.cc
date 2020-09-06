#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::solve(const unsigned int step)
{
  diffusion_step((step % parameters.solver_update_preconditioner == 0) ||
                 (step == time_stepping.get_order()));
  projection_step((step == time_stepping.get_order()));
  pressure_correction((step == time_stepping.get_order()));
}

template <int dim>
void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
{
  // In the following scopes we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the sadd()
  // operations.
  {
    const std::vector<double> phi = time_stepping.get_phi();

    LinearAlgebra::MPI::Vector distributed_old_velocity(velocity_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity.old_solution;
    distributed_old_old_velocity  = velocity.old_old_solution;
    distributed_old_velocity.sadd(phi[1],
                                  phi[0],
                                  distributed_old_old_velocity);
    extrapolated_velocity = distributed_old_velocity;
  }

  {
    LinearAlgebra::MPI::Vector distributed_old_pressure(pressure_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_phi(pressure_rhs);
    LinearAlgebra::MPI::Vector distributed_phi(pressure_rhs);
    distributed_old_pressure  = pressure.old_solution;
    distributed_old_old_phi   = old_old_phi;
    distributed_phi           = old_phi;
    /*
     * These coefficients are wrong in case of a variable size of the time step.
     */
    distributed_old_pressure.sadd(1.,
                                  4. / 3.,
                                  distributed_phi);
    distributed_old_pressure.sadd(1.,
                                  -1. / 3.,
                                  distributed_old_old_phi);
    pressure_tmp = distributed_old_pressure;
  }

  {

    const std::vector<double> alpha = time_stepping.get_alpha();

    LinearAlgebra::MPI::Vector distributed_old_velocity(velocity_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity.old_solution;
    distributed_old_old_velocity  = velocity.old_old_solution;
    distributed_old_velocity.sadd(alpha[1],
                                  alpha[0],
                                  distributed_old_old_velocity);
    velocity_tmp = distributed_old_velocity;
  }

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
  // This boolean will be used later when a proper solver is chosen
  (void)reinit_prec;

  switch (parameters.projection_method)
    {
      case RunTimeParameters::ProjectionMethod::standard:
        pressure.solution += phi;
        break;
      case RunTimeParameters::ProjectionMethod::rotational:
        // In the following scope we create temporal non ghosted copies
        // of the pertinent vectors to be able to perform the solve()
        // operation.
        {
          LinearAlgebra::MPI::Vector distributed_pressure(pressure_rhs);
          LinearAlgebra::MPI::Vector distributed_old_pressure(pressure_rhs);
          LinearAlgebra::MPI::Vector distributed_phi(pressure_rhs);

          distributed_pressure      = pressure.solution;
          distributed_old_pressure  = pressure.old_solution;
          distributed_phi           = phi;

          SolverControl solver_control(parameters.n_maximum_iterations,
                                       std::max(parameters.relative_tolerance * pressure_rhs.l2_norm(),
                                                absolute_tolerance));

          if (reinit_prec)
            correction_step_preconditioner.initialize(pressure_mass_matrix);

          #ifdef USE_PETSC_LA
            LinearAlgebra::SolverCG solver(solver_control,
                                           MPI_COMM_WORLD);
          #else
            LinearAlgebra::SolverCG solver(solver_control);
          #endif

          try
          {
            solver.solve(pressure_mass_matrix,
                         distributed_pressure,
                         pressure_rhs,
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

          pressure.constraints.distribute(distributed_pressure);

          distributed_pressure.sadd(1.0 / parameters.Re, 1., distributed_old_pressure);
          distributed_pressure += distributed_phi;

          pressure.solution = distributed_pressure;
        }

        break;
      default:
        Assert(false, ExcNotImplemented());
    };
}


} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::solve(const unsigned int);
template void RMHD::NavierStokesProjection<3>::solve(const unsigned int);

template void RMHD::NavierStokesProjection<2>::diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::diffusion_step(const bool);

template void RMHD::NavierStokesProjection<2>::projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::projection_step(const bool);

template void RMHD::NavierStokesProjection<2>::pressure_correction(const bool);
template void RMHD::NavierStokesProjection<3>::pressure_correction(const bool);
