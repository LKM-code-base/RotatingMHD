#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

using namespace RunTimeParameters;

template <int dim>
void NavierStokesProjection<dim>::solve()
{
  if (velocity->solution.size() != velocity_tmp.size())
  {
    setup();

    diffusion_step(true);

    projection_step(true);

    pressure_correction(true);
  }
  else 
  {
    diffusion_step(time_stepping.get_step_number() % 
                    parameters.solver_update_preconditioner == 0);

    projection_step(false);

    pressure_correction(false);
  }

  phi->update_solution_vectors();
}

template <int dim>
void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Diffusion step..." << std::endl;

  // In the following scopes we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the sadd()
  // operations.
  {
    const std::vector<double> eta = time_stepping.get_eta();
    AssertIsFinite(eta[0]);
    AssertIsFinite(eta[1]);

    LinearAlgebra::MPI::Vector distributed_old_velocity(velocity_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity->old_solution;
    distributed_old_old_velocity  = velocity->old_old_solution;
    distributed_old_velocity.sadd(eta[0],
                                  eta[1],
                                  distributed_old_old_velocity);
    extrapolated_velocity = distributed_old_velocity;
  }

  {
    const std::vector<double> alpha           = time_stepping.get_alpha();
    const std::vector<double> old_alpha_zero  = time_stepping.get_old_alpha_zero();
    const std::vector<double> old_step_size   = time_stepping.get_old_step_size();
    AssertIsFinite(time_stepping.get_next_step_size());
    AssertIsFinite(alpha[1]);
    AssertIsFinite(alpha[2]);
    AssertIsFinite(old_alpha_zero[0]);
    AssertIsFinite(old_alpha_zero[1]);
    AssertIsFinite(old_step_size[0]);
    AssertIsFinite(old_step_size[1]);

    LinearAlgebra::MPI::Vector distributed_old_pressure(pressure_rhs);
    LinearAlgebra::MPI::Vector distributed_old_phi(pressure_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_phi(pressure_rhs);
    distributed_old_pressure  = pressure->old_solution;
    distributed_old_phi       = phi->old_solution;
    distributed_old_old_phi   = phi->old_old_solution;

    distributed_old_pressure.sadd(1.,
                                  - old_step_size[0] /
                                  time_stepping.get_next_step_size() *
                                  alpha[1] / old_alpha_zero[0],
                                  distributed_old_phi);

    distributed_old_pressure.sadd(1.,
                                  - old_step_size[1] /
                                  time_stepping.get_next_step_size() *
                                  alpha[2] / old_alpha_zero[1],
                                  distributed_old_old_phi);
                                  
    pressure_tmp = distributed_old_pressure;
  }

  {
    const std::vector<double> alpha = time_stepping.get_alpha();
    AssertIsFinite(alpha[1]);
    AssertIsFinite(alpha[2]);
    AssertIsFinite(time_stepping.get_next_step_size());
    AssertIsFinite(alpha[1] / time_stepping.get_next_step_size());
    AssertIsFinite(alpha[2] / time_stepping.get_next_step_size());

    LinearAlgebra::MPI::Vector distributed_old_velocity(velocity_rhs);
    LinearAlgebra::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity->old_solution;
    distributed_old_old_velocity  = velocity->old_old_solution;
    distributed_old_velocity.sadd(alpha[1] / time_stepping.get_next_step_size(),
                                  alpha[2] / time_stepping.get_next_step_size(),
                                  distributed_old_old_velocity);
    velocity_tmp = distributed_old_velocity;
  }

  /* Assemble linear system */
  assemble_diffusion_step();

  norm_diffusion_rhs = velocity_rhs.l2_norm();

  /* Solve linear system */
  solve_diffusion_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_projection_step();

  norm_projection_rhs = pressure_rhs.l2_norm();

  /* Solve linear system */
  solve_projection_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::pressure_correction(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Pressure correction step..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Pressure correction step");

  switch (parameters.projection_method)
  {
    case RunTimeParameters::ProjectionMethod::standard:
      pressure->solution += phi->solution;
      break;
    case RunTimeParameters::ProjectionMethod::rotational:
      // In the following scope we create temporal non ghosted copies
      // of the pertinent vectors to be able to perform the solve()
      // operation.
    {
      LinearAlgebra::MPI::Vector distributed_pressure(pressure_rhs);
      LinearAlgebra::MPI::Vector distributed_old_pressure(pressure_rhs);
      LinearAlgebra::MPI::Vector distributed_phi(pressure_rhs);

      distributed_pressure      = pressure->solution;
      distributed_old_pressure  = pressure->old_solution;
      distributed_phi           = phi->solution;

      pressure_rhs /= (!flag_initializing ?
          time_stepping.get_alpha()[0] /
          time_stepping.get_next_step_size()  :
          1.0 / time_stepping.get_next_step_size());

      const LinearSolverParameters &prm = parameters.linear_solver_control;

      SolverControl solver_control(prm.n_maximum_iterations,
                                   std::max(prm.relative_tolerance * pressure_rhs.l2_norm(),
                                            prm.absolute_tolerance));

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

      pressure->constraints.distribute(distributed_pressure);

      distributed_pressure.sadd(1.0 / parameters.Re, 1., distributed_old_pressure);
      distributed_pressure += distributed_phi;

      if (flag_normalize_pressure)
        VectorTools::subtract_mean_value(distributed_pressure);

      pressure->solution = distributed_pressure;

      break;
    }
    default:
      Assert(false, ExcNotImplemented());
      break;
  };
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
