#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/trilinos_solver.h>
namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
solve(const unsigned int step)
{
  diffusion_step((step % solver_update_preconditioner == 0) || 
                 (step == time_stepping.get_order()));
  projection_step((step == time_stepping.get_order()));
  pressure_correction((step == time_stepping.get_order()));
}

template <int dim>
void NavierStokesProjection<dim>::
diffusion_step(const bool reinit_prec)
{
  // In the following scopes we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the sadd()
  // operations.
  {
    TrilinosWrappers::MPI::Vector distributed_old_velocity(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity.old_solution;
    distributed_old_old_velocity  = velocity.old_old_solution;
    distributed_old_velocity.sadd(VSIMEX.phi[1], 
                                  VSIMEX.phi[0],
                                  distributed_old_old_velocity);
    extrapolated_velocity = distributed_old_velocity;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_old_pressure(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_old_old_phi(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_phi(pressure_rhs);
    distributed_old_pressure  = pressure.old_solution;
    distributed_old_old_phi   = old_old_phi;
    distributed_phi           = old_phi;
    distributed_old_pressure.sadd(+1.,
                                  +4. / 3., 
                                  distributed_phi);
    distributed_old_pressure.sadd(+1.,
                                  -1. / 3.,
                                  distributed_old_old_phi);
    pressure_tmp = distributed_old_pressure;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_old_velocity(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_velocity      = velocity.old_solution;
    distributed_old_old_velocity  = velocity.old_old_solution;
    distributed_old_velocity.sadd(VSIMEX.alpha[1],
                              VSIMEX.alpha[0],
                              distributed_old_old_velocity);
    velocity_tmp = distributed_old_velocity;
  }

  /* Assemble linear system */
  assemble_diffusion_step();

  /* Solve linear system */
  solve_diffusion_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::
projection_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_projection_step();

  /* Solve linear system */
  solve_projection_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::
pressure_correction(const bool reinit_prec)
{
  // This boolean will be used later when a proper solver is chosen
  (void)reinit_prec;
  
  switch (projection_method)
    {
      case RunTimeParameters::ProjectionMethod::standard:
        pressure.solution += phi;
        break;
      case RunTimeParameters::ProjectionMethod::rotational:
        static TrilinosWrappers::SolverDirect::AdditionalData data(
                                                          false, 
                                                          "Amesos_Klu");
        static SolverControl solver_control(1, 0);
        // In the following scope we create temporal non ghosted copies
        // of the pertinent vectors to be able to perform the solve()
        // operation.
        {
          TrilinosWrappers::MPI::Vector distributed_pressure(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_old_pressure(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_phi(pressure_rhs);

          distributed_pressure      = pressure.solution;
          distributed_old_pressure  = pressure.old_solution;
          distributed_phi           = phi;

          /* Using a direct solver */
          TrilinosWrappers::SolverDirect solver(solver_control, data);
          solver.solve(pressure_mass_matrix,
                       distributed_pressure, 
                       pressure_rhs);

          /* Using CG */
          /*if (reinit_prec)
            correction_step_preconditioner.initialize(
                                              pressure_mass_matrix);

          SolverControl solver__control(solver_max_iterations, 
                                solver_tolerance * pressure_rhs.l2_norm());
          
          SolverCG<TrilinosWrappers::MPI::Vector>    cg_solver(solver__control);
          cg_solver.solve(pressure_mass_matrix, 
                  distributed_pressure, 
                  pressure_rhs, 
                  correction_step_preconditioner);*/

          pressure.constraints.distribute(distributed_pressure);
          distributed_pressure.sadd(1.0 / Re, 1., distributed_old_pressure);
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
