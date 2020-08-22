#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/trilinos_solver.h>
namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
solve(const unsigned int step)
{
  diffusion_step((step % solver_update_preconditioner == 0) || 
                 (step == 2));
  projection_step((step == 2));
  pressure_correction((step == 2));
}

template <int dim>
void NavierStokesProjection<dim>::
diffusion_step(const bool reinit_prec)
{
  {
    TrilinosWrappers::MPI::Vector distributed_velocity_n(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_velocity_n_minus_1(velocity_rhs);
    distributed_velocity_n = velocity.solution_n;
    distributed_velocity_n_minus_1 = velocity.solution_n_minus_1;
    /*Extrapolate velocity by a Taylor expansion
      v^{\textrm{k}+1} \approx 2 * v^\textrm{k} - v^{\textrm{k}-1 */
    /* The VSIMEXMethod class considers a variable time steps and 
       modifies the weights accordingly with the phi parameters */
    distributed_velocity_n.sadd(VSIMEX.phi[1], 
                                VSIMEX.phi[0],
                                distributed_velocity_n_minus_1);
    extrapolated_velocity = distributed_velocity_n;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_pressure_n(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_old_phi(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_phi(pressure_rhs);
    distributed_pressure_n = pressure.solution_n;
    distributed_old_phi = old_phi;
    distributed_phi = phi;
    /*Define auxiliary pressure
    p^{\#} = p^\textrm{k} + 4/3 * \phi^\textrm{k} 
                - 1/3 * \phi^{\textrm{k}-1} */
    distributed_pressure_n.sadd(+1.,
                                +4. / 3., 
                                distributed_phi);
    distributed_pressure_n.sadd(+1.,
                                 -1. / 3.,
                                distributed_old_phi);
    pressure_tmp = distributed_pressure_n;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_velocity_n(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_velocity_n_minus_1(velocity_rhs);
    distributed_velocity_n = velocity.solution_n;
    distributed_velocity_n_minus_1 = velocity.solution_n_minus_1;
    /*Define the auxiliary velocity as the weighted sum from the 
      velocities product of the VSIMEX method time discretization that 
      belong to the right hand side*/
    distributed_velocity_n.sadd(VSIMEX.alpha[1],
                                VSIMEX.alpha[0],
                                distributed_velocity_n_minus_1);
    velocity_tmp = distributed_velocity_n;
  }

  /* Assemble linear system */
  assemble_diffusion_step();

  /* Update for the next time step */
  velocity.solution_n_minus_1 = velocity.solution_n;

  /* Solve linear system */
  solve_diffusion_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::
projection_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_projection_step();

  /* Update for the next time step */
  old_phi = phi;

  /* Solve linear system */
  solve_projection_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::
pressure_correction(const bool reinit_prec)
{
  // This boolean will be used later when a proper solver is chosen
  (void)reinit_prec;
  /* Update for the next time step */
  pressure.solution_n_minus_1 = pressure.solution_n;
  switch (projection_method)
    {
      case RunTimeParameters::ProjectionMethod::standard:
        pressure.solution_n += phi;
        break;
      case RunTimeParameters::ProjectionMethod::rotational:
        static TrilinosWrappers::SolverDirect::AdditionalData data(
                                                          false, 
                                                          "Amesos_Klu");
        static SolverControl solver_control(1, 0);
        {
          TrilinosWrappers::MPI::Vector distributed_pressure_n(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_pressure_n_minus_1(pressure_rhs);
          TrilinosWrappers::MPI::Vector distributed_phi(pressure_rhs);

          distributed_pressure_n = pressure.solution_n;
          distributed_pressure_n_minus_1 = pressure.solution_n_minus_1;
          distributed_phi = phi;

          /* Using a direct solver */
          TrilinosWrappers::SolverDirect solver(solver_control, data);
          solver.solve(pressure_mass_matrix,
                       distributed_pressure_n, 
                       pressure_rhs);

          /* Using CG */
          /*if (reinit_prec)
            correction_step_preconditioner.initialize(
                                              pressure_mass_matrix);

          SolverControl solver__control(solver_max_iterations, 
                                solver_tolerance * pressure_rhs.l2_norm());
          
          SolverCG<TrilinosWrappers::MPI::Vector>    cg_solver(solver__control);
          cg_solver.solve(pressure_mass_matrix, 
                  distributed_pressure_n, 
                  pressure_rhs, 
                  correction_step_preconditioner);*/

          pressure.constraints.distribute(distributed_pressure_n);
          distributed_pressure_n.sadd(1.0 / Re, 1., distributed_pressure_n_minus_1);
          distributed_pressure_n += distributed_phi;
          pressure.solution_n = distributed_pressure_n;
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
