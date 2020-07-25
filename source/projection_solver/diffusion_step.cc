#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
diffusion_step(const bool reinit_prec)
{
  /*Extrapolate velocity by a Taylor expansion
    v^{\textrm{k}+1} \approx 2 * v^\textrm{k} - v^{\textrm{k}-1 */

  extrapolated_velocity.equ(1.0 + dt_n / dt_n_minus_1, velocity_n);
  extrapolated_velocity.add(-dt_n / dt_n_minus_1, velocity_n_minus_1);

  /*Define auxiliary pressure
    p^{\#} = p^\textrm{k} + 4/3 * \phi^\textrm{k} 
              - 1/3 * \phi^{\textrm{k}-1} 
    Note: The signs are inverted since p_gradient_matrix is
    defined as negative */

  pressure_tmp.equ(-1., pressure_n);
  pressure_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_minus_1);

  assemble_diffusion_step();

  /* Update for the next time step */
  velocity_n_minus_1 = velocity_n;

  solve_diffusion_step(reinit_prec);
}
template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step()
{
  /* System matrix setup */
  assemble_velocity_advection_matrix();
  if (flag_adpative_time_step)
  {
    velocity_mass_plus_laplace_matrix = 0.;
    velocity_mass_plus_laplace_matrix.add(1.0 / Re, 
                                          velocity_laplace_matrix);
    velocity_mass_plus_laplace_matrix.add((2.0 * dt_n + dt_n_minus_1) /  //
                                          (dt_n * (dt_n + dt_n_minus_1)), 
                                          velocity_mass_matrix);
  }
  velocity_system_matrix.copy_from(velocity_mass_plus_laplace_matrix);
  velocity_system_matrix.add(1., velocity_advection_matrix);

  /* Right hand side setup */
  velocity_rhs = 0.;
  velocity_tmp.equ((dt_n + dt_n_minus_1) / (dt_n * dt_n_minus_1), 
                   velocity_n);
  velocity_tmp.add(- (dt_n * dt_n) / (dt_n * dt_n_minus_1 *             //
                   (dt_n + dt_n_minus_1)), 
                   velocity_n_minus_1);
  velocity_mass_matrix.vmult_add(velocity_rhs, velocity_tmp);
  pressure_gradient_matrix.vmult_add(velocity_rhs, pressure_tmp);

  velocity_constraints.condense(velocity_system_matrix, velocity_rhs);
}

template <int dim>
void
NavierStokesProjection<dim>::
solve_diffusion_step(const bool reinit_prec)
{
  if (reinit_prec)
    diffusion_step_preconditioner.initialize(
                                      velocity_system_matrix,
                                      SparseILU<double>::AdditionalData(
                                        solver_diag_strength, 
                                        solver_off_diagonals));

  SolverControl solver_control(solver_max_iterations, 
                               solver_tolerance * velocity_rhs.l2_norm());
  SolverGMRES<> gmres(solver_control,
                      SolverGMRES<>::AdditionalData(solver_krylov_size));
  gmres.solve(velocity_system_matrix, 
              velocity_n, 
              velocity_rhs, 
              diffusion_step_preconditioner);
  velocity_constraints.distribute(velocity_n);
}
}
// explicit instantiations
template void Step35::NavierStokesProjection<2>::diffusion_step(const bool);
template void Step35::NavierStokesProjection<3>::diffusion_step(const bool);
template void Step35::NavierStokesProjection<2>::assemble_diffusion_step();
template void Step35::NavierStokesProjection<3>::assemble_diffusion_step();
template void Step35::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void Step35::NavierStokesProjection<3>::solve_diffusion_step(const bool);