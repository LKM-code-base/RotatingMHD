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
  {
    TrilinosWrappers::MPI::Vector distributed_velocity_n(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_velocity_n_minus_1(velocity_rhs);
    distributed_velocity_n = velocity_n;
    distributed_velocity_n_minus_1 = velocity_n_minus_1;
    /*Extrapolate velocity by a Taylor expansion
      v^{\textrm{k}+1} \approx 2 * v^\textrm{k} - v^{\textrm{k}-1 */
    distributed_velocity_n.sadd(1.0 + dt_n / dt_n_minus_1, 
                                -dt_n / dt_n_minus_1,
                                distributed_velocity_n_minus_1);
    extrapolated_velocity = distributed_velocity_n;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_pressure_n(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_phi_n_minus_1(pressure_rhs);
    TrilinosWrappers::MPI::Vector distributed_phi_n(pressure_rhs);
    distributed_pressure_n = pressure_n;
    distributed_phi_n_minus_1 = phi_n_minus_1;
    distributed_phi_n = phi_n;
    /*Define auxiliary pressure
    p^{\#} = p^\textrm{k} + 4/3 * \phi^\textrm{k} 
                - 1/3 * \phi^{\textrm{k}-1} */
    distributed_pressure_n.sadd(+1.,
                                +4. / 3., 
                                distributed_phi_n);
    distributed_pressure_n.sadd(+1.,
                                 -1. / 3.,
                                distributed_phi_n_minus_1);
    pressure_tmp = distributed_pressure_n;
  }

  {
    TrilinosWrappers::MPI::Vector distributed_velocity_n(velocity_rhs);
    TrilinosWrappers::MPI::Vector distributed_velocity_n_minus_1(velocity_rhs);
    distributed_velocity_n = velocity_n;
    distributed_velocity_n_minus_1 = velocity_n_minus_1;
    /*Define the auxiliary velocity as the sum from the velocities at
    previous time steps weighted accordingly to the VSIMEX method */
    distributed_velocity_n.sadd(- (dt_n + dt_n_minus_1) / (dt_n * dt_n_minus_1),
                                (dt_n * dt_n) / (dt_n * dt_n_minus_1 *
                                (dt_n + dt_n_minus_1)),
                                distributed_velocity_n_minus_1);
    velocity_tmp = distributed_velocity_n;
  }

  /* Assemble linear system */
  assemble_diffusion_step();

  /* Update for the next time step */
  velocity_n_minus_1 = velocity_n;

  /* Solve linear system */
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
  assemble_diffusion_step_rhs();
}

template <int dim>
void
NavierStokesProjection<dim>::
solve_diffusion_step(const bool reinit_prec)
{
  pcout << "Diffusion step" << std::endl;
  TrilinosWrappers::MPI::Vector distributed_velocity_n(velocity_rhs);
  distributed_velocity_n = velocity_n;

  if (reinit_prec)
    diffusion_step_preconditioner.initialize(
                                      velocity_system_matrix/*,
                                      SparseILU<double>::AdditionalData(
                                        solver_diag_strength, 
                                        solver_off_diagonals)*/);

  SolverControl solver_control(solver_max_iterations, 
                               solver_tolerance * velocity_rhs.l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> gmres(solver_control,
                      SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(solver_krylov_size));
  gmres.solve(velocity_system_matrix, 
              distributed_velocity_n, 
              velocity_rhs, 
              diffusion_step_preconditioner);
  velocity_constraints.distribute(distributed_velocity_n);
  velocity_n = distributed_velocity_n;
}
}
// explicit instantiations
template void Step35::NavierStokesProjection<2>::diffusion_step(const bool);
template void Step35::NavierStokesProjection<3>::diffusion_step(const bool);
template void Step35::NavierStokesProjection<2>::assemble_diffusion_step();
template void Step35::NavierStokesProjection<3>::assemble_diffusion_step();
template void Step35::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void Step35::NavierStokesProjection<3>::solve_diffusion_step(const bool);