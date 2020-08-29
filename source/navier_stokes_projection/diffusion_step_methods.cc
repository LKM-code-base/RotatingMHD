#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step()
{
  /* System matrix setup */
  assemble_velocity_advection_matrix();

  static bool initialized(false);
  if (!initialized)
  {
    velocity_mass_plus_laplace_matrix = 0.;
    velocity_mass_plus_laplace_matrix.add(1.0 / Re, 
                                          velocity_laplace_matrix);
    velocity_mass_plus_laplace_matrix.add(VSIMEX.alpha[2], 
                                          velocity_mass_matrix);
    if (!flag_adpative_time_step)
      initialized = true; 
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
  TrilinosWrappers::MPI::Vector distributed_velocity(velocity_rhs);
  distributed_velocity = velocity.solution;

  if (reinit_prec)
    diffusion_step_preconditioner.initialize(
                                      velocity_system_matrix);

  SolverControl solver_control(solver_max_iterations, 
                               solver_tolerance * velocity_rhs.l2_norm());
  SolverGMRES<TrilinosWrappers::MPI::Vector> gmres(
                            solver_control,
                            SolverGMRES<TrilinosWrappers::MPI::Vector>::
                              AdditionalData(solver_krylov_size));
  gmres.solve(velocity_system_matrix, 
              distributed_velocity, 
              velocity_rhs, 
              diffusion_step_preconditioner);
  velocity.constraints.distribute(distributed_velocity);
  velocity.solution = distributed_velocity;
}
}
// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step();
template void RMHD::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_diffusion_step(const bool);