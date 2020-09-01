#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/trilinos_solver.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
initialize()
{
  poisson_prestep();
  diffusion_prestep();
  projection_prestep();
  pressure_correction_prestep();
}

template <int dim>
void NavierStokesProjection<dim>::
poisson_prestep()
{
  /* Assemble linear system */
  assemble_poisson_prestep();
  /* Solve linear system */
  solve_poisson_prestep();
}

template <int dim>
void NavierStokesProjection<dim>::
diffusion_prestep()
{
  /* In the diffusion prestep the extrapolated velocity reduces to
     the velocity at t = t_0 */
  {
    TrilinosWrappers::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_old_velocity  = velocity.old_old_solution;
    extrapolated_velocity = distributed_old_old_velocity;
  }
  /* The temporary pressure reduces to the pressure at t = t_0 */
  {
    TrilinosWrappers::MPI::Vector distributed_old_old_pressure(pressure_rhs);
    distributed_old_old_pressure  = pressure.old_old_solution;
    pressure_tmp = distributed_old_old_pressure;
  }
  /* The temporary velocity reduces to that of a first order IMEX
     method */
  {
    TrilinosWrappers::MPI::Vector distributed_old_old_velocity(velocity_rhs);
    distributed_old_old_velocity  = velocity.old_old_solution;
    distributed_old_old_velocity  *= -1.0 / time_stepping.get_next_step_size();
    velocity_tmp                  = distributed_old_old_velocity;
  }

  /* Assemble linear system */
  assemble_diffusion_prestep();
  /* Solve linear system */
  solve_diffusion_step(true);
  velocity.old_solution = velocity.solution;
}

template <int dim>
void NavierStokesProjection<dim>::
projection_prestep()
{
  /* Assemble linear system */
  assemble_projection_step();
  /* Solve linear system */
  solve_projection_step(true);
  old_phi = phi;
}

template <int dim>
void NavierStokesProjection<dim>::
pressure_correction_prestep()
{
  pressure.old_solution = pressure.old_old_solution;
  pressure.old_solution += phi;
}

} // namespace RMHD

// explicit instantiations

template void RMHD::NavierStokesProjection<2>::initialize();
template void RMHD::NavierStokesProjection<3>::initialize();
template void RMHD::NavierStokesProjection<2>::poisson_prestep();
template void RMHD::NavierStokesProjection<3>::poisson_prestep();
template void RMHD::NavierStokesProjection<2>::diffusion_prestep();
template void RMHD::NavierStokesProjection<3>::diffusion_prestep();
template void RMHD::NavierStokesProjection<2>::projection_prestep();
template void RMHD::NavierStokesProjection<3>::projection_prestep();
template void RMHD::NavierStokesProjection<2>::pressure_correction_prestep();
template void RMHD::NavierStokesProjection<3>::pressure_correction_prestep();