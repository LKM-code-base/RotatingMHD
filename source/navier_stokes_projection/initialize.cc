#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/trilinos_solver.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
initialize()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Initializing the solver..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Initialize ");

  flag_initializing = true;

  if (velocity->solution.size() != velocity_rhs.size())
    setup();

  if (body_force_ptr != nullptr)
    body_force_ptr->set_time(time_stepping.get_start_time());

  poisson_prestep();

  if (time_stepping.get_order() == 1)
  {    
    flag_initializing = false;
    return;
  }

  if (body_force_ptr != nullptr)
    body_force_ptr->advance_time(time_stepping.get_next_step_size());

  diffusion_prestep();
  projection_prestep();
  pressure_correction_prestep();
  
  flag_initializing = false;
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
  /* Assemble linear system */
  assemble_diffusion_prestep();
  /* Solve linear system */
  solve_diffusion_step(true);
  velocity->old_solution = velocity->solution;
}

template <int dim>
void NavierStokesProjection<dim>::
projection_prestep()
{
  /* Assemble linear system */
  assemble_projection_step();
  /* Solve linear system */
  solve_projection_step(true);

  phi->old_solution = phi->solution;
}

template <int dim>
void NavierStokesProjection<dim>::
pressure_correction_prestep()
{
  pressure->old_solution = pressure->old_old_solution;
  pressure->old_solution += phi->old_solution;
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