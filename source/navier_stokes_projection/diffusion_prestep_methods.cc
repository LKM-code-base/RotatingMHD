#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/solver_gmres.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_prestep()
{
  /* System matrix setup */
  assemble_velocity_advection_matrix();

  velocity_mass_plus_laplace_matrix = 0.;
  velocity_mass_plus_laplace_matrix.add(1.0 / Re, 
                                        velocity_laplace_matrix);
  velocity_mass_plus_laplace_matrix.add(1.0 / 
                                        time_stepping.get_next_step_size(), 
                                        velocity_mass_matrix);
  velocity_system_matrix.copy_from(velocity_mass_plus_laplace_matrix);
  velocity_system_matrix.add(1., velocity_advection_matrix);

  /* Right hand side setup */
  assemble_diffusion_step_rhs();
}

} // namespace RMHD

template void RMHD::NavierStokesProjection<2>::assemble_diffusion_prestep();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_prestep();