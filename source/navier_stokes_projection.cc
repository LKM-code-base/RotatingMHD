#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

  template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::ParameterSet   &parameters,
 Entities::VectorEntity<dim>             &velocity,
 Entities::ScalarEntity<dim>             &pressure,
 TimeDiscretization::VSIMEXMethod        &time_stepping)
:
parameters(parameters),
velocity(velocity),
pressure(pressure),
time_stepping(time_stepping),
flag_diffusion_matrix_assembled(false),
flag_initializing(false),
flag_normalize_pressure(false)
{
  body_force_ptr = nullptr;
}

}  // namespace RMHD

// explicit instantiations
template class RMHD::NavierStokesProjection<2>;
template class RMHD::NavierStokesProjection<3>;

