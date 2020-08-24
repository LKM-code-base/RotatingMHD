#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
update_internal_entities()
{
  old_old_phi = old_phi;
  old_phi     = phi;
}

} // namespace RMHD

template void RMHD::NavierStokesProjection<2>::update_internal_entities();
template void RMHD::NavierStokesProjection<3>::update_internal_entities();