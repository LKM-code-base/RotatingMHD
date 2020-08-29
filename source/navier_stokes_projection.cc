#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

namespace RMHD
{
template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::ParameterSet   &parameters,
 Entities::VectorEntity<dim>             &velocity,
 Entities::ScalarEntity<dim>             &pressure,
 TimeDiscretization::VSIMEXCoefficients  &VSIMEX,
 TimeDiscretization::VSIMEXMethod        &time_stepping)
:
projection_method(parameters.projection_method),
Re(parameters.Re),
velocity(velocity),
pressure(pressure),
VSIMEX(VSIMEX),
time_stepping(time_stepping),
solver_max_iterations(parameters.solver_max_iterations),
solver_krylov_size(parameters.solver_krylov_size),
solver_off_diagonals(parameters.solver_off_diagonals),
solver_update_preconditioner(parameters.solver_update_preconditioner),
solver_tolerance(parameters.solver_tolerance),
solver_diag_strength(parameters.solver_diag_strength),
flag_adpative_time_step(parameters.flag_adaptive_time_step)
{}

}  // namespace Step35

// explicit instantiations
template class RMHD::NavierStokesProjection<2>;
template class RMHD::NavierStokesProjection<3>;
