#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
double NavierStokesProjection<dim>::
compute_next_time_step()
{
  if (!parameters.time_stepping_parameters.adaptive_time_step)
    return time_stepping.get_next_step_size();

  const QIterated<dim>  quadrature_formula(QTrapez<1>(),
                                           velocity.fe_degree +1);
  FEValues<dim>         fe_values(velocity.fe,
                                  quadrature_formula,
                                  update_values);

  const unsigned int  n_q_points    = quadrature_formula.size();
  std::vector<Tensor<1, dim>>       velocity_values(n_q_points);
  double min_local_timestep         = 1e+10;
  double scaling_parameter          = 1.0;

  const FEValuesExtractors::Vector  velocities(0);
  
  for (const auto &cell : velocity.dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        double max_local_velocity = 1e-10;
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(velocity.solution,
                                                  velocity_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          max_local_velocity =
            std::max(max_local_velocity, velocity_values[q].norm());
        min_local_timestep = std::min(min_local_timestep, 
                                      cell->diameter() / 
                                      max_local_velocity);
      }
  min_local_timestep = 
                Utilities::MPI::min(min_local_timestep, MPI_COMM_WORLD);
  return scaling_parameter * min_local_timestep;
}

template <int dim>
void NavierStokesProjection<dim>::
update_internal_entities()
{
  old_old_phi = old_phi;
  old_phi     = phi;
}

} // namespace RMHD

// explicit instantiations
template double RMHD::NavierStokesProjection<2>::compute_next_time_step();
template double RMHD::NavierStokesProjection<3>::compute_next_time_step();

template void RMHD::NavierStokesProjection<2>::update_internal_entities();
template void RMHD::NavierStokesProjection<3>::update_internal_entities();
