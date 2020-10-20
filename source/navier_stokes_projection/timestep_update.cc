#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
double NavierStokesProjection<dim>::
compute_next_time_step()
{
  if (!parameters.time_stepping_parameters.adaptive_time_stepping)
    return time_stepping.get_next_step_size();

  double max_cfl_number = 1.0;

  return max_cfl_number / get_cfl_number() * 
         time_stepping.get_next_step_size();
}

template <int dim>
double NavierStokesProjection<dim>::
get_cfl_number()
{
  const QIterated<dim>  quadrature_formula(QTrapez<1>(),
                                           velocity.fe_degree +1);
  FEValues<dim>         fe_values(velocity.fe,
                                  quadrature_formula,
                                  update_values);

  const unsigned int  n_q_points    = quadrature_formula.size();
  std::vector<Tensor<1, dim>>       velocity_values(n_q_points);
  double max_cfl_number             = 1e-10;

  const FEValuesExtractors::Vector  velocities(0);
  
  for (const auto &cell : velocity.dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        double max_local_velocity = 1e-10;
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(velocity.old_solution,
                                                  velocity_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          max_local_velocity =
            std::max(max_local_velocity, velocity_values[q].norm());
        max_cfl_number = std::max(max_cfl_number, 
                                  time_stepping.get_next_step_size() *
                                  max_local_velocity /
                                  cell->diameter());
      }

  max_cfl_number = 
                Utilities::MPI::max(max_cfl_number, MPI_COMM_WORLD);

  return max_cfl_number;
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
template double RMHD::NavierStokesProjection<2>::get_cfl_number();
template double RMHD::NavierStokesProjection<3>::get_cfl_number();

template double RMHD::NavierStokesProjection<2>::compute_next_time_step();
template double RMHD::NavierStokesProjection<3>::compute_next_time_step();

template void RMHD::NavierStokesProjection<2>::update_internal_entities();
template void RMHD::NavierStokesProjection<3>::update_internal_entities();
