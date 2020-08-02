#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::assemble_velocity_advection_matrix()
{
  velocity_advection_matrix = 0.;
  AdvectionAssembly::MappingData<dim> data(velocity_fe.dofs_per_cell);
  AdvectionAssembly::LocalCellData<dim> scratch(
                                            velocity_fe,
                                            velocity_quadrature_formula,
                                            update_values | 
                                            update_JxW_values |
                                            update_gradients);
  WorkStream::run(
    velocity_dof_handler.begin_active(),
    velocity_dof_handler.end(),
    *this,
    &NavierStokesProjection<dim>::assemble_local_velocity_advection_matrix,
    &NavierStokesProjection<dim>::copy_local_to_global_velocity_advection_matrix,
    scratch,
    data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_velocity_advection_matrix(
  const typename DoFHandler<dim>::active_cell_iterator  &cell,
  AdvectionAssembly::LocalCellData<dim>                 &scratch,
  AdvectionAssembly::MappingData<dim>                   &data)
{
  scratch.fe_values.reinit(cell);
  cell->get_dof_indices(data.local_dof_indices);
  const FEValuesExtractors::Vector      velocity(0);

  scratch.fe_values[velocity].get_function_values(
                                  extrapolated_velocity, 
                                  scratch.extrapolated_velocity_values);
  scratch.fe_values[velocity].get_function_divergences(
                            extrapolated_velocity, 
                            scratch.extrapolated_velocity_divergences);

  data.local_matrix = 0.;
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocity].value(i,q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocity].gradient(i,q);
    }
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        data.local_matrix(i, j) += ( 
                            scratch.phi_velocity[i] *
                            scratch.grad_phi_velocity[j] *  
                            scratch.extrapolated_velocity_values[q]            
                            +                                    
                            0.5 *                                
                            scratch.extrapolated_velocity_divergences[q] *            
                            scratch.phi_velocity[i] * 
                            scratch.phi_velocity[j])  
                            * scratch.fe_values.JxW(q);
  }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_velocity_advection_matrix(
  const AdvectionAssembly::MappingData<dim> &data)
{
  velocity_constraints.distribute_local_to_global(
                                      data.local_matrix,
                                      data.local_dof_indices,
                                      velocity_advection_matrix);
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::assemble_velocity_advection_matrix();
template void Step35::NavierStokesProjection<3>::assemble_velocity_advection_matrix();
template void Step35::NavierStokesProjection<2>::assemble_local_velocity_advection_matrix(
    const typename DoFHandler<2>::active_cell_iterator  &,
    Step35::AdvectionAssembly::LocalCellData<2>         &,
    Step35::AdvectionAssembly::MappingData<2>           &);
template void Step35::NavierStokesProjection<3>::assemble_local_velocity_advection_matrix(
    const typename DoFHandler<3>::active_cell_iterator  &,
    Step35::AdvectionAssembly::LocalCellData<3>         &,
    Step35::AdvectionAssembly::MappingData<3>           &);
template void Step35::NavierStokesProjection<2>::copy_local_to_global_velocity_advection_matrix(
    const Step35::AdvectionAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_velocity_advection_matrix(
    const Step35::AdvectionAssembly::MappingData<3> &);