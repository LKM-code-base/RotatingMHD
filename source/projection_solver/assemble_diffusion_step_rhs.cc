#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step_rhs()
{
  velocity_rhs  = 0.;

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           VelocityRightHandSideAssembly::LocalCellData<dim>    &scratch,
           VelocityRightHandSideAssembly::MappingData<dim>      &data)
    {
      this->assemble_local_diffusion_step_rhs(cell, 
                                              scratch,
                                              data);
    };
  
  auto copier =
    [this](const VelocityRightHandSideAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_diffusion_step_rhs(data);
    };

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             velocity_dof_handler.begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             velocity_dof_handler.end()),
                  worker,
                  copier,
                  VelocityRightHandSideAssembly::LocalCellData<dim>(
                                          velocity_fe,
                                          pressure_fe,
                                          velocity_quadrature_formula,
                                          update_values |
                                          update_gradients |
                                          update_JxW_values,
                                          update_values),
                  VelocityRightHandSideAssembly::MappingData<dim>(
                                          velocity_fe.dofs_per_cell));
  velocity_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_diffusion_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator  &cell, 
  VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
  VelocityRightHandSideAssembly::MappingData<dim>       &data)
{
  data.local_diffusion_step_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  scratch.velocity_fe_values.reinit(cell);
  
  typename DoFHandler<dim>::active_cell_iterator pressure_cell(
                                        &triangulation, 
                                        cell->level(), 
                                        cell->index(), 
                                        &pressure_dof_handler);
  scratch.pressure_fe_values.reinit(pressure_cell);
  
  cell->get_dof_indices(data.local_velocity_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);

  scratch.velocity_fe_values[velocity].get_function_values(
                                          velocity_tmp,
                                          scratch.velocity_tmp_values);
  scratch.pressure_fe_values.get_function_values(
                                          pressure_tmp,
                                          scratch.pressure_tmp_values);

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = 
                  scratch.velocity_fe_values[velocity].value(i, q);
      scratch.div_phi_velocity[i] =
                  scratch.velocity_fe_values[velocity].divergence(i, q);
    }
    
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      data.local_diffusion_step_rhs(i) +=
                                  scratch.velocity_fe_values.JxW(q) * (
                                  - scratch.velocity_tmp_values[q] *
                                  scratch.phi_velocity[i]
                                  +
                                  scratch.pressure_tmp_values[q] *
                                  scratch.div_phi_velocity[i]);
      
      if (velocity_constraints.is_inhomogeneously_constrained(
        data.local_velocity_dof_indices[i]))
      {
        scratch.velocity_fe_values[velocity].get_function_values(
                                  extrapolated_velocity, 
                                  scratch.extrapolated_velocity_values);
        scratch.velocity_fe_values[velocity].get_function_divergences(
                            extrapolated_velocity, 
                            scratch.extrapolated_velocity_divergences);

        for (unsigned int k = 0; k < scratch.velocity_dofs_per_cell; ++k)
          scratch.grad_phi_velocity[k] = 
                    scratch.velocity_fe_values[velocity].gradient(k, q);

        for (unsigned int j = 0; j < scratch.velocity_dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) += (
                            (2.0 * dt_n + dt_n_minus_1) /
                            (dt_n * (dt_n + dt_n_minus_1)) *
                            scratch.phi_velocity[j] *
                            scratch.phi_velocity[i]
                            +
                            1.0 / Re *
                            scalar_product(
                              scratch.grad_phi_velocity[j],
                              scratch.grad_phi_velocity[i])
                            +
                            scratch.phi_velocity[j] *
                            scratch.grad_phi_velocity[i] *  
                            scratch.extrapolated_velocity_values[q]            
                            +                                    
                            0.5 *                                
                            scratch.extrapolated_velocity_divergences[q] *            
                            scratch.phi_velocity[j] * 
                            scratch.phi_velocity[i])  
                            * scratch.velocity_fe_values.JxW(q);
      }
    }
  }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_diffusion_step_rhs(
  const VelocityRightHandSideAssembly::MappingData<dim>  &data)
{
  velocity_constraints.distribute_local_to_global(
                                data.local_diffusion_step_rhs,
                                data.local_velocity_dof_indices,
                                velocity_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace Step35

// Explicit instantiations

template void Step35::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void Step35::NavierStokesProjection<3>::assemble_diffusion_step_rhs();
template void Step35::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs(
    const typename DoFHandler<2>::active_cell_iterator          &,
    Step35::VelocityRightHandSideAssembly::LocalCellData<2>     &,
    Step35::VelocityRightHandSideAssembly::MappingData<2>       &);
template void Step35::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs(
    const typename DoFHandler<3>::active_cell_iterator          &,
    Step35::VelocityRightHandSideAssembly::LocalCellData<3>     &,
    Step35::VelocityRightHandSideAssembly::MappingData<3>       &);
template void Step35::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs(
    const Step35::VelocityRightHandSideAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs(
    const Step35::VelocityRightHandSideAssembly::MappingData<3> &);