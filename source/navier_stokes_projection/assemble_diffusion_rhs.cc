#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
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

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.end()),
   worker,
   copier,
   VelocityRightHandSideAssembly::LocalCellData<dim>(velocity.fe,
                                                     pressure.fe,
                                                     velocity.quadrature_formula,
                                                     update_values|
                                                     update_gradients|
                                                     update_JxW_values,
                                                     update_values),
   VelocityRightHandSideAssembly::MappingData<dim>(velocity.fe.dofs_per_cell));

  velocity_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
 VelocityRightHandSideAssembly::MappingData<dim>       &data)
{
  // reset local matrix and vector
  data.local_diffusion_step_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // prepare velocity part
  const FEValuesExtractors::Vector  velocities(0);

  scratch.velocity_fe_values.reinit(cell);
  
  cell->get_dof_indices(data.local_velocity_dof_indices);

  scratch.velocity_fe_values[velocities].get_function_values
  (velocity_tmp,
   scratch.velocity_tmp_values);

  // prepare pressure part
  typename DoFHandler<dim>::active_cell_iterator
  pressure_cell(&velocity.dof_handler.get_triangulation(),
                 cell->level(),
                 cell->index(),
                &pressure.dof_handler);

  scratch.pressure_fe_values.reinit(pressure_cell);
  
  scratch.pressure_fe_values.get_function_values(
                                          pressure_tmp,
                                          scratch.pressure_tmp_values);

  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = 
                  scratch.velocity_fe_values[velocities].value(i, q);
      scratch.div_phi_velocity[i] =
                  scratch.velocity_fe_values[velocities].divergence(i, q);
    }
    
    // loop over local dofs
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      data.local_diffusion_step_rhs(i) +=
                                  scratch.velocity_fe_values.JxW(q) * (
                                  - scratch.velocity_tmp_values[q] *
                                  scratch.phi_velocity[i]
                                  +
                                  scratch.pressure_tmp_values[q] *
                                  scratch.div_phi_velocity[i]);

      // assemble matrix for inhomogeneous boundary conditions
      if (velocity.constraints.is_inhomogeneously_constrained(
          data.local_velocity_dof_indices[i]))
      {
        scratch.velocity_fe_values[velocities].get_function_values(
                                  extrapolated_velocity, 
                                  scratch.extrapolated_velocity_values);
        scratch.velocity_fe_values[velocities].get_function_divergences(
                            extrapolated_velocity, 
                            scratch.extrapolated_velocity_divergences);

        for (unsigned int k = 0; k < scratch.velocity_dofs_per_cell; ++k)
          scratch.grad_phi_velocity[k] = 
                    scratch.velocity_fe_values[velocities].gradient(k, q);

        for (unsigned int j = 0; j < scratch.velocity_dofs_per_cell; ++j)
          /*
           * Do we need the inline if at all?
           */
          data.local_matrix_for_inhomogeneous_bc(j, i) += (
                            ((time_stepping.get_step_number() > 1) ? 
                              time_stepping.get_alpha()[2] / time_stepping.get_next_time():
                              (1.0 / time_stepping.get_next_step_size())) *
                            scratch.phi_velocity[j] *
                            scratch.phi_velocity[i]
                            +
                            1.0 / parameters.Re *
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

    } // loop over local dofs
  } // loop over quadrature points
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_diffusion_step_rhs
(const VelocityRightHandSideAssembly::MappingData<dim>  &data)
{
  velocity.constraints.distribute_local_to_global(
                                data.local_diffusion_step_rhs,
                                data.local_velocity_dof_indices,
                                velocity_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace Step35

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<2>::active_cell_iterator     &,
 RMHD::VelocityRightHandSideAssembly::LocalCellData<2>  &,
 RMHD::VelocityRightHandSideAssembly::MappingData<2>    &);
template void RMHD::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<3>::active_cell_iterator     &,
 RMHD::VelocityRightHandSideAssembly::LocalCellData<3>  &,
 RMHD::VelocityRightHandSideAssembly::MappingData<3>    &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs
(const RMHD::VelocityRightHandSideAssembly::MappingData<2>  &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs
(const RMHD::VelocityRightHandSideAssembly::MappingData<3>  &);
