#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
assemble_pressure_gradient_matrix()
{
  PressureGradientAssembly::MappingData<dim> per_task_data(
                                            velocity_fe.dofs_per_cell,
                                            pressure_fe.dofs_per_cell);
  PressureGradientAssembly::LocalCellData<dim> scratch_data(
                                            velocity_fe,
                                            pressure_fe,
                                            velocity_quadrature_formula,
                                            update_gradients | 
                                            update_JxW_values,
                                            update_values);

  WorkStream::run(
    IteratorPair(IteratorTuple(velocity_dof_handler.begin_active(),
                                pressure_dof_handler.begin_active())),
    IteratorPair(IteratorTuple(velocity_dof_handler.end(),
                                pressure_dof_handler.end())),
    *this,
    &NavierStokesProjection<dim>::assemble_local_pressure_gradient_matrix,
    &NavierStokesProjection<dim>::copy_loca_to_global_pressure_gradient_matrix,
    scratch_data,
    per_task_data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_pressure_gradient_matrix(
                const IteratorPair                            &SI,
                PressureGradientAssembly::LocalCellData<dim>  &scratch,
                PressureGradientAssembly::MappingData<dim>    &data)
{
  scratch.velocity_fe_values.reinit(std::get<0>(*SI));
  scratch.pressure_fe_values.reinit(std::get<1>(*SI));

  std::get<0>(*SI)->get_dof_indices(data.local_velocity_dof_indices);
  std::get<1>(*SI)->get_dof_indices(data.local_pressure_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);

  data.local_matrix = 0.;
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < data.velocity_dofs_per_cell; ++i)
      scratch.div_phi_velocity[i] = 
                  scratch.velocity_fe_values[velocity].divergence(i, q);
    for (unsigned int i = 0; i < data.pressure_dofs_per_cell; ++i)
      scratch.phi_pressure[i] = 
                  scratch.pressure_fe_values.shape_value(i, q);

    for (unsigned int i = 0; i < data.velocity_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < data.pressure_dofs_per_cell; ++j)
        data.local_matrix(i, j) += -scratch.velocity_fe_values.JxW(q) *
                                    scratch.phi_pressure[j] *
                                    scratch.div_phi_velocity[i];
  }
}


template <int dim>
void NavierStokesProjection<dim>::
copy_loca_to_global_pressure_gradient_matrix(
  const PressureGradientAssembly::MappingData<dim> &data)
{
  for (unsigned int i = 0; i < data.velocity_dofs_per_cell; ++i)
    for (unsigned int j = 0; j < data.pressure_dofs_per_cell; ++j)
      pressure_gradient_matrix.add(data.local_velocity_dof_indices[i],
                                   data.local_pressure_dof_indices[j],
                                   data.local_matrix(i, j));
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::assemble_pressure_gradient_matrix();
template void Step35::NavierStokesProjection<3>::assemble_pressure_gradient_matrix();
template void Step35::NavierStokesProjection<2>::assemble_local_pressure_gradient_matrix(
    const IteratorPair &,
    Step35::PressureGradientAssembly::LocalCellData<2>  &,
    Step35::PressureGradientAssembly::MappingData<2>    &);
template void Step35::NavierStokesProjection<3>::assemble_local_pressure_gradient_matrix(
    const IteratorPair &, 
    Step35::PressureGradientAssembly::LocalCellData<3>  &,
    Step35::PressureGradientAssembly::MappingData<3>    &);
template void Step35::NavierStokesProjection<2>::copy_loca_to_global_pressure_gradient_matrix(
    const Step35::PressureGradientAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_loca_to_global_pressure_gradient_matrix(
    const Step35::PressureGradientAssembly::MappingData<3> &);