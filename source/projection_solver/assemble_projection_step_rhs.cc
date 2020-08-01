#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step_rhs()
{
  pressure_rhs = 0.;
  ProjectionStepRightHandSideAssembly::MappingData<dim> 
                                        data(pressure_fe.dofs_per_cell);
  ProjectionStepRightHandSideAssembly::LocalCellData<dim>
                                        scratch(
                                          velocity_fe,
                                          pressure_fe,
                                          pressure_quadrature_formula,
                                          update_values |
                                          update_gradients,
                                          update_JxW_values |
                                          update_values);
  
  WorkStream::run(
    IteratorPair(IteratorTuple(velocity_dof_handler.begin_active(),
                                pressure_dof_handler.begin_active())),
    IteratorPair(IteratorTuple(velocity_dof_handler.end(),
                                pressure_dof_handler.end())),
    *this,
    &NavierStokesProjection<dim>::assemble_local_projection_step_rhs,
    &NavierStokesProjection<dim>::copy_local_to_global_projection_step_rhs,
    scratch,
    data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_projection_step_rhs(
  const IteratorPair                                           &SI,
  ProjectionStepRightHandSideAssembly::LocalCellData<dim>      &scratch,
  ProjectionStepRightHandSideAssembly::MappingData<dim>        &data)
{
  data.local_projection_step_rhs = 0.;

  scratch.velocity_fe_values.reinit(std::get<0>(*SI));
  scratch.pressure_fe_values.reinit(std::get<1>(*SI));
  
  std::get<1>(*SI)->get_dof_indices(data.local_pressure_dof_indices);
  
  const FEValuesExtractors::Vector  velocity(0);

  scratch.velocity_fe_values[velocity].get_function_divergences(
                              velocity_n,
                              scratch.velocity_n_divergence_values);

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      data.local_projection_step_rhs(i) += 
                          -1.0 *
                          scratch.pressure_fe_values.JxW(q) *
                          scratch.velocity_n_divergence_values[q] *
                          scratch.pressure_fe_values.shape_value(i, q);
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_projection_step_rhs(
  const ProjectionStepRightHandSideAssembly::MappingData<dim>  &data)
{
  for (unsigned int i = 0; i < pressure_fe.dofs_per_cell; ++i)
    pressure_rhs(data.local_pressure_dof_indices[i]) +=
      data.local_projection_step_rhs(i);
}

} // namespace Step35

// Explicit instantiations

template void Step35::NavierStokesProjection<2>::assemble_projection_step_rhs();
template void Step35::NavierStokesProjection<3>::assemble_projection_step_rhs();
template void Step35::NavierStokesProjection<2>::assemble_local_projection_step_rhs(
    const IteratorPair                                                    &,
    Step35::ProjectionStepRightHandSideAssembly::LocalCellData<2>         &,
    Step35::ProjectionStepRightHandSideAssembly::MappingData<2>           &);
template void Step35::NavierStokesProjection<3>::assemble_local_projection_step_rhs(
    const IteratorPair                                                    &,
    Step35::ProjectionStepRightHandSideAssembly::LocalCellData<3>         &,
    Step35::ProjectionStepRightHandSideAssembly::MappingData<3>           &);
template void Step35::NavierStokesProjection<2>::copy_local_to_global_projection_step_rhs(
    const Step35::ProjectionStepRightHandSideAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_projection_step_rhs(
    const Step35::ProjectionStepRightHandSideAssembly::MappingData<3> &);