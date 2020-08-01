#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step_rhs()
{
  velocity_rhs  = 0.;
  DiffusionStepRightHandSideAssembly::MappingData<dim> 
                                        data(velocity_fe.dofs_per_cell);
  DiffusionStepRightHandSideAssembly::LocalCellData<dim>
                                        scratch(
                                          velocity_fe,
                                          pressure_fe,
                                          velocity_quadrature_formula,
                                          update_values |
                                          update_gradients |
                                          update_JxW_values,
                                          update_values);
  WorkStream::run(
    IteratorPair(IteratorTuple(velocity_dof_handler.begin_active(),
                                pressure_dof_handler.begin_active())),
    IteratorPair(IteratorTuple(velocity_dof_handler.end(),
                                pressure_dof_handler.end())),
    *this,
    &NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs,
    &NavierStokesProjection<dim>::copy_local_to_global_diffusion_step_rhs,
    scratch,
    data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_diffusion_step_rhs(
  const IteratorPair                                          &SI,
  DiffusionStepRightHandSideAssembly::LocalCellData<dim>      &scratch,
  DiffusionStepRightHandSideAssembly::MappingData<dim>        &data)
{
  data.local_diffusion_step_rhs = 0.;

  scratch.velocity_fe_values.reinit(std::get<0>(*SI));
  scratch.pressure_fe_values.reinit(std::get<1>(*SI));
  
  std::get<0>(*SI)->get_dof_indices(data.local_velocity_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);

  scratch.velocity_fe_values[velocity].get_function_values(
                                          velocity_tmp,
                                          scratch.velocity_tmp_values);
  scratch.pressure_fe_values.get_function_values(
                                          pressure_tmp,
                                          scratch.pressure_tmp_values);
                                          
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
      data.local_diffusion_step_rhs(i) +=
                scratch.velocity_fe_values.JxW(q) * (
                - scratch.velocity_tmp_values[q] *
                scratch.velocity_fe_values[velocity].value(i, q)
                +
                scratch.pressure_tmp_values[q] *
                scratch.velocity_fe_values[velocity].divergence(i, q));
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_diffusion_step_rhs(
  const DiffusionStepRightHandSideAssembly::MappingData<dim>  &data)
{
  for (unsigned int i = 0; i < velocity_fe.dofs_per_cell; ++i)
    velocity_rhs(data.local_velocity_dof_indices[i]) +=
      data.local_diffusion_step_rhs(i);
}

} // namespace Step35

// Explicit instantiations

template void Step35::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void Step35::NavierStokesProjection<3>::assemble_diffusion_step_rhs();
template void Step35::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs(
    const IteratorPair                                                   &,
    Step35::DiffusionStepRightHandSideAssembly::LocalCellData<2>         &,
    Step35::DiffusionStepRightHandSideAssembly::MappingData<2>           &);
template void Step35::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs(
    const IteratorPair                                                   &,
    Step35::DiffusionStepRightHandSideAssembly::LocalCellData<3>         &,
    Step35::DiffusionStepRightHandSideAssembly::MappingData<3>           &);
template void Step35::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs(
    const Step35::DiffusionStepRightHandSideAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs(
    const Step35::DiffusionStepRightHandSideAssembly::MappingData<3> &);