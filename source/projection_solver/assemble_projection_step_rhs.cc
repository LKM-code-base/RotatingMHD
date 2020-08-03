#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step_rhs()
{
  pressure_rhs = 0.;
  PressureRightHandSideAssembly::MappingData<dim> 
                                        data(pressure_fe.dofs_per_cell);
  PressureRightHandSideAssembly::LocalCellData<dim>
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
  const IteratorPair                                  &SI,
  PressureRightHandSideAssembly::LocalCellData<dim>   &scratch,
  PressureRightHandSideAssembly::MappingData<dim>     &data)
{
  data.local_projection_step_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  scratch.velocity_fe_values.reinit(std::get<0>(*SI));
  scratch.pressure_fe_values.reinit(std::get<1>(*SI));
  
  std::get<1>(*SI)->get_dof_indices(data.local_pressure_dof_indices);
  
  const FEValuesExtractors::Vector  velocity(0);

  scratch.velocity_fe_values[velocity].get_function_divergences(
                              velocity_n,
                              scratch.velocity_n_divergence_values);

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      scratch.phi_pressure[i] = 
                          scratch.pressure_fe_values.shape_value(i, q);

    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
    {
      data.local_projection_step_rhs(i) += 
                          -1.0 *
                          scratch.pressure_fe_values.JxW(q) *
                          scratch.velocity_n_divergence_values[q] *
                          scratch.phi_pressure[i];
      if (pressure_constraints.is_inhomogeneously_constrained(
        data.local_pressure_dof_indices[i]))
      {
        for (unsigned int k = 0; k < scratch.pressure_dofs_per_cell; ++k)
          scratch.grad_phi_pressure[k] =
                          scratch.pressure_fe_values.shape_grad(k, q);

        for (unsigned int j = 0; j < scratch.pressure_dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) +=
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.grad_phi_pressure[i] *
                                    scratch.grad_phi_pressure[j];
      }
    } 
  }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_projection_step_rhs(
  const PressureRightHandSideAssembly::MappingData<dim>  &data)
{
  pressure_constraints.distribute_local_to_global(
                                data.local_projection_step_rhs,
                                data.local_pressure_dof_indices,
                                pressure_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace Step35

// Explicit instantiations

template void Step35::NavierStokesProjection<2>::assemble_projection_step_rhs();
template void Step35::NavierStokesProjection<3>::assemble_projection_step_rhs();
template void Step35::NavierStokesProjection<2>::assemble_local_projection_step_rhs(
    const IteratorPair                                          &,
    Step35::PressureRightHandSideAssembly::LocalCellData<2>     &,
    Step35::PressureRightHandSideAssembly::MappingData<2>       &);
template void Step35::NavierStokesProjection<3>::assemble_local_projection_step_rhs(
    const IteratorPair                                          &,
    Step35::PressureRightHandSideAssembly::LocalCellData<3>     &,
    Step35::PressureRightHandSideAssembly::MappingData<3>       &);
template void Step35::NavierStokesProjection<2>::copy_local_to_global_projection_step_rhs(
    const Step35::PressureRightHandSideAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_projection_step_rhs(
    const Step35::PressureRightHandSideAssembly::MappingData<3> &);