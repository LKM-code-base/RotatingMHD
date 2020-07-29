#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_velocity_matrices()
{
  VelocityMassLaplaceAssembly::MappingData<dim>   data(
                                            velocity_fe.dofs_per_cell);
  VelocityMassLaplaceAssembly::LocalCellData<dim> scratch(
                                            velocity_fe,
                                            velocity_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values);
  WorkStream::run(
    velocity_dof_handler.begin_active(),
    velocity_dof_handler.end(),
    *this,
    &NavierStokesProjection<dim>::assemble_local_velocity_matrices,
    &NavierStokesProjection<dim>::copy_local_to_global_velocity_matrices,
    scratch,
    data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_velocity_matrices(
  const typename DoFHandler<dim>::active_cell_iterator  &cell, 
  VelocityMassLaplaceAssembly::LocalCellData<dim>       &scratch, 
  VelocityMassLaplaceAssembly::MappingData<dim>         &data)
{
  scratch.velocity_fe_values.reinit(cell);
  cell->get_dof_indices(data.local_velocity_dof_indices);
  const FEValuesExtractors::Vector  velocity(0);

  data.local_velocity_mass_matrix = 0.;
  data.local_velocity_laplace_matrix = 0.;

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = 
                    scratch.velocity_fe_values[velocity].value(i, q);
      scratch.grad_phi_velocity[i] =
                    scratch.velocity_fe_values[velocity].gradient(i, q);
    }
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.velocity_dofs_per_cell; ++j)
      {
        data.local_velocity_mass_matrix(i, j) += 
                                    scratch.velocity_fe_values.JxW(q) *
                                    scratch.phi_velocity[i] *
                                    scratch.phi_velocity[j];
        data.local_velocity_laplace_matrix(i, j) +=
                                    scratch.velocity_fe_values.JxW(q) *
                                    scalar_product(
                                      scratch.grad_phi_velocity[i],
                                      scratch.grad_phi_velocity[j]);
      }
  }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_velocity_matrices(
  const VelocityMassLaplaceAssembly::MappingData<dim> &data)
{
  for (unsigned int i = 0; i < velocity_fe.dofs_per_cell; ++i)
    for (unsigned int j = 0; j < velocity_fe.dofs_per_cell; ++j)
    {
      velocity_mass_matrix.add(data.local_velocity_dof_indices[i],
                               data.local_velocity_dof_indices[j],
                               data.local_velocity_mass_matrix(i, j));
      //velocity_laplace_matrix.add(data.local_velocity_dof_indices[i],
      //                            data.local_velocity_dof_indices[j],
      //                            data.local_velocity_laplace_matrix(i, j));
    }
  /*velocity_constraints.distribute_local_to_global(
                                      data.local_velocity_mass_matrix,
                                      data.local_velocity_dof_indices,
                                      velocity_mass_matrix);*/
  velocity_constraints.distribute_local_to_global(
                                      data.local_velocity_laplace_matrix,
                                      data.local_velocity_dof_indices,
                                      velocity_laplace_matrix);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_pressure_matrices()
{
  PressureMassLaplaceAssembly::MappingData<dim>   data(
                                            pressure_fe.dofs_per_cell);
  PressureMassLaplaceAssembly::LocalCellData<dim> scratch(
                                            pressure_fe,
                                            pressure_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values);
  WorkStream::run(
    pressure_dof_handler.begin_active(),
    pressure_dof_handler.end(),
    *this,
    &NavierStokesProjection<dim>::assemble_local_pressure_matrices,
    &NavierStokesProjection<dim>::copy_local_to_global_pressure_matrices,
    scratch,
    data);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_pressure_matrices(
  const typename DoFHandler<dim>::active_cell_iterator  &cell, 
  PressureMassLaplaceAssembly::LocalCellData<dim>       &scratch, 
  PressureMassLaplaceAssembly::MappingData<dim>         &data)
{
  scratch.pressure_fe_values.reinit(cell);
  cell->get_dof_indices(data.local_pressure_dof_indices);

  data.local_pressure_mass_matrix = 0.;
  data.local_pressure_laplace_matrix = 0.;

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
    {
      scratch.phi_pressure[i] = 
                    scratch.pressure_fe_values.shape_value(i, q);
      scratch.grad_phi_pressure[i] =
                    scratch.pressure_fe_values.shape_grad(i, q);
    }
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.pressure_dofs_per_cell; ++j)
      {
        data.local_pressure_mass_matrix(i, j) += 
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.phi_pressure[i] *
                                    scratch.phi_pressure[j];
        data.local_pressure_laplace_matrix(i, j) +=
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.grad_phi_pressure[i] *
                                    scratch.grad_phi_pressure[j];
      }
  }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_pressure_matrices(
  const PressureMassLaplaceAssembly::MappingData<dim> &data)
{
  for (unsigned int i = 0; i < pressure_fe.dofs_per_cell; ++i)
    for (unsigned int j = 0; j < pressure_fe.dofs_per_cell; ++j)
    {
      pressure_mass_matrix.add(data.local_pressure_dof_indices[i],
                               data.local_pressure_dof_indices[j],
                               data.local_pressure_mass_matrix(i, j));
      /*pressure_laplace_matrix.add(data.local_pressure_dof_indices[i],
                                  data.local_pressure_dof_indices[j],
                                  data.local_pressure_laplace_matrix(i, j));
      */
    }
  /*pressure_constraints.distribute_local_to_global(
                                      data.local_pressure_mass_matrix,
                                      data.local_pressure_dof_indices,
                                      pressure_mass_matrix);*/
  pressure_constraints.distribute_local_to_global(
                                      data.local_pressure_laplace_matrix,
                                      data.local_pressure_dof_indices,
                                      pressure_laplace_matrix);
}
} // namespace Step35

// explicit instantiations
template void Step35::NavierStokesProjection<2>::assemble_velocity_matrices();
template void Step35::NavierStokesProjection<3>::assemble_velocity_matrices();

template void Step35::NavierStokesProjection<2>::assemble_local_velocity_matrices(
    const typename DoFHandler<2>::active_cell_iterator      &,
    Step35::VelocityMassLaplaceAssembly::LocalCellData<2>   &,
    Step35::VelocityMassLaplaceAssembly::MappingData<2>     &);
template void Step35::NavierStokesProjection<3>::assemble_local_velocity_matrices(
    const typename DoFHandler<3>::active_cell_iterator      &,
    Step35::VelocityMassLaplaceAssembly::LocalCellData<3>   &,
    Step35::VelocityMassLaplaceAssembly::MappingData<3>     &);

template void Step35::NavierStokesProjection<2>::copy_local_to_global_velocity_matrices(
  const Step35::VelocityMassLaplaceAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_velocity_matrices(
  const Step35::VelocityMassLaplaceAssembly::MappingData<3> &);

template void Step35::NavierStokesProjection<2>::assemble_pressure_matrices();
template void Step35::NavierStokesProjection<3>::assemble_pressure_matrices();

template void Step35::NavierStokesProjection<2>::assemble_local_pressure_matrices(
    const typename DoFHandler<2>::active_cell_iterator      &,
    Step35::PressureMassLaplaceAssembly::LocalCellData<2>   &,
    Step35::PressureMassLaplaceAssembly::MappingData<2>     &);
template void Step35::NavierStokesProjection<3>::assemble_local_pressure_matrices(
    const typename DoFHandler<3>::active_cell_iterator      &,
    Step35::PressureMassLaplaceAssembly::LocalCellData<3>   &,
    Step35::PressureMassLaplaceAssembly::MappingData<3>     &);

template void Step35::NavierStokesProjection<2>::copy_local_to_global_pressure_matrices(
  const Step35::PressureMassLaplaceAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_pressure_matrices(
  const Step35::PressureMassLaplaceAssembly::MappingData<3> &);
