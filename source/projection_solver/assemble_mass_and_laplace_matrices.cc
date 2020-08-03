#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_velocity_matrices()
{
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           VelocityMatricesAssembly::LocalCellData<dim>         &scratch,
           VelocityMatricesAssembly::MappingData<dim>           &data)
    {
      this->assemble_local_velocity_matrices(cell, 
                                             scratch,
                                             data);
    };
  
  auto copier =
    [this](const VelocityMatricesAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_velocity_matrices(data);
    };

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             velocity_dof_handler.begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             velocity_dof_handler.end()),
                  worker,
                  copier,
                  VelocityMatricesAssembly::LocalCellData<dim>(
                                            velocity_fe,
                                            velocity_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values),
                  VelocityMatricesAssembly::MappingData<dim>(
                                            velocity_fe.dofs_per_cell));
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_velocity_matrices(
  const typename DoFHandler<dim>::active_cell_iterator  &cell, 
  VelocityMatricesAssembly::LocalCellData<dim>       &scratch, 
  VelocityMatricesAssembly::MappingData<dim>         &data)
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
  const VelocityMatricesAssembly::MappingData<dim> &data)
{
  velocity_constraints.distribute_local_to_global(
                                      data.local_velocity_mass_matrix,
                                      data.local_velocity_dof_indices,
                                      velocity_mass_matrix);
  velocity_constraints.distribute_local_to_global(
                                      data.local_velocity_laplace_matrix,
                                      data.local_velocity_dof_indices,
                                      velocity_laplace_matrix);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_pressure_matrices()
{
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           PressureMatricesAssembly::LocalCellData<dim>         &scratch,
           PressureMatricesAssembly::MappingData<dim>           &data)
    {
      this->assemble_local_pressure_matrices(cell, 
                                             scratch,
                                             data);
    };
  
  auto copier =
    [this](const PressureMatricesAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_pressure_matrices(data);
    };

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             pressure_dof_handler.begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             pressure_dof_handler.end()),
                  worker,
                  copier,
                  PressureMatricesAssembly::LocalCellData<dim>(
                                            pressure_fe,
                                            pressure_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values),
                  PressureMatricesAssembly::MappingData<dim>(
                                            pressure_fe.dofs_per_cell));
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_local_pressure_matrices(
  const typename DoFHandler<dim>::active_cell_iterator  &cell, 
  PressureMatricesAssembly::LocalCellData<dim>          &scratch, 
  PressureMatricesAssembly::MappingData<dim>            &data)
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
  const PressureMatricesAssembly::MappingData<dim> &data)
{
  /*for (unsigned int i = 0; i < pressure_fe.dofs_per_cell; ++i)
    for (unsigned int j = 0; j < pressure_fe.dofs_per_cell; ++j)
    {
      pressure_mass_matrix.add(data.local_pressure_dof_indices[i],
                               data.local_pressure_dof_indices[j],
                               data.local_pressure_mass_matrix(i, j));
    }*/
  pressure_constraints.distribute_local_to_global(
                                      data.local_pressure_mass_matrix,
                                      data.local_pressure_dof_indices,
                                      pressure_mass_matrix);
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
    Step35::VelocityMatricesAssembly::LocalCellData<2>   &,
    Step35::VelocityMatricesAssembly::MappingData<2>     &);
template void Step35::NavierStokesProjection<3>::assemble_local_velocity_matrices(
    const typename DoFHandler<3>::active_cell_iterator      &,
    Step35::VelocityMatricesAssembly::LocalCellData<3>   &,
    Step35::VelocityMatricesAssembly::MappingData<3>     &);

template void Step35::NavierStokesProjection<2>::copy_local_to_global_velocity_matrices(
  const Step35::VelocityMatricesAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_velocity_matrices(
  const Step35::VelocityMatricesAssembly::MappingData<3> &);

template void Step35::NavierStokesProjection<2>::assemble_pressure_matrices();
template void Step35::NavierStokesProjection<3>::assemble_pressure_matrices();

template void Step35::NavierStokesProjection<2>::assemble_local_pressure_matrices(
    const typename DoFHandler<2>::active_cell_iterator      &,
    Step35::PressureMatricesAssembly::LocalCellData<2>   &,
    Step35::PressureMatricesAssembly::MappingData<2>     &);
template void Step35::NavierStokesProjection<3>::assemble_local_pressure_matrices(
    const typename DoFHandler<3>::active_cell_iterator      &,
    Step35::PressureMatricesAssembly::LocalCellData<3>   &,
    Step35::PressureMatricesAssembly::MappingData<3>     &);

template void Step35::NavierStokesProjection<2>::copy_local_to_global_pressure_matrices(
  const Step35::PressureMatricesAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::copy_local_to_global_pressure_matrices(
  const Step35::PressureMatricesAssembly::MappingData<3> &);
