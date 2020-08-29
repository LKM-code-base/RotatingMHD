#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::assemble_velocity_matrices()
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

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.end()),
   worker,
   copier,
   VelocityMatricesAssembly::LocalCellData<dim>(velocity.fe,
                                                velocity.quadrature_formula,
                                                update_values|
                                                update_gradients|
                                                update_JxW_values),
   VelocityMatricesAssembly::MappingData<dim>(velocity.fe.dofs_per_cell));

  velocity_mass_matrix.compress(VectorOperation::add);
  velocity_laplace_matrix.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_velocity_matrices
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 VelocityMatricesAssembly::LocalCellData<dim>          &scratch,
 VelocityMatricesAssembly::MappingData<dim>            &data)
{
  // reset local matrices
  data.local_velocity_mass_matrix = 0.;
  data.local_velocity_laplace_matrix = 0.;

  // prepare velocity part
  const FEValuesExtractors::Vector  velocities(0);

  scratch.velocity_fe_values.reinit(cell);

  cell->get_dof_indices(data.local_velocity_dof_indices);

  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = 
                    scratch.velocity_fe_values[velocities].value(i, q);
      scratch.grad_phi_velocity[i] =
                    scratch.velocity_fe_values[velocities].gradient(i, q);
    }
    // loop over local dofs
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
      // exploit symmetry and compute values only for the lower triangular part
      for (unsigned int j = 0; j <= i; ++j)
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
      } // loop over local dofs
  } // loop over quadrature points

  // exploit symmetry and copy values to upper triangular part
  for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.velocity_dofs_per_cell; ++j)
    {
      data.local_velocity_mass_matrix(i, j) =
                            data.local_velocity_mass_matrix(j, i);
      data.local_velocity_laplace_matrix(i, j) =
                            data.local_velocity_laplace_matrix(j, i);
    }
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_velocity_matrices
(const VelocityMatricesAssembly::MappingData<dim> &data)
{
  velocity.constraints.distribute_local_to_global(
                                      data.local_velocity_mass_matrix,
                                      data.local_velocity_dof_indices,
                                      velocity_mass_matrix);
  velocity.constraints.distribute_local_to_global(
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

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              pressure.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              pressure.dof_handler.end()),
   worker,
   copier,
   PressureMatricesAssembly::LocalCellData<dim>(pressure.fe,
                                                pressure.quadrature_formula,
                                                update_values|
                                                update_gradients|
                                                update_JxW_values),
   PressureMatricesAssembly::MappingData<dim>(pressure.fe.dofs_per_cell));

  pressure_mass_matrix.compress(VectorOperation::add);
  pressure_laplace_matrix.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_pressure_matrices
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 PressureMatricesAssembly::LocalCellData<dim>          &scratch,
 PressureMatricesAssembly::MappingData<dim>            &data)
{
  // reset local matrices
  data.local_pressure_mass_matrix = 0.;
  data.local_pressure_laplace_matrix = 0.;

  // prepare pressure part
  scratch.pressure_fe_values.reinit(cell);

  cell->get_dof_indices(data.local_pressure_dof_indices);

  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
    {
      scratch.phi_pressure[i] = 
                    scratch.pressure_fe_values.shape_value(i, q);
      scratch.grad_phi_pressure[i] =
                    scratch.pressure_fe_values.shape_grad(i, q);
    }
    // loop over local dofs
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      // exploit symmetry and compute values only for the lower triangular part
      for (unsigned int j = 0; j <= i; ++j)
      {
        data.local_pressure_mass_matrix(i, j) += 
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.phi_pressure[i] *
                                    scratch.phi_pressure[j];
        data.local_pressure_laplace_matrix(i, j) +=
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.grad_phi_pressure[i] *
                                    scratch.grad_phi_pressure[j];
      } // loop over local dofs
  } // loop over quadrature points

  // exploit symmetry and copy values to upper triangular part
  for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.pressure_dofs_per_cell; ++j)
    {
      data.local_pressure_mass_matrix(i, j) =
                            data.local_pressure_mass_matrix(j, i);
      data.local_pressure_laplace_matrix(i, j) =
                            data.local_pressure_laplace_matrix(j, i);
    }
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_pressure_matrices
(const PressureMatricesAssembly::MappingData<dim> &data)
{
  pressure.constraints.distribute_local_to_global(
                                      data.local_pressure_mass_matrix,
                                      data.local_pressure_dof_indices,
                                      pressure_mass_matrix);
  pressure.constraints.distribute_local_to_global(
                                      data.local_pressure_laplace_matrix,
                                      data.local_pressure_dof_indices,
                                      pressure_laplace_matrix);
}

} // namespace Step35

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_velocity_matrices();
template void RMHD::NavierStokesProjection<3>::assemble_velocity_matrices();

template void RMHD::NavierStokesProjection<2>::assemble_local_velocity_matrices
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::VelocityMatricesAssembly::LocalCellData<2>    &,
 RMHD::VelocityMatricesAssembly::MappingData<2>      &);
template void RMHD::NavierStokesProjection<3>::assemble_local_velocity_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::VelocityMatricesAssembly::LocalCellData<3>    &,
 RMHD::VelocityMatricesAssembly::MappingData<3>      &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_velocity_matrices
(const RMHD::VelocityMatricesAssembly::MappingData<2>  &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_velocity_matrices
(const RMHD::VelocityMatricesAssembly::MappingData<3>  &);

template void RMHD::NavierStokesProjection<2>::assemble_pressure_matrices();
template void RMHD::NavierStokesProjection<3>::assemble_pressure_matrices();

template void RMHD::NavierStokesProjection<2>::assemble_local_pressure_matrices
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::PressureMatricesAssembly::LocalCellData<2>    &,
 RMHD::PressureMatricesAssembly::MappingData<2>      &);
template void RMHD::NavierStokesProjection<3>::assemble_local_pressure_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::PressureMatricesAssembly::LocalCellData<3>    &,
 RMHD::PressureMatricesAssembly::MappingData<3>      &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_pressure_matrices
(const RMHD::PressureMatricesAssembly::MappingData<2>  &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_pressure_matrices
(const RMHD::PressureMatricesAssembly::MappingData<3>  &);
