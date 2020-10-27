#include <rotatingMHD/heat_equation.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::assemble_constant_matrices()
{
  if (parameters.verbose)
    *pcout << "  Assembling constant matrices..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Constant matrices assembly");

  const QGauss<dim>  quadrature_formula(2 * temperature.fe_degree);

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           TemperatureConstantMatricesAssembly::LocalCellData<dim>  &scratch,
           TemperatureConstantMatricesAssembly::MappingData<dim>    &data)
    {
      this->assemble_local_constant_matrices(cell, 
                                             scratch,
                                             data);
    };
  
  auto copier =
    [this](const TemperatureConstantMatricesAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_constant_matrices(data);
    };

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.end()),
   worker,
   copier,
   TemperatureConstantMatricesAssembly::LocalCellData<dim>(
    *mapping,
    temperature.fe,
    quadrature_formula,
    update_values|
    update_gradients|
    update_JxW_values),
   TemperatureConstantMatricesAssembly::MappingData<dim>(
     temperature.fe.dofs_per_cell));

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
}

template <int dim>
void HeatEquation<dim>::assemble_local_constant_matrices
(const typename DoFHandler<dim>::active_cell_iterator     &cell,
 TemperatureConstantMatricesAssembly::LocalCellData<dim>  &scratch,
 TemperatureConstantMatricesAssembly::MappingData<dim>    &data)
{
  // Reset local matrices
  data.local_mass_matrix      = 0.;
  data.local_stiffness_matrix = 0.;

  // Prepare temperature part
  scratch.fe_values.reinit(cell);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Populate test functions
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.fe_values.shape_grad(i, q);
    }

    // Loop over local DoFs
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      // Compute values of the lower triangular part (Symmetry)
      for (unsigned int j = 0; j <= i; ++j)
      {
        data.local_mass_matrix(i, j) += 
                                    scratch.fe_values.JxW(q) *
                                    scratch.phi[i] *
                                    scratch.phi[j];
        data.local_stiffness_matrix(i, j) +=
                                    scratch.fe_values.JxW(q) *
                                    scratch.grad_phi[i] *
                                    scratch.grad_phi[j];
      } // Loop over local DoFs
    
    // Copy lower triangular part values into the upper triangular part
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < scratch.dofs_per_cell; ++j)
      {
        data.local_mass_matrix(i, j) = 
                                      data.local_mass_matrix(j, i);
        data.local_stiffness_matrix(i, j) = 
                                      data.local_stiffness_matrix(j, i);
      }
  }
}

template <int dim>
void HeatEquation<dim>::copy_local_to_global_constant_matrices
(const TemperatureConstantMatricesAssembly::MappingData<dim> &data)
{
  temperature.constraints.distribute_local_to_global(
                                      data.local_mass_matrix,
                                      data.local_dof_indices,
                                      mass_matrix);
  temperature.constraints.distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      stiffness_matrix);
}


} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::assemble_constant_matrices();
template void RMHD::HeatEquation<3>::assemble_constant_matrices();

template void RMHD::HeatEquation<2>::assemble_local_constant_matrices
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::TemperatureConstantMatricesAssembly::LocalCellData<2>    &,
 RMHD::TemperatureConstantMatricesAssembly::MappingData<2>      &);
template void RMHD::HeatEquation<3>::assemble_local_constant_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::TemperatureConstantMatricesAssembly::LocalCellData<3>    &,
 RMHD::TemperatureConstantMatricesAssembly::MappingData<3>      &);

template void RMHD::HeatEquation<2>::copy_local_to_global_constant_matrices
(const RMHD::TemperatureConstantMatricesAssembly::MappingData<2>  &);
template void RMHD::HeatEquation<3>::copy_local_to_global_constant_matrices
(const RMHD::TemperatureConstantMatricesAssembly::MappingData<3>  &);