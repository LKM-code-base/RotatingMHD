#include <rotatingMHD/heat_equation.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::assemble_advection_matrix()
{
  if (parameters.verbose)
    *pcout << "    Heat Equation: Assembling advection matrix..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Heat Equation: Advection matrix assembly");

  advection_matrix = 0.;

  // Polynomial degree of the integrand
  const int p_degree = velocity->fe_degree + 2 * temperature.fe_degree - 1;

  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;


  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           TemperatureAdvectionMatrixAssembly::LocalCellData<dim>  &scratch,
           TemperatureAdvectionMatrixAssembly::MappingData<dim>    &data)
    {
      this->assemble_local_advection_matrix(cell, 
                                             scratch,
                                             data);
    };

  auto copier =
    [this](const TemperatureAdvectionMatrixAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_advection_matrix(data);
    };

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.end()),
   worker,
   copier,
   TemperatureAdvectionMatrixAssembly::LocalCellData<dim>(
    *mapping,
    temperature.fe,
    velocity->fe,
    quadrature_formula,
    update_values|
    update_gradients|
    update_JxW_values,
    update_values | update_quadrature_points),
   TemperatureAdvectionMatrixAssembly::MappingData<dim>(
     temperature.fe.dofs_per_cell));

  advection_matrix.compress(VectorOperation::add);
}

template <int dim>
void HeatEquation<dim>::assemble_local_advection_matrix
(const typename DoFHandler<dim>::active_cell_iterator     &cell,
 TemperatureAdvectionMatrixAssembly::LocalCellData<dim>  &scratch,
 TemperatureAdvectionMatrixAssembly::MappingData<dim>    &data)
{
  // Reset local matrices
  data.local_matrix      = 0.;

  // Prepare temperature part
  scratch.temperature_fe_values.reinit(cell);

  // Prepare velocity part
  typename DoFHandler<dim>::active_cell_iterator
  velocity_cell(&temperature.dof_handler.get_triangulation(),
                 cell->level(),
                 cell->index(),
                &velocity->dof_handler);

  const FEValuesExtractors::Vector velocities(0);

  scratch.velocity_fe_values.reinit(velocity_cell);

  if (velocity != nullptr)
    scratch.velocity_fe_values[velocities].get_function_values
    (velocity->solution,
    scratch.velocity_values);
  else
    ZeroTensorFunction<1,dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.velocity_values);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Populate test functions
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.temperature_fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.temperature_fe_values.shape_grad(i, q);
    }

    // Loop over local DoFs
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        data.local_matrix(i, j) += 
                            scratch.temperature_fe_values.JxW(q) *
                            scratch.phi[i] *
                            (scratch.velocity_values[q] *
                             scratch.grad_phi[j]);
  }
}

template <int dim>
void HeatEquation<dim>::copy_local_to_global_advection_matrix
(const TemperatureAdvectionMatrixAssembly::MappingData<dim> &data)
{
  temperature.constraints.distribute_local_to_global(
                                      data.local_matrix,
                                      data.local_dof_indices,
                                      advection_matrix);
}


} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::assemble_advection_matrix();
template void RMHD::HeatEquation<3>::assemble_advection_matrix();

template void RMHD::HeatEquation<2>::assemble_local_advection_matrix
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::TemperatureAdvectionMatrixAssembly::LocalCellData<2>    &,
 RMHD::TemperatureAdvectionMatrixAssembly::MappingData<2>      &);
template void RMHD::HeatEquation<3>::assemble_local_advection_matrix
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::TemperatureAdvectionMatrixAssembly::LocalCellData<3>    &,
 RMHD::TemperatureAdvectionMatrixAssembly::MappingData<3>      &);

template void RMHD::HeatEquation<2>::copy_local_to_global_advection_matrix
(const RMHD::TemperatureAdvectionMatrixAssembly::MappingData<2>  &);
template void RMHD::HeatEquation<3>::copy_local_to_global_advection_matrix
(const RMHD::TemperatureAdvectionMatrixAssembly::MappingData<3>  &);