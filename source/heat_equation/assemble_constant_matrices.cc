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
    *pcout << "  Heat Equation: Assembling constant matrices...";

  TimerOutput::Scope  t(*computing_timer, "Heat Equation: Constant matrices assembly");

  // Reset data
  mass_matrix       = 0.;
  stiffness_matrix  = 0.;

  // Compute the highest polynomial degree from all the integrands 
  const int p_degree = 2 * temperature->fe_degree;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           AssemblyData::HeatEquation::ConstantMatrices::Scratch<dim>  &scratch,
           AssemblyData::HeatEquation::ConstantMatrices::Copy<dim>    &data)
    {
      this->assemble_local_constant_matrices(cell, 
                                             scratch,
                                             data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::HeatEquation::ConstantMatrices::Copy<dim> &data) 
    {
      this->copy_local_to_global_constant_matrices(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              (temperature->dof_handler)->begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              (temperature->dof_handler)->end()),
   worker,
   copier,
   AssemblyData::HeatEquation::ConstantMatrices::Scratch<dim>(
    *mapping,
    quadrature_formula,
    temperature->fe,
    update_values|
    update_gradients|
    update_JxW_values),
   AssemblyData::HeatEquation::ConstantMatrices::Copy<dim>(
     temperature->fe.dofs_per_cell));

  // Compress global data
  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
  
  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void HeatEquation<dim>::assemble_local_constant_matrices
(const typename DoFHandler<dim>::active_cell_iterator     &cell,
 AssemblyData::HeatEquation::ConstantMatrices::Scratch<dim>  &scratch,
 AssemblyData::HeatEquation::ConstantMatrices::Copy<dim>    &data)
{
  // Reset local data
  data.local_mass_matrix      = 0.;
  data.local_stiffness_matrix = 0.;

  // Temperature's cell data
  scratch.fe_values.reinit(cell);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.fe_values.shape_grad(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      // Compute values of the lower triangular part (Symmetry)
      for (unsigned int j = 0; j <= i; ++j)
      {
        // Local matrices
        data.local_mass_matrix(i, j) += 
                                    scratch.phi[i] *
                                    scratch.phi[j] *
                                    scratch.fe_values.JxW(q);
        data.local_stiffness_matrix(i, j) +=
                                    scratch.grad_phi[i] *
                                    scratch.grad_phi[j] *
                                    scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
    
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
(const AssemblyData::HeatEquation::ConstantMatrices::Copy<dim> &data)
{
  temperature->constraints.distribute_local_to_global(
                                      data.local_mass_matrix,
                                      data.local_dof_indices,
                                      mass_matrix);
  temperature->constraints.distribute_local_to_global(
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
 RMHD::AssemblyData::HeatEquation::ConstantMatrices::Scratch<2>    &,
 RMHD::AssemblyData::HeatEquation::ConstantMatrices::Copy<2>      &);
template void RMHD::HeatEquation<3>::assemble_local_constant_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::AssemblyData::HeatEquation::ConstantMatrices::Scratch<3>    &,
 RMHD::AssemblyData::HeatEquation::ConstantMatrices::Copy<3>      &);

template void RMHD::HeatEquation<2>::copy_local_to_global_constant_matrices
(const RMHD::AssemblyData::HeatEquation::ConstantMatrices::Copy<2>  &);
template void RMHD::HeatEquation<3>::copy_local_to_global_constant_matrices
(const RMHD::AssemblyData::HeatEquation::ConstantMatrices::Copy<3>  &);