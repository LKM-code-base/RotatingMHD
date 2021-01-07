#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::assemble_velocity_matrices()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling velocity mass and stiffness matrices...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Constant matrices assembly - Velocity");

  // Reset data
  velocity_mass_matrix    = 0.;
  velocity_laplace_matrix = 0.;

  // Compute the highest polynomial degree from all the integrands 
  const int p_degree = 2 * velocity->fe_degree;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<dim> &scratch,
           AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy         &data)
    {
      this->assemble_local_velocity_matrices(cell, 
                                             scratch,
                                             data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy   &data)
    {
      this->copy_local_to_global_velocity_matrices(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              (velocity->dof_handler)->begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              (velocity->dof_handler)->end()),
   worker,
   copier,
   AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<dim>(
    *mapping,
    quadrature_formula,
    velocity->fe,
    update_values|
    update_gradients|
    update_JxW_values),
   AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy(velocity->fe.dofs_per_cell));

  // Compress global data
  velocity_mass_matrix.compress(VectorOperation::add);
  velocity_laplace_matrix.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_velocity_matrices
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<dim>   &scratch,
 AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy           &data)
{
  // Reset local data
  data.local_mass_matrix = 0.;
  data.local_stiffness_matrix = 0.;

  // Velocity's cell data
  scratch.fe_values.reinit(cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values[vector_extractor].value(i, q);
      scratch.grad_phi[i] = scratch.fe_values[vector_extractor].gradient(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      // Compute values of the lower triangular part (Symmetry)
      for (unsigned int j = 0; j <= i; ++j)
      {
        // Local matrices
        data.local_mass_matrix(i, j) += scratch.phi[i] *
                                        scratch.phi[j] *
                                        scratch.fe_values.JxW(q);
        data.local_stiffness_matrix(i, j) +=  scalar_product(
                                                scratch.grad_phi[i],
                                                scratch.grad_phi[j]) *
                                              scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Copy lower triangular part values into the upper triangular part
  for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.dofs_per_cell; ++j)
    {
      data.local_mass_matrix(i, j)      = data.local_mass_matrix(j, i);
      data.local_stiffness_matrix(i, j) = data.local_stiffness_matrix(j, i);
    }
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_velocity_matrices
(const AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy &data)
{
  velocity->constraints.distribute_local_to_global(
                                      data.local_mass_matrix,
                                      data.local_dof_indices,
                                      velocity_mass_matrix);
  velocity->constraints.distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      velocity_laplace_matrix);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_pressure_matrices()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling pressure mass and stiffness matrices...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Constant matrices assembly - Pressure");

  // Reset data
  pressure_mass_matrix    = 0.;
  pressure_laplace_matrix = 0.;

  // Compute the highest polynomial degree from all the integrands 
  const int p_degree = 2 * pressure->fe_degree;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<dim> &scratch,
           AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy         &data)
    {
      this->assemble_local_pressure_matrices(cell, 
                                             scratch,
                                             data);
    };
  
  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy &data)
    {
      this->copy_local_to_global_pressure_matrices(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              (pressure->dof_handler)->begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              (pressure->dof_handler)->end()),
   worker,
   copier,
   AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<dim>(
    *mapping,
    quadrature_formula,
    pressure->fe,
    update_values|
    update_gradients|
    update_JxW_values),
   AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy(pressure->fe.dofs_per_cell));

  // Compress global data
  pressure_mass_matrix.compress(VectorOperation::add);
  pressure_laplace_matrix.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_pressure_matrices
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<dim>   &scratch,
 AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy           &data)
{
  // Reset local data
  data.local_mass_matrix      = 0.;
  data.local_stiffness_matrix = 0.;

  // Pressure's cell data
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
        data.local_mass_matrix(i, j) += scratch.phi[i] *
                                        scratch.phi[j] *
                                        scratch.fe_values.JxW(q);
        data.local_stiffness_matrix(i, j) +=  scratch.grad_phi[i] *
                                              scratch.grad_phi[j] *
                                              scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Copy lower triangular part values into the upper triangular part
  for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.dofs_per_cell; ++j)
    {
      data.local_mass_matrix(i, j)      = data.local_mass_matrix(j, i);
      data.local_stiffness_matrix(i, j) = data.local_stiffness_matrix(j, i);
    }
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_pressure_matrices
(const AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy &data)
{
  pressure->constraints.distribute_local_to_global(
                                      data.local_mass_matrix,
                                      data.local_dof_indices,
                                      pressure_mass_matrix);
  pressure->constraints.distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      pressure_laplace_matrix);
}

} // namespace Step35

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_velocity_matrices();
template void RMHD::NavierStokesProjection<3>::assemble_velocity_matrices();

template void RMHD::NavierStokesProjection<2>::assemble_local_velocity_matrices
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<2> &,
 RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_velocity_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<3> &,
 RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy       &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_velocity_matrices
(const RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_velocity_matrices
(const RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy &);

template void RMHD::NavierStokesProjection<2>::assemble_pressure_matrices();
template void RMHD::NavierStokesProjection<3>::assemble_pressure_matrices();

template void RMHD::NavierStokesProjection<2>::assemble_local_pressure_matrices
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<2> &,
 RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_pressure_matrices
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<3>   &,
 RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy         &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_pressure_matrices
(const RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy   &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_pressure_matrices
(const RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy   &);
