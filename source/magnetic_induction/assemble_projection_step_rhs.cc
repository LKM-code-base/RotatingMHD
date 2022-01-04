#include <rotatingMHD/magnetic_induction.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_projection_step_rhs()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic induction: Assembling the projection step's right hand side...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic induction: Projection step - RHS assembly");

  // Reset data
  this->projection_step_rhs = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(this->auxiliary_scalar->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<dim> &scratch,
           AssemblyData::MagneticInduction::ProjectionStepRHS::Copy         &data)
    {
      this->assemble_local_projection_step_rhs(cell,
                                               scratch,
                                               data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::MagneticInduction::ProjectionStepRHS::Copy  &data)
    {
      this->copy_local_to_global_projection_step_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags magnetic_field_update_flags =
    update_gradients;

  const UpdateFlags auxiliary_scalar_update_flags =
    update_values|
    update_JxW_values;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               this->auxiliary_scalar->get_dof_handler().begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               this->auxiliary_scalar->get_dof_handler().end()),
    worker,
    copier,
    AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<dim>(
      *this->mapping,
      quadrature_formula,
      magnetic_field->get_finite_element(),
      magnetic_field_update_flags,
      this->auxiliary_scalar->get_finite_element(),
      auxiliary_scalar_update_flags),
    AssemblyData::MagneticInduction::ProjectionStepRHS::Copy(
      this->auxiliary_scalar->get_finite_element().dofs_per_cell));

  // Compress global data
  this->projection_step_rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  this->norm_projection_step_rhs = this->projection_step_rhs.l2_norm();

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl
                 << "    Right-hand side's L2-norm = "
                 << std::scientific << std::setprecision(6)
                 << this->norm_projection_step_rhs
                 << std::endl;
}



template <int dim>
void MagneticInduction<dim>::assemble_local_projection_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator            &cell,
  AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<dim> &scratch,
  AssemblyData::MagneticInduction::ProjectionStepRHS::Copy         &data)
{
  // Reset local data
  data.local_rhs = 0.;

  // VSIMEX coefficient
  const std::vector<double> alpha = this->time_stepping.get_alpha();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Auxiliary scalar
  scratch.auxiliary_scalar_fe_values.reinit(cell);

  // Magnetic field
  typename DoFHandler<dim>::active_cell_iterator
  magnetic_field_cell(&magnetic_field->get_triangulation(),
                      cell->level(),
                      cell->index(),
                      &magnetic_field->get_dof_handler());

  scratch.magnetic_field_fe_values.reinit(magnetic_field_cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  scratch.magnetic_field_fe_values[vector_extractor].get_function_divergences(
    magnetic_field->solution,
    scratch.magnetic_field_divergences);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.phi[i] = scratch.auxiliary_scalar_fe_values.shape_value(i, q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      data.local_rhs(i) +=
              alpha[0] / this->time_stepping.get_next_step_size() *
              scratch.phi[i] *
              scratch.magnetic_field_divergences[q] *
              scratch.auxiliary_scalar_fe_values.JxW(q);
  } // Loop over quadrature points
}



template <int dim>
void MagneticInduction<dim>::copy_local_to_global_projection_step_rhs(
  const AssemblyData::MagneticInduction::ProjectionStepRHS::Copy &data)
{
  this->auxiliary_scalar->get_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    this->projection_step_rhs,
    data.local_matrix_for_inhomogeneous_bc);
}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_projection_step_rhs();
template void RMHD::Solvers::MagneticInduction<3>::assemble_projection_step_rhs();

template void RMHD::Solvers::MagneticInduction<2>::assemble_local_projection_step_rhs
(const typename DoFHandler<2>::active_cell_iterator                   &,
 RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<2> &,
 RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Copy       &);
template void RMHD::Solvers::MagneticInduction<3>::assemble_local_projection_step_rhs
(const typename DoFHandler<3>::active_cell_iterator                   &,
 RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<3> &,
 RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Copy       &);

template void RMHD::Solvers::MagneticInduction<2>::copy_local_to_global_projection_step_rhs
(const RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Copy &);
template void RMHD::Solvers::MagneticInduction<3>::copy_local_to_global_projection_step_rhs
(const RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Copy &);
