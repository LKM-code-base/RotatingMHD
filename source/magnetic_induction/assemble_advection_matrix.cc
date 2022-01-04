#include <rotatingMHD/magnetic_induction.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/filtered_iterator.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_advection_matrix()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic induction: Assembling advection matrix...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic induction: Advection matrix assembly");

  // Reset data
  this->diffusion_step_advection_matrix = 0.;

  // Dummy finite element for when the velocity is given by a function
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(1), dim);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim>* const velocity_fe =
              (velocity != nullptr) ?
                &velocity->get_finite_element() :
                &dummy_fe_system;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(magnetic_field->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = typename AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<dim>;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<dim> &scratch,
           AssemblyData::MagneticInduction::AdvectionMatrix::Copy         &data)
    {
      this->assemble_local_advection_matrix(cell,
                                                     scratch,
                                                     data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::MagneticInduction::AdvectionMatrix::Copy &data)
    {
      this->copy_local_to_global_advection_matrix(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags advection_update_flags = update_values|
                                             update_gradients|
                                             update_JxW_values;
  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              magnetic_field->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              magnetic_field->get_dof_handler().end()),
   worker,
   copier,
   AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<dim>(
     *this->mapping,
     quadrature_formula,
     magnetic_field->get_finite_element(),
     advection_update_flags,
     *velocity_fe,
     advection_update_flags),
   AssemblyData::MagneticInduction::AdvectionMatrix::Copy(
     magnetic_field->get_finite_element().dofs_per_cell));

  // Compress global data
  this->diffusion_step_advection_matrix.compress(VectorOperation::add);

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl;
}



template <int dim>
void MagneticInduction<dim>::assemble_local_advection_matrix(
  const typename DoFHandler<dim>::active_cell_iterator            &cell,
  AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::AdvectionMatrix::Copy          &data)
{
  // Reset local data
  data.local_matrix = 0.;

  // Taylor extrapolation coefficients
  const std::vector<double> eta = this->time_stepping.get_eta();

  // Vector extractor
  const FEValuesExtractors::Vector  vector_extractor(0);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Magnetic field's cell data
  scratch.magnetic_field_fe_values.reinit(cell);

  // Velocity's cell data
  if (velocity != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell(&magnetic_field->get_triangulation(),
                  cell->level(),
                  cell->index(),
                  //Pointer to the velocity's DoFHandler
                  &velocity->get_dof_handler());

    scratch.velocity_fe_values.reinit(velocity_cell);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_solution,
      scratch.old_velocity_values);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_old_solution,
      scratch.old_old_velocity_values);

    scratch.velocity_fe_values[vector_extractor].get_function_gradients(
      velocity->old_solution,
      scratch.old_velocity_gradients);

    scratch.velocity_fe_values[vector_extractor].get_function_gradients(
      velocity->old_old_solution,
      scratch.old_old_velocity_gradients);

    for (unsigned int i = 0; i < scratch.n_q_points; ++i)
    {
      scratch.velocity_values[i] =
        eta[0] * scratch.old_velocity_values[i] +
        eta[1] * scratch.old_old_velocity_values[i];

      scratch.velocity_gradients[i] =
        eta[0] * scratch.old_velocity_gradients[i] +
        eta[1] * scratch.old_old_velocity_gradients[i];
    }
  }
  else if (ptr_velocity_function != nullptr)
  {
    ptr_velocity_function->value_list(
      scratch.magnetic_field_fe_values.get_quadrature_points(),
      scratch.velocity_values);

    ptr_velocity_function->gradient_list(
      scratch.magnetic_field_fe_values.get_quadrature_points(),
      scratch.velocity_gradients);
  }
  else
  {
    ZeroTensorFunction<1,dim>().value_list(
      scratch.magnetic_field_fe_values.get_quadrature_points(),
      scratch.velocity_values);

    ZeroTensorFunction<2,dim>().value_list(
      scratch.magnetic_field_fe_values.get_quadrature_points(),
      scratch.velocity_gradients);
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.magnetic_field_fe_values[vector_extractor].value(i, q);
      scratch.grad_phi[i] = scratch.magnetic_field_fe_values[vector_extractor].gradient(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        data.local_matrix(i,j) +=
          scratch.phi[i] *
          (scratch.velocity_gradients[q] *
           scratch.phi[j]
           -
           scratch.grad_phi[j] *
           scratch.velocity_values[q]) *
          scratch.magnetic_field_fe_values.JxW(q);
    // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template <int dim>
void MagneticInduction<dim>::copy_local_to_global_advection_matrix(
  const AssemblyData::MagneticInduction::AdvectionMatrix::Copy  &data)
{
  magnetic_field->get_constraints().distribute_local_to_global(
    data.local_matrix,
    data.local_dof_indices,
    this->diffusion_step_advection_matrix);
}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_advection_matrix();
template void RMHD::Solvers::MagneticInduction<3>::assemble_advection_matrix();

template void RMHD::Solvers::MagneticInduction<2>::assemble_local_advection_matrix
(const typename DoFHandler<2>::active_cell_iterator                 &,
 RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<2> &,
 RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Copy       &);
template void RMHD::Solvers::MagneticInduction<3>::assemble_local_advection_matrix
(const typename DoFHandler<3>::active_cell_iterator                 &,
 RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<3> &,
 RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Copy       &);

template void RMHD::Solvers::MagneticInduction<2>::copy_local_to_global_advection_matrix
(const RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Copy &);
template void RMHD::Solvers::MagneticInduction<3>::copy_local_to_global_advection_matrix
(const RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Copy &);
