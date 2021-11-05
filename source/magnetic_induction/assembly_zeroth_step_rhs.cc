#include <rotatingMHD/magnetic_induction.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_initialization_step_rhs()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic induction: Assembling zeroth step's right hand side...";

  TimerOutput::Scope  t(*this->computing_timer, "Magnetic induction: Zeroth step - RHS assembly");

  // Reset data
  this->initialization_step_rhs = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(pseudo_pressure->fe_degree() + 1);

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim-1> face_quadrature_formula(pseudo_pressure->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator         &cell,
           AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<dim> &scratch,
           AssemblyData::MagneticInduction::ZerothStepRHS::Copy         &data)
    {
      this->assemble_local_initialization_step_rhs(cell,
                                           scratch,
                                           data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::MagneticInduction::ZerothStepRHS::Copy &data)
    {
      this->copy_local_to_global_initialization_step_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags magnetic_field_face_update_flags =
    update_hessians;

  const UpdateFlags pseudo_pressure_update_flags =
    update_gradients|
    update_quadrature_points|
    update_JxW_values;

  const UpdateFlags pseudo_pressure_face_update_flags =
    update_values|
    update_quadrature_points|
    update_normal_vectors|
    update_JxW_values;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               pseudo_pressure->get_dof_handler().begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               pseudo_pressure->get_dof_handler().end()),
    worker,
    copier,
    AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<dim>(
      *this->mapping,
      quadrature_formula,
      face_quadrature_formula,
      magnetic_field->get_finite_element(),
      magnetic_field_face_update_flags,
      pseudo_pressure->get_finite_element(),
      pseudo_pressure_update_flags,
      pseudo_pressure_face_update_flags),
    AssemblyData::MagneticInduction::ZerothStepRHS::Copy(
      pseudo_pressure->get_finite_element().dofs_per_cell));

  // Compress global data
  this->initialization_step_rhs.compress(VectorOperation::add);

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl
                 << "    Right-hand side's L2-norm = "
                 << std::scientific << std::setprecision(6)
                 << this->initialization_step_rhs.l2_norm()
                 << std::endl;
}



template <int dim>
void MagneticInduction<dim>::assemble_local_initialization_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator            &cell,
  AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<dim> &scratch,
  AssemblyData::MagneticInduction::ZerothStepRHS::Copy         &data)
{
  // Reset local data
  data.local_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Pseudo-pressure
  scratch.pseudo_pressure_fe_values.reinit(cell);

  // Supply term
  if (this->ptr_supply_term != nullptr)
    this->ptr_supply_term->value_list(
    scratch.pseudo_pressure_fe_values.get_quadrature_points(),
    scratch.supply_term_values);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.grad_phi[i] = scratch.pseudo_pressure_fe_values.shape_grad(i, q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
        scratch.grad_phi[i] *
        scratch.supply_term_values[q] *
        scratch.pseudo_pressure_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (pseudo_pressure->get_constraints().is_inhomogeneously_constrained(
        data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) +=
                                    scratch.grad_phi[j] *
                                    scratch.grad_phi[i] *
                                    scratch.pseudo_pressure_fe_values.JxW(q);
      // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Loop over boundary cell's faces
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
      {
        // Pseudo-pressure
        scratch.pseudo_pressure_fe_face_values.reinit(cell, face);

        // magnetic_field
        typename DoFHandler<dim>::active_cell_iterator
        magnetic_field_cell(&magnetic_field->get_triangulation(),
                            cell->level(),
                            cell->index(),
                            &magnetic_field->get_dof_handler());

        typename DoFHandler<dim>::active_face_iterator
        magnetic_field_face(&magnetic_field->get_triangulation(),
                            face->level(),
                            face->index(),
                            &magnetic_field->get_dof_handler());

        const FEValuesExtractors::Vector  vector_extractor(0);

        scratch.pseudo_pressure_fe_face_values.reinit(magnetic_field_cell,
                                               magnetic_field_face);

        scratch.magnetic_field_fe_face_values[vector_extractor].get_function_laplacians(
          magnetic_field->old_solution,
          scratch.magnetic_field_face_laplacians);

        // Normal vector
        scratch.normal_vectors =
          scratch.pseudo_pressure_fe_face_values.get_normal_vectors();

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
          {
            // Extract the test function's values at the face quadrature points
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
              scratch.face_phi[i] =
                scratch.pseudo_pressure_fe_face_values.shape_value(i, q);

            // Loop over the degrees of freedom
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            {
              data.local_rhs(i) +=
                              parameters.C8 *
                              scratch.face_phi[i] *
                              scratch.magnetic_field_face_laplacians[q] *
                              scratch.normal_vectors[q] *
                              scratch.pseudo_pressure_fe_face_values.JxW(q);
            } // Loop over the degrees of freedom
          } // Loop over face quadrature points
      }
}



template <int dim>
void MagneticInduction<dim>::copy_local_to_global_initialization_step_rhs(
  const AssemblyData::MagneticInduction::ZerothStepRHS::Copy &data)
{
  pseudo_pressure->get_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    this->initialization_step_rhs,
    data.local_matrix_for_inhomogeneous_bc);
}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_initialization_step_rhs();
template void RMHD::Solvers::MagneticInduction<3>::assemble_initialization_step_rhs();

template void RMHD::Solvers::MagneticInduction<2>::assemble_local_initialization_step_rhs
(const typename DoFHandler<2>::active_cell_iterator               &,
 RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<2> &,
 RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Copy       &);
template void RMHD::Solvers::MagneticInduction<3>::assemble_local_initialization_step_rhs
(const typename DoFHandler<3>::active_cell_iterator               &,
 RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<3> &,
 RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Copy       &);

template void RMHD::Solvers::MagneticInduction<2>::copy_local_to_global_initialization_step_rhs
(const RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Copy &);
template void RMHD::Solvers::MagneticInduction<3>::copy_local_to_global_initialization_step_rhs
(const RMHD::AssemblyData::MagneticInduction::ZerothStepRHS::Copy &);
