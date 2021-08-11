#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_poisson_prestep_rhs()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling Poisson pre-step's right hand side...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Poisson pre-step - RHS assembly");

  // Reset data
  poisson_prestep_rhs = 0.;

  // Dummy finite element for when there is no buoyancy
  const FE_Q<dim> dummy_fe(1);

  // Create pointer to the pertinent finite element
  const FE_Q<dim> * const temperature_fe_ptr =
          (temperature.get() != nullptr) ? &temperature->fe : &dummy_fe;

  // Set polynomial degree of the body force function.
  // Hardcoded to match that of the velocity.
  const int p_degree_body_force = velocity->fe_degree;

  // Compute the highest polynomial degree from all the integrands
  const int p_degree = pressure->fe_degree + p_degree_body_force - 1;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Compute the highest polynomial degree from all the boundary integrands
  const int face_p_degree = std::max(pressure->fe_degree + p_degree_body_force,
                                     pressure->fe_degree + velocity->fe_degree -2);

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim-1>   face_quadrature_formula(std::ceil(0.5 * double(face_p_degree + 1)));


  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>   &scratch,
           AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy           &data)
    {
      this->assemble_local_poisson_prestep_rhs(cell,
                                               scratch,
                                               data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy &data)
    {
      this->copy_local_to_global_poisson_prestep_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               velocity->get_dof_handler().begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               velocity->get_dof_handler().end()),
    worker,
    copier,
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>(
      *mapping,
      quadrature_formula,
      face_quadrature_formula,
      velocity->fe,
      update_values,
      update_hessians,
      pressure->fe,
      update_JxW_values|
      update_gradients |
      update_quadrature_points,
      update_JxW_values |
      update_values |
      update_quadrature_points |
      update_normal_vectors,
      *temperature_fe_ptr,
      update_values),
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy(pressure->fe.dofs_per_cell));

  // Compress global data
  poisson_prestep_rhs.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << std::scientific << std::setprecision(6)
           << poisson_prestep_rhs.l2_norm()
           << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_poisson_prestep_rhs
(const typename DoFHandler<dim>::active_cell_iterator        &cell,
 AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim> &scratch,
 AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy         &data)
{
  // Reset local data
  data.local_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Initialize weak forms of the right-hand side's terms
  std::vector<Tensor<1,dim>>  body_force_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  buoyancy_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  coriolis_acceleration_term(scratch.n_q_points);

  // Pressure
  scratch.pressure_fe_values.reinit(cell);

  // Body force
  if (body_force_ptr != nullptr)
    body_force_ptr->value_list(
    scratch.pressure_fe_values.get_quadrature_points(),
    body_force_term);

  // Buoyancy
  if (temperature != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&pressure->get_triangulation(),
                     cell->level(),
                     cell->index(),
                     &temperature->get_dof_handler());

    scratch.temperature_fe_values.reinit(temperature_cell);

    scratch.temperature_fe_values.get_function_values(
      temperature->old_solution,
      scratch.temperature_values);

    AssertThrow(gravity_vector_ptr != nullptr,
                ExcMessage("No unit vector for the gravity has been specified."))

    gravity_vector_ptr->value_list(
      scratch.pressure_fe_values.get_quadrature_points(),
      scratch.gravity_vector_values);

    // Loop over quadrature points
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      buoyancy_term[q] =
        parameters.C3 *
        scratch.gravity_vector_values[q] *
        scratch.temperature_values[q];
  }

  // Coriolis acceleration
  if (angular_velocity_vector_ptr != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell(&velocity->get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &velocity->get_dof_handler());

    scratch.velocity_fe_values.reinit(velocity_cell);

    const FEValuesExtractors::Vector  vector_extractor(0);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_old_solution,
      scratch.velocity_values);

    scratch.angular_velocity_value = angular_velocity_vector_ptr->rotation();

    if constexpr(dim == 2)
      // Loop over quadrature points
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        coriolis_acceleration_term[q] =
          parameters.C1 *
          scratch.angular_velocity_value[0] *
          cross_product_2d(-scratch.velocity_values[q]);
    else if constexpr(dim == 3)
      // Loop over quadrature points
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        coriolis_acceleration_term[q] =
          parameters.C1 *
          cross_product_3d(scratch.angular_velocity_value,
                            scratch.velocity_values[q]);
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.grad_phi[i] = scratch.pressure_fe_values.shape_grad(i, q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      // Local right hand side (Domain integrals)
      data.local_rhs(i) +=
              1.0 / parameters.C6  *
              scratch.grad_phi[i] *
              (body_force_term[q]
               -
               buoyancy_term[q]
               -
               coriolis_acceleration_term[q]) *
              scratch.pressure_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (pressure->get_constraints().is_inhomogeneously_constrained(
        data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) +=
                                    scratch.grad_phi[j] *
                                    scratch.grad_phi[i] *
                                    scratch.pressure_fe_values.JxW(q);
        // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Loop over the faces of the cell
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
      {
        // Pressure
        scratch.pressure_fe_face_values.reinit(cell, face);

        // Velocity
        typename DoFHandler<dim>::active_cell_iterator
        velocity_cell(&velocity->get_triangulation(),
                      cell->level(),
                      cell->index(),
                      &velocity->get_dof_handler());

        typename DoFHandler<dim>::active_face_iterator
        velocity_face(&velocity->get_triangulation(),
                      face->level(),
                      face->index(),
                      &velocity->get_dof_handler());

        const FEValuesExtractors::Vector  vector_extractor(0);

        scratch.velocity_fe_face_values.reinit(velocity_cell, velocity_face);

        scratch.velocity_fe_face_values[vector_extractor].get_function_laplacians(
          velocity->old_solution,
          scratch.velocity_laplacians);

        // Normal vector
        scratch.normal_vectors =
                      scratch.pressure_fe_face_values.get_normal_vectors();

        // Loop over face quadrature points
        for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
          {
            // Extract the test function's values at the face quadrature points
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
              scratch.face_phi[i] =
                          scratch.pressure_fe_face_values.shape_value(i, q);

            // Loop over the degrees of freedom
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            {
              data.local_rhs(i) +=
                              parameters.C2 / parameters.C6 *
                              scratch.face_phi[i] *
                              scratch.velocity_laplacians[q] *
                              scratch.normal_vectors[q] *
                              scratch.pressure_fe_face_values.JxW(q);
            } // Loop over the degrees of freedom
          } // Loop over face quadrature points
      } // Loop over the faces of the cell
}


template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_poisson_prestep_rhs(
  const AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy  &data)
{
  pressure->get_constraints().distribute_local_to_global(
                                data.local_rhs,
                                data.local_dof_indices,
                                poisson_prestep_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

template void RMHD::NavierStokesProjection<2>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<2>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<2>::active_cell_iterator                      &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<2>  &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy        &);
template void RMHD::NavierStokesProjection<3>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<3>::active_cell_iterator                      &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<3>  &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy        &);
template void RMHD::NavierStokesProjection<2>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy &);
