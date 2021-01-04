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
    *pcout << "  Navier Stokes: Assembling poisson pre-step's right hand side...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Poisson pre-step - RHS assembly");

  // Reset data
  poisson_prestep_rhs = 0.;

  // Dummy finite element for when there is no bouyancy
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
           AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>  &scratch,
           AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<dim>    &data)
    {
      this->assemble_local_poisson_prestep_rhs(cell, 
                                               scratch,
                                               data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<dim> &data) 
    {
      this->copy_local_to_global_poisson_prestep_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               (pressure->dof_handler)->begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               (pressure->dof_handler)->end()),
    worker,
    copier,
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>(
      *mapping,
      quadrature_formula,
      face_quadrature_formula,
      velocity->fe,
      update_values |
      update_gradients,
      update_values |
      update_hessians,
      pressure->fe,
      update_JxW_values|
      update_values|
      update_gradients |
      update_quadrature_points,
      update_JxW_values |
      update_values |
      update_quadrature_points |
      update_normal_vectors,
      *temperature_fe_ptr,
      update_values |
      update_gradients,
      update_values),
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<dim>(
      pressure->fe.dofs_per_cell));
  
  // Compress global data
  poisson_prestep_rhs.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_poisson_prestep_rhs
(const typename DoFHandler<dim>::active_cell_iterator        &cell,
 AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>     &scratch,
 AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<dim>       &data)
{
  // Reset local data
  data.local_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // Pressure
  scratch.pressure_fe_values.reinit(cell);

  // Velocity
  typename DoFHandler<dim>::active_cell_iterator
  velocity_cell(&velocity->get_triangulation(),
                 cell->level(),
                 cell->index(),
                velocity->dof_handler.get());

  scratch.velocity_fe_values.reinit(velocity_cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  /*scratch.velocity_fe_values[vector_extractor].get_function_values(
    velocity->old_old_solution,
    scratch.velocity_values);

  scratch.velocity_fe_values[vector_extractor].get_function_curls(
    velocity->old_old_solution,
    scratch.velocity_curls);*/

  // Temperature 
  if (!flag_ignore_bouyancy_term)
  {
    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&pressure->get_triangulation(),
                     cell->level(),
                     cell->index(),
                     (temperature->dof_handler).get());

    scratch.temperature_fe_values.reinit(temperature_cell);

    
    scratch.temperature_fe_values.get_function_values(
      temperature->old_old_solution,
      scratch.temperature_values);

    scratch.temperature_fe_values.get_function_gradients(
      temperature->old_old_solution,
      scratch.temperature_gradients);

    Assert(gravity_unit_vector_ptr != nullptr,
           ExcMessage("No unit vector for the gravity has been specified."))
    
    gravity_unit_vector_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_unit_vector_values);

    gravity_unit_vector_ptr->divergence_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_unit_vector_divergences);
  }
  else
  {
    ZeroFunction<dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.temperature_values);

    ZeroTensorFunction<1,dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.temperature_gradients);

    ZeroTensorFunction<1,dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_unit_vector_values);

    ZeroFunction<dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_unit_vector_divergences);
  }

  // Body force
  if (body_force_ptr != nullptr)
    body_force_ptr->divergence_list(
      scratch.pressure_fe_values.get_quadrature_points(),
      scratch.body_force_divergences);
  else
    ZeroFunction<dim>().value_list(
      scratch.pressure_fe_values.get_quadrature_points(),
      scratch.body_force_divergences);

  // Rotation
  /*if (rotation_ptr != nullptr)
  {

  }
  else
  {
    
  }*/

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);
  
  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.phi[i] = scratch.pressure_fe_values.shape_value(i, q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      // Local right hand side (Domain integrals)
      data.local_rhs(i) -= 
              scratch.phi[i] *
              (scratch.body_force_divergences[q]
               -
               // parameters.C2 * 
               (scratch.phi[i] *
                scratch.temperature_gradients[q] *
                scratch.gravity_unit_vector_values[q]
                +
                scratch.temperature_values[q] *
                scratch.gravity_unit_vector_divergences[q])
               /*+
               -
               - 1.0 * // parameters.C3
               (scratch.rotation_curls[q] *
                scratch.velocity_values[q]
                -
                scratch.velocity_curls[q] *
                scratch.)*/) *
              scratch.pressure_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (pressure->constraints.is_inhomogeneously_constrained(
        data.local_dof_indices[i]))
      {
        // Extract test function values at the quadrature points
        for (unsigned int k = 0; k < scratch.dofs_per_cell; ++k)
          scratch.grad_phi[k] =
                          scratch.pressure_fe_values.shape_grad(k, q);

        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) +=
                                    scratch.grad_phi[j] *
                                    scratch.grad_phi[i] *
                                    scratch.pressure_fe_values.JxW(q);
      } // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Loop over the faces of the cell
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
                    velocity->dof_handler.get());

      typename DoFHandler<dim>::active_face_iterator
      velocity_face(&velocity->get_triangulation(),
                     face->level(),
                     face->index(),
                    velocity->dof_handler.get());

      scratch.velocity_fe_face_values.reinit(velocity_cell, velocity_face);

      scratch.velocity_fe_face_values[vector_extractor].get_function_values(
        velocity->old_old_solution,
        scratch.velocity_face_values);

      scratch.velocity_fe_face_values[vector_extractor].get_function_laplacians(
                                    velocity->old_old_solution,
                                    scratch.velocity_laplacians);

      // Temperature
      if (!flag_ignore_bouyancy_term)
      {
        typename DoFHandler<dim>::active_cell_iterator
        temperature_cell(&pressure->get_triangulation(),
                         cell->level(),
                         cell->index(),
                         (temperature->dof_handler).get());

        typename DoFHandler<dim>::active_face_iterator
        temperature_face(&velocity->get_triangulation(),
                         face->level(),
                         face->index(),
                         temperature->dof_handler.get());

        scratch.temperature_fe_face_values.reinit(temperature_cell,
                                                  temperature_face);

        scratch.temperature_fe_face_values.get_function_values(
          temperature->old_old_solution,
          scratch.temperature_face_values);

        Assert(gravity_unit_vector_ptr != nullptr,
              ExcMessage("No unit vector for the gravity has been specified."))
        
        gravity_unit_vector_ptr->value_list(
          scratch.temperature_fe_face_values.get_quadrature_points(),
          scratch.gravity_unit_vector_face_values);
      }
      else
      {
        ZeroFunction<dim>().value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.temperature_face_values);
      
        ZeroTensorFunction<1, dim>().value_list(
          scratch.temperature_fe_face_values.get_quadrature_points(),
          scratch.gravity_unit_vector_face_values);
      }

      // Body force
      if (body_force_ptr != nullptr)
        body_force_ptr->value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.body_force_values);
      else
        ZeroTensorFunction<1,dim>().value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.body_force_values);

      // Coriolis acceleration
      /*if (rotation_ptr != nullptr)
        rotation_ptr->value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.rotation_face_values);
      else
        ZeroTensorFunction<1,dim>().value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.rotation_face_values);*/
      
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
                            scratch.face_phi[i] * 
                            (1.0 / parameters.Re * // parameters.C1
                             scratch.velocity_laplacians[q]
                             +
                             scratch.body_force_values[q]
                             -
                             // parameters.C2
                             scratch.temperature_face_values[q] *
                             scratch.gravity_unit_vector_face_values[q])*
                            scratch.normal_vectors[q] *
                            scratch.pressure_fe_face_values.JxW(q);  
            if constexpr(dim == 2)
              data.local_rhs(i) -=
                            0.0 * /* parameters.C3 * 
                            scratch.face_phi[i] **/
                            scratch.pressure_fe_face_values.JxW(q);
            else if constexpr(dim == 3)
              data.local_rhs(i) -=
                            0.0 * /* parameters.C3 *
                            scratch.face_phi[i] *
                            cross_product_3d(scratch.rotation_values[q],
                                             scratch.velocity_values[q]) * */
                            scratch.pressure_fe_face_values.JxW(q);
          } // Loop over the degrees of freedom
        } // Loop over face quadrature points
    } // Loop over the faces of the cell
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_poisson_prestep_rhs(
  const AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<dim>  &data)
{
  pressure->constraints.distribute_local_to_global(
                                data.local_rhs,
                                data.local_dof_indices,
                                poisson_prestep_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

template void RMHD::NavierStokesProjection<2>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<2>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<2>::active_cell_iterator              &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<2>     &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<2>       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<3>::active_cell_iterator              &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<3>     &,
    RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<3>       &);
template void RMHD::NavierStokesProjection<2>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<2> &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy<3> &);
