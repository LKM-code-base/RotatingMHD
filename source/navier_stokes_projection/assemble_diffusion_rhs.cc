#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step_rhs()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling diffusion step's right hand side...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Diffusion step - RHS assembly");

  // Reset data
  diffusion_step_rhs  = 0.;

  // Dummy finite element for when there is no bouyancy
  const FE_Q<dim> dummy_fe(1);

  // Create pointer to the pertinent finite element
  const FE_Q<dim> * const temperature_fe_ptr =
          (temperature.get() != nullptr) ? &temperature->fe : &dummy_fe;

  // Polynomial degree of the body force and the neumann function
  const int p_degree_body_force       = velocity->fe_degree;

  const int p_degree_neumann_function = velocity->fe_degree;

  // Compute the highest polynomial degree from all the integrands
  const int p_degree = std::max(3 * velocity->fe_degree - 1,
                                velocity->fe_degree + p_degree_body_force);

  // Compute the highest polynomial degree from all the boundary integrands
  const int face_p_degree = velocity->fe_degree + p_degree_neumann_function;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Initiate the face quadrature formula for exact numerical integration
  const QGauss<dim-1> face_quadrature_formula(std::ceil(0.5 * double(face_p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator                 &cell,
           AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim> &scratch,
           AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy         &data)
    {
      this->assemble_local_diffusion_step_rhs(cell,
                                              scratch,
                                              data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy   &data)
    {
      this->copy_local_to_global_diffusion_step_rhs(data);
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
   AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim>(
     *mapping,
     quadrature_formula,
     face_quadrature_formula,
     velocity->fe,
     update_values|
     update_gradients|
     update_JxW_values|
     update_quadrature_points,
     update_values |
     update_JxW_values |
     update_quadrature_points,
     pressure->fe,
     update_values,
     *temperature_fe_ptr,
     update_values),
   AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy(velocity->fe.dofs_per_cell));

  // Compress global data
  diffusion_step_rhs.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator                 &cell,
 AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim> &scratch,
 AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy         &data)
{
  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // Velocity
  scratch.velocity_fe_values.reinit(cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

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

  scratch.velocity_fe_values[vector_extractor].get_function_divergences(
    velocity->old_solution,
    scratch.old_velocity_divergences);

  scratch.velocity_fe_values[vector_extractor].get_function_divergences(
    velocity->old_old_solution,
    scratch.old_old_velocity_divergences);

  scratch.velocity_fe_values[vector_extractor].get_function_curls(
    velocity->old_solution,
    scratch.old_velocity_curls);

  scratch.velocity_fe_values[vector_extractor].get_function_curls(
    velocity->old_old_solution,
    scratch.old_old_velocity_curls);

  // Pressure
  typename DoFHandler<dim>::active_cell_iterator
  pressure_cell(&velocity->get_triangulation(),
                 cell->level(),
                 cell->index(),
                (pressure->dof_handler).get());

  scratch.pressure_fe_values.reinit(pressure_cell);

  scratch.pressure_fe_values.get_function_values(
    pressure->old_solution,
    scratch.old_pressure_values);

  // Phi
  scratch.pressure_fe_values.get_function_values(
    phi->old_solution,
    scratch.old_phi_values);

  scratch.pressure_fe_values.get_function_values(
    phi->old_old_solution,
    scratch.old_old_phi_values);

  // Temperature and the gravitiy's unit vector.
  if (!flag_ignore_bouyancy_term)
  {
    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&velocity->get_triangulation(),
                     cell->level(),
                     cell->index(),
                     (temperature->dof_handler).get());

    scratch.temperature_fe_values.reinit(temperature_cell);

    scratch.temperature_fe_values.get_function_values(
      temperature->old_solution,
      scratch.old_temperature_values);

    scratch.temperature_fe_values.get_function_values(
      temperature->old_old_solution,
      scratch.old_old_temperature_values);

    Assert(gravity_unit_vector_ptr != nullptr,
           ExcMessage("No unit vector for the gravity has been specified."))

    gravity_unit_vector_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_unit_vector_values);
  }
  else
  {
    ZeroFunction<dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.old_temperature_values);

    scratch.old_old_temperature_values = scratch.old_temperature_values;

    for (auto &gravity_unit_vector : scratch.gravity_unit_vector_values)
      gravity_unit_vector = Tensor<1, dim>();
  }

  // Body force
  /*! @note Should body forces be also treated inside the VSIMEX scheme?
      Just a thought, as they are given functions that do not depend
      (in our code formulation) on the fields.*/
  if (body_force_ptr != nullptr)
  {
    body_force_ptr->set_time(time_stepping.get_previous_time());
    body_force_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.old_old_body_force_values);

    body_force_ptr->set_time(time_stepping.get_current_time());
    body_force_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.old_body_force_values);

    body_force_ptr->set_time(time_stepping.get_next_time());
    body_force_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.body_force_values);
  }
  else
  {
    ZeroTensorFunction<1,dim>().value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.body_force_values);

    scratch.old_body_force_values     = scratch.body_force_values;
    scratch.old_old_body_force_values = scratch.body_force_values;
  }

  // Coreolis acceleration
  if (angular_velocity_unit_vector_ptr != nullptr)
  {
    if constexpr(dim == 2)
    {
      const std::vector<Tensor<1,1>> z_unit_vectors(
        scratch.old_angular_velocity_values.size(),
        Tensor<1,1>({1.0}));

      scratch.old_angular_velocity_values     = z_unit_vectors;
      scratch.old_old_angular_velocity_values = z_unit_vectors;
    }
    else if constexpr(dim == 3)
    {
      angular_velocity_unit_vector_ptr->set_time(time_stepping.get_previous_time());
      angular_velocity_unit_vector_ptr->value_list(
        scratch.velocity_fe_values.get_quadrature_points(),
        scratch.old_old_angular_velocity_values);

      angular_velocity_unit_vector_ptr->set_time(time_stepping.get_current_time());
      angular_velocity_unit_vector_ptr->value_list(
        scratch.velocity_fe_values.get_quadrature_points(),
        scratch.old_old_angular_velocity_values);
    }
  }
  else
  {
    if constexpr(dim == 2)
    {
      const std::vector<Tensor<1,1>> zero_vectors(
        scratch.old_angular_velocity_values.size(),
        Tensor<1,1>());

      scratch.old_angular_velocity_values     = zero_vectors;
      scratch.old_old_angular_velocity_values = zero_vectors;
    }
    else if constexpr(dim == 3)
    {
      ZeroTensorFunction<1,dim>(dim).value_list(
        scratch.velocity_fe_values.get_quadrature_points(),
        scratch.old_angular_velocity_values);

      scratch.old_old_angular_velocity_values =
                                    scratch.old_angular_velocity_values;
    }
  }

  // VSIMEX coefficients
  const std::vector<double> alpha = time_stepping.get_alpha();
  const std::vector<double> beta  = time_stepping.get_beta();
  const std::vector<double> gamma = time_stepping.get_gamma();

  // Data for the elimination of the selonoidal velocity
  const std::vector<double> old_alpha_zero  = time_stepping.get_old_alpha_zero();
  const std::vector<double> old_step_size   = time_stepping.get_old_step_size();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = time_stepping.get_eta();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.velocity_fe_values[vector_extractor].value(i,q);
      scratch.grad_phi[i] = scratch.velocity_fe_values[vector_extractor].gradient(i,q);
      scratch.div_phi[i]  = scratch.velocity_fe_values[vector_extractor].divergence(i,q);
      scratch.curl_phi[i] = scratch.velocity_fe_values[vector_extractor].curl(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      // Local right hand side (Domain integrals)
      data.local_rhs(i) +=
                (scratch.div_phi[i] *
                 (scratch.old_pressure_values[q]
                  -
                  old_step_size[0] / time_stepping.get_next_step_size() *
                  alpha[1] / old_alpha_zero[0] *
                  scratch.old_phi_values[q]
                  -
                  old_step_size[1] / time_stepping.get_next_step_size() *
                  alpha[2] / old_alpha_zero[1] *
                  scratch.old_old_phi_values[q])
                -
                scratch.phi[i] *
                (alpha[1] / time_stepping.get_next_step_size() *
                 scratch.old_velocity_values[q]
                 +
                 alpha[2] / time_stepping.get_next_step_size() *
                 scratch.old_old_velocity_values[q])
                -
                beta[0] *
                scratch.phi[i] *
                scratch.gravity_unit_vector_values[q] *
                scratch.old_temperature_values[q]
                -
                beta[1] *
                scratch.phi[i] *
                scratch.gravity_unit_vector_values[q] *
                scratch.old_old_temperature_values[q]
                +
                gamma[0] *
                scratch.phi[i] *
                scratch.body_force_values[q]
                -
                gamma[1] *
                (parameters.C2 *
                 scalar_product(scratch.grad_phi[i],
                                scratch.old_velocity_gradients[q])
                 -
                 scratch.phi[i] *
                 scratch.old_body_force_values[q])
                -
                gamma[2] *
                (parameters.C2 *
                 scalar_product(scratch.grad_phi[i],
                                scratch.old_old_velocity_gradients[q])
                 -
                 scratch.phi[i] *
                 scratch.old_old_body_force_values[q])) *
                scratch.velocity_fe_values.JxW(q);

      if (angular_velocity_unit_vector_ptr != nullptr)
      {
        if constexpr(dim == 2)
          data.local_rhs(i) -=
                (beta[0] *
                 parameters.C1 *
                 scratch.old_angular_velocity_values[q][0] *
                 cross_product_2d(scratch.phi[i]) *
                 scratch.old_velocity_values[q]
                 +
                 beta[1] *
                 parameters.C1 *
                 scratch.old_old_angular_velocity_values[q][0] *
                 cross_product_2d(scratch.phi[i]) *
                 scratch.old_old_velocity_values[q]) *
                scratch.velocity_fe_values.JxW(q);
        else if constexpr(dim == 3)
          data.local_rhs(i) -=
                (beta[0] *
                 parameters.C1 *
                 scratch.phi[i] *
                 cross_product_3d(scratch.old_angular_velocity_values[q],
                                  scratch.old_velocity_values[q])
                 +
                 beta[1] *
                 parameters.C1 *
                 scratch.phi[i] *
                 cross_product_3d(scratch.old_old_angular_velocity_values[q],
                                  scratch.old_old_velocity_values[q]))*
                scratch.velocity_fe_values.JxW(q);
      }

      if (parameters.convective_term_time_discretization ==
          RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit)
        switch (parameters.convective_term_weak_form)
        {
          case RunTimeParameters::ConvectiveTermWeakForm::standard:
          {
            data.local_rhs(i) -=
                (beta[0] *
                 scratch.phi[i] *
                 scratch.old_velocity_gradients[q] *
                 scratch.old_velocity_values[q]
                 +
                 beta[1] *
                 scratch.phi[i] *
                 scratch.old_old_velocity_gradients[q] *
                 scratch.old_old_velocity_values[q]) *
                scratch.velocity_fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
          {
            data.local_rhs(i) -=
                (beta[0] *
                 (scratch.phi[i] *
                  scratch.old_velocity_gradients[q] *
                  scratch.old_velocity_values[q]
                  +
                  0.5 *
                  scratch.old_velocity_divergences[q] *
                  scratch.phi[i] *
                  scratch.old_velocity_values[q])
                 +
                 beta[1] *
                 (scratch.phi[i] *
                  scratch.old_old_velocity_gradients[q] *
                  scratch.old_old_velocity_values[q]
                  +
                  0.5 *
                  scratch.old_old_velocity_divergences[q] *
                  scratch.phi[i] *
                  scratch.old_old_velocity_values[q])) *
                scratch.velocity_fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectiveTermWeakForm::divergence:
          {
            data.local_rhs(i) -=
                (beta[0] *
                 (scratch.phi[i] *
                  scratch.old_velocity_gradients[q] *
                  scratch.old_velocity_values[q]
                  +
                  scratch.old_velocity_divergences[q] *
                  scratch.phi[i] *
                  scratch.old_velocity_values[q])
                 +
                 beta[1] *
                 (scratch.phi[i] *
                  scratch.old_old_velocity_gradients[q] *
                  scratch.old_old_velocity_values[q]
                  +
                  scratch.old_old_velocity_divergences[q] *
                  scratch.phi[i] *
                  scratch.old_old_velocity_values[q])) *
                scratch.velocity_fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectiveTermWeakForm::rotational:
          {
            // The minus sign in the argument of cross_product_2d
            // method is due to how the method is defined.
            if constexpr(dim == 2)
              data.local_rhs(i) -=
                (beta[0] *
                 scratch.phi[i] *
                 scratch.old_velocity_curls[q][0] *
                 cross_product_2d(-scratch.old_velocity_values[q])
                 +
                 beta[1] *
                 scratch.phi[i] *
                 scratch.old_old_velocity_curls[q][0] *
                 cross_product_2d(-scratch.old_old_velocity_values[q])) *
                scratch.velocity_fe_values.JxW(q);
            else if constexpr(dim == 3)
              data.local_rhs(i) -=
                (beta[0] *
                 scratch.phi[i] *
                 cross_product_3d(
                   scratch.old_velocity_curls[q],
                   scratch.old_velocity_values[q])
                 +
                 beta[1] *
                 scratch.phi[i] *
                 cross_product_3d(
                   scratch.old_old_velocity_curls[q],
                   scratch.old_old_velocity_values[q])) *
                scratch.velocity_fe_values.JxW(q);
            break;
          }
          default:
            Assert(false, ExcNotImplemented());
        };

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (velocity->constraints.is_inhomogeneously_constrained(
            data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bc(j,i) +=
                (alpha[0] / time_stepping.get_next_step_size() *
                 scratch.phi[j] *
                 scratch.phi[i]
                 +
                 gamma[0] * parameters.C2 *
                 scalar_product(scratch.grad_phi[j],
                                scratch.grad_phi[i])) *
                scratch.velocity_fe_values.JxW(q);

          if (parameters.convective_term_time_discretization ==
              RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
            switch (parameters.convective_term_weak_form)
            {
              case RunTimeParameters::ConvectiveTermWeakForm::standard:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      (scratch.phi[j] *
                       scratch.grad_phi[i] *
                       (eta[0] *
                        scratch.old_velocity_values[q]
                        +
                        eta[1] *
                        scratch.old_old_velocity_values[q])) *
                      scratch.velocity_fe_values.JxW(q);
                break;
              }
              case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      (scratch.phi[j] *
                       scratch.grad_phi[i] *
                       (eta[0] *
                        scratch.old_velocity_values[q]
                        +
                        eta[1] *
                        scratch.old_old_velocity_values[q])
                       +
                       0.5 *
                       (eta[0] *
                        scratch.old_velocity_divergences[q]
                        +
                        eta[1] *
                        scratch.old_old_velocity_divergences[q]) *
                       scratch.phi[j] *
                       scratch.phi[i]) *
                      scratch.velocity_fe_values.JxW(q);
                break;
              }
              case RunTimeParameters::ConvectiveTermWeakForm::divergence:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      (scratch.phi[j] *
                       scratch.grad_phi[i] *
                       (eta[0] *
                        scratch.old_velocity_values[q]
                        +
                        eta[1] *
                        scratch.old_old_velocity_values[q])
                       +
                       (eta[0] *
                        scratch.old_velocity_divergences[q]
                        +
                        eta[1] *
                        scratch.old_old_velocity_divergences[q]) *
                       scratch.phi[j] *
                       scratch.phi[i]) *
                      scratch.velocity_fe_values.JxW(q);
                break;
              }
              case RunTimeParameters::ConvectiveTermWeakForm::rotational:
              {
                // The minus sign in the argument of cross_product_2d
                // method is due to how the method is defined.
                if constexpr(dim == 2)
                  data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      (scratch.phi[j] *
                       scratch.curl_phi[i][0] *
                       cross_product_2d(
                        -eta[0] *
                        scratch.old_velocity_values[q]
                        -
                        eta[1] *
                        scratch.old_old_velocity_values[q])) *
                      scratch.velocity_fe_values.JxW(q);
                else if constexpr(dim == 3)
                  data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      (scratch.phi[j] *
                       cross_product_3d(
                         scratch.curl_phi[i],
                         eta[0] *
                         scratch.old_velocity_values[q]
                         +
                         eta[1] *
                         scratch.old_old_velocity_values[q])) *
                      scratch.velocity_fe_values.JxW(q);
                break;
              }
              default:
                Assert(false, ExcNotImplemented());
            };
        } // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Loop over the faces of the cell
  for (const auto &face : cell->face_iterators())
    if (face->at_boundary() &&
        velocity->boundary_conditions.neumann_bcs.find(face->boundary_id())
        != velocity->boundary_conditions.neumann_bcs.end())
    {
      // Neumann boundary condition
      scratch.velocity_fe_face_values.reinit(cell, face);

      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
        time_stepping.get_previous_time());
      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
        scratch.velocity_fe_face_values.get_quadrature_points(),
        scratch.old_old_neumann_bc_values);

      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
        time_stepping.get_current_time());
      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
        scratch.velocity_fe_face_values.get_quadrature_points(),
        scratch.old_neumann_bc_values);

      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
        time_stepping.get_next_time());
      velocity->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
        scratch.velocity_fe_face_values.get_quadrature_points(),
        scratch.neumann_bc_values);

      // Loop over face quadrature points
      for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
      {
        // Extract the test function's values at the face quadrature points
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          scratch.face_phi[i] =
            scratch.velocity_fe_face_values[vector_extractor].value(i,q);

        // Loop over the degrees of freedom
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          data.local_rhs(i) +=
            scratch.face_phi[i] * (
              gamma[0] *
              scratch.neumann_bc_values[q]
              +
              gamma[1] *
              scratch.old_neumann_bc_values[q]
              +
              gamma[2] *
              scratch.old_old_neumann_bc_values[q]) *
            scratch.velocity_fe_face_values.JxW(q);
      } // Loop over face quadrature points
    } // Loop over the faces of the cell
} // assemble_local_diffusion_step_rhs

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_diffusion_step_rhs
(const AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy &data)
{
  velocity->constraints.distribute_local_to_global(
                                data.local_rhs,
                                data.local_dof_indices,
                                diffusion_step_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<2>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<2> &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<3>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<3> &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy       &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy &);
