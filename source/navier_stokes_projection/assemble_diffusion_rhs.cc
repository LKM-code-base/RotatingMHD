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

  // Dummy finite element for when there is no buoyancy
  const FE_Q<dim> dummy_fe(1);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim> * const temperature_fe_ptr =
          (temperature.get() != nullptr) ? &temperature->get_finite_element() : &dummy_fe;

  // Polynomial degree of the body force and the Neumann function
  const int p_degree_body_force       = velocity->fe_degree();

  const int p_degree_neumann_function = velocity->fe_degree();

  // Compute the highest polynomial degree from all the integrands
  const int p_degree = std::max(3 * velocity->fe_degree() - 1,
                                velocity->fe_degree() + p_degree_body_force);

  // Compute the highest polynomial degree from all the boundary integrands
  const int face_p_degree = velocity->fe_degree() + p_degree_neumann_function;

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
              velocity->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity->get_dof_handler().end()),
   worker,
   copier,
   AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim>(
     *mapping,
     quadrature_formula,
     face_quadrature_formula,
     velocity->get_finite_element(),
     update_values|
     update_gradients|
     update_JxW_values|
     update_quadrature_points,
     update_values |
     update_JxW_values |
     update_quadrature_points,
     pressure->get_finite_element(),
     update_values,
     *temperature_fe_ptr,
     update_values),
   AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy(velocity->get_finite_element().dofs_per_cell));

  // Compress global data
  diffusion_step_rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  norm_diffusion_rhs = diffusion_step_rhs.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << std::scientific << std::setprecision(6)
           << norm_diffusion_rhs
           << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator                 &cell,
 AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim> &scratch,
 AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy         &data)
{
  const typename Entities::VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity->get_neumann_boundary_conditions();

  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // VSIMEX coefficients
  const std::vector<double> alpha = time_stepping.get_alpha();
  const std::vector<double> beta  = time_stepping.get_beta();
  const std::vector<double> gamma = time_stepping.get_gamma();

  // Data for the elimination of the solenoidal velocity
  const std::vector<double> old_alpha_zero  = time_stepping.get_previous_alpha_zeros();
  const std::vector<double> old_step_size   = time_stepping.get_previous_step_sizes();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = time_stepping.get_eta();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Initialize weak forms of the right-hand side's terms
  std::vector<Tensor<1,dim>>  acceleration_term(scratch.n_q_points);
  std::vector<double>         pressure_gradient_term(scratch.n_q_points);
  std::vector<Tensor<2,dim>>  diffusion_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  body_force_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  buoyancy_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  coriolis_acceleration_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  advection_term(scratch.n_q_points);

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
                &pressure->get_dof_handler());

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

  // Body force
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

    // Loop over quadrature points
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      body_force_term[q] =
        1.0 * /* parameters.C7 */
        (gamma[0] *
         scratch.body_force_values[q]
         +
         gamma[1] *
         scratch.old_body_force_values[q]
         +
         gamma[2] *
         scratch.old_old_body_force_values[q]);
  }

  // Buoyancy
  if (temperature != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&velocity->get_triangulation(),
                     cell->level(),
                     cell->index(),
                     &temperature->get_dof_handler());

    scratch.temperature_fe_values.reinit(temperature_cell);

    scratch.temperature_fe_values.get_function_values(
      temperature->old_solution,
      scratch.old_temperature_values);

    scratch.temperature_fe_values.get_function_values(
      temperature->old_old_solution,
      scratch.old_old_temperature_values);

    Assert(gravity_vector_ptr != nullptr,
           ExcMessage("No unit vector for the gravity has been specified."))

    gravity_vector_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.gravity_vector_values);

    // Loop over quadrature points
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      buoyancy_term[q] =
        parameters.C3 *
        scratch.gravity_vector_values[q] *
        (beta[0] *
         scratch.old_temperature_values[q]
         +
         beta[1] *
         scratch.old_old_temperature_values[q]);
  }

  // Coriolis acceleration
  if (angular_velocity_vector_ptr != nullptr)
  {
    angular_velocity_vector_ptr->set_time(time_stepping.get_previous_time());
    scratch.old_old_angular_velocity_value =
                                angular_velocity_vector_ptr->rotation();

    angular_velocity_vector_ptr->set_time(time_stepping.get_current_time());
    scratch.old_angular_velocity_value =
                                angular_velocity_vector_ptr->rotation();

    if constexpr(dim == 2)
      // Loop over quadrature points
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        coriolis_acceleration_term[q] =
          parameters.C1 *
          (beta[0] *
           scratch.old_angular_velocity_value[0] *
           cross_product_2d(-scratch.old_velocity_values[q])
           +
           beta[1] *
           scratch.old_old_angular_velocity_value[0] *
           cross_product_2d(-scratch.old_old_velocity_values[q]));
    else if constexpr(dim == 3)
      // Loop over quadrature points
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        coriolis_acceleration_term[q] =
          parameters.C1 *
          (beta[0] *
           cross_product_3d(scratch.old_angular_velocity_value,
                            scratch.old_velocity_values[q])
           +
           beta[1] *
           cross_product_3d(scratch.old_old_angular_velocity_value,
                            scratch.old_old_velocity_values[q]));
  }

  // Advection term
  if (parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit)
  {
    switch (parameters.convective_term_weak_form)
    {
      case RunTimeParameters::ConvectiveTermWeakForm::standard:
      {
        // Loop over quadrature points
        for (unsigned int q = 0; q < scratch.n_q_points; ++q)
          advection_term[q] =
                beta[0] *
                scratch.old_velocity_gradients[q] *
                scratch.old_velocity_values[q]
                +
                beta[1] *
                scratch.old_old_velocity_gradients[q] *
                scratch.old_old_velocity_values[q];
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
      {
        // Loop over quadrature points
        for (unsigned int q = 0; q < scratch.n_q_points; ++q)
          advection_term[q] =
                beta[0] *
                (scratch.old_velocity_gradients[q] *
                 scratch.old_velocity_values[q]
                 +
                 0.5 *
                 scratch.old_velocity_divergences[q] *
                 scratch.old_velocity_values[q])
                +
                beta[1] *
                (scratch.old_old_velocity_gradients[q] *
                 scratch.old_old_velocity_values[q]
                 +
                 0.5 *
                 scratch.old_old_velocity_divergences[q] *
                 scratch.old_old_velocity_values[q]);
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::divergence:
      {
        // Loop over quadrature points
        for (unsigned int q = 0; q < scratch.n_q_points; ++q)
          advection_term[q] =
                beta[0] *
                (scratch.old_velocity_gradients[q] *
                 scratch.old_velocity_values[q]
                 +
                 scratch.old_velocity_divergences[q] *
                 scratch.old_velocity_values[q])
                +
                beta[1] *
                (scratch.old_old_velocity_gradients[q] *
                 scratch.old_old_velocity_values[q]
                 +
                 scratch.old_old_velocity_divergences[q] *
                 scratch.old_old_velocity_values[q]);
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::rotational:
      {
        // The minus sign in the argument of cross_product_2d
        // method is due to how the method is defined.
        if constexpr(dim == 2)
           // Loop over quadrature points
          for (unsigned int q = 0; q < scratch.n_q_points; ++q)
            advection_term[q] =
                beta[0] *
                scratch.old_velocity_curls[q][0] *
                cross_product_2d(-scratch.old_velocity_values[q])
                +
                beta[1] *
                scratch.old_old_velocity_curls[q][0] *
                cross_product_2d(-scratch.old_old_velocity_values[q]);
        else if constexpr(dim == 3)
          // Loop over quadrature points
          for (unsigned int q = 0; q < scratch.n_q_points; ++q)
            advection_term[q] =
                beta[0] *
                cross_product_3d(
                  scratch.old_velocity_curls[q],
                  scratch.old_velocity_values[q])
                +
                beta[1] *
                cross_product_3d(
                  scratch.old_old_velocity_curls[q],
                  scratch.old_old_velocity_values[q]);
        break;
      }
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Evaluate the weak form of the right-hand side's terms at
    // the quadrature point
    acceleration_term[q] =
              alpha[1] / time_stepping.get_next_step_size() *
              scratch.old_velocity_values[q]
              +
              alpha[2] / time_stepping.get_next_step_size() *
              scratch.old_old_velocity_values[q];

    pressure_gradient_term[q] =
              parameters.C6 *
              (scratch.old_pressure_values[q]
               -
               old_step_size[0] / time_stepping.get_next_step_size() *
               alpha[1] / old_alpha_zero[0] *
               scratch.old_phi_values[q]
               -
               old_step_size[1] / time_stepping.get_next_step_size() *
               alpha[2] / old_alpha_zero[1] *
               scratch.old_old_phi_values[q]);

    diffusion_term[q] =
              parameters.C2 *
              (gamma[1] *
               scratch.old_velocity_gradients[q]
               +
               gamma[2] *
               scratch.old_old_velocity_gradients[q]);

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
      data.local_rhs(i) +=
                    (scratch.div_phi[i] *
                     pressure_gradient_term[q]
                     -
                     scalar_product(scratch.grad_phi[i],
                                    diffusion_term[q])
                     +
                     scratch.phi[i] *
                     (body_force_term[q]
                      -
                      buoyancy_term[q]
                      -
                      coriolis_acceleration_term[q]
                      -
                      acceleration_term[q]
                      -
                      advection_term[q])) *
                    scratch.velocity_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (velocity->get_constraints().is_inhomogeneously_constrained(
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
  if (!neumann_bcs.empty())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
        {
          // Neumann boundary condition
          scratch.velocity_fe_face_values.reinit(cell, face);

          const types::boundary_id  boundary_id{face->boundary_id()};
          const std::vector<Point<dim>> face_quadrature_points{scratch.velocity_fe_face_values.get_quadrature_points()};

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_previous_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
                                                  scratch.old_old_neumann_bc_values);

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_current_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
                                                  scratch.old_neumann_bc_values);

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_next_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
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
  velocity->get_constraints().distribute_local_to_global(
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
