#include <rotatingMHD/magnetic_induction.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/filtered_iterator.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_diffusion_step()
{
  /* System matrix setup */

  /* This if scope makes sure that if the time step did not change
     between solve calls, the following matrix summation is only done once */
  if (this->time_stepping.coefficients_changed() == true ||
      this->flag_matrices_were_updated)
  {
    TimerOutput::Scope  t(*this->computing_timer, "Magnetic induction: Mass and stiffness matrix addition");
    this->diffusion_step_mass_plus_stiffness_matrix = 0.;

    this->diffusion_step_mass_plus_stiffness_matrix.add
    (this->time_stepping.get_alpha()[0] / this->time_stepping.get_next_step_size(),
     this->diffusion_step_mass_matrix);

    this->diffusion_step_mass_plus_stiffness_matrix.add
    (this->time_stepping.get_gamma()[0] * parameters.C,
     this->diffusion_step_stiffness_matrix);
  }

  /* In case of a semi-implicit scheme, the advection matrix has to be
  assembled and added to the system matrix */
  if (parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
  {
    assemble_advection_matrix();
    this->diffusion_step_stiffness_matrix.copy_from(this->diffusion_step_mass_plus_stiffness_matrix);
    this->diffusion_step_stiffness_matrix.add(1.0, this->diffusion_step_advection_matrix);
  }
  /* Right hand side setup */
  assemble_diffusion_step_rhs();
}



template <int dim>
void MagneticInduction<dim>::assemble_diffusion_step_rhs()
{
  if (parameters.verbose)
    *this->pcout << "  Magnetic induction: Assembling diffusion step's right hand side...";

  TimerOutput::Scope  t(*this->computing_timer,
                        "Magnetic induction: Diffusion step - RHS assembly");

  // Reset data
  this->diffusion_step_rhs  = 0.;

  // Dummy finite element for when the velocity is given by a function
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(2), dim);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim>* const velocity_fe =
              (velocity != nullptr) ? &velocity->get_finite_element() : &dummy_fe_system;

    // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(magnetic_field->fe_degree() + 2);

  // Initiate the face quadrature formula for exact numerical integration
  const QGauss<dim-1> face_quadrature_formula(magnetic_field->fe_degree() + 2);

  // Set up the lambda function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator             &cell,
           AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<dim>  &scratch,
           AssemblyData::MagneticInduction::DiffusionStepRHS::Copy          &data)
    {
      this->assemble_local_diffusion_step_rhs(cell,
                                              scratch,
                                              data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::MagneticInduction::DiffusionStepRHS::Copy  &data)
    {
      this->copy_local_to_global_diffusion_step_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags magnetic_field_update_flags =
    update_values|
    update_gradients|
    update_quadrature_points|
    update_JxW_values;
  const UpdateFlags pseudo_pressure_update_flags =
    update_values;
  const UpdateFlags velocity_update_flags =
    update_values|
    update_gradients;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              magnetic_field->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              magnetic_field->get_dof_handler().end()),
   worker,
   copier,
   AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<dim>(
     *this->mapping,
     quadrature_formula,
     magnetic_field->get_finite_element(),
     magnetic_field_update_flags,
     pseudo_pressure->get_finite_element(),
     pseudo_pressure_update_flags,
     *velocity_fe,
     velocity_update_flags),
   AssemblyData::MagneticInduction::DiffusionStepRHS::Copy(
     magnetic_field->get_finite_element().dofs_per_cell));

  // Compress global data
  this->diffusion_step_rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  this->norm_diffusion_step_rhs = this->diffusion_step_rhs.l2_norm();

  if (parameters.verbose)
    *this->pcout << " done!" << std::endl
                 << "    Right-hand side's L2-norm = "
                 << std::scientific << std::setprecision(6)
                 << this->norm_diffusion_step_rhs
                 << std::endl;
}



template <int dim>
void MagneticInduction<dim>::assemble_local_diffusion_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator            &cell,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<dim> &scratch,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Copy         &data)
{
  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // VSIMEX coefficients
  const std::vector<double> alpha = this->time_stepping.get_alpha();
  const std::vector<double> beta  = this->time_stepping.get_beta();
  const std::vector<double> gamma = this->time_stepping.get_gamma();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = this->time_stepping.get_eta();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  using curl_type = typename FEValuesViews::Vector< dim >::curl_type;

  // Initialize weak forms of the right-hand side's terms
  std::vector<Tensor<1,dim>>  time_derivative_term(
                                scratch.n_q_points,
                                Tensor<1,dim>());
  std::vector<Tensor<1,dim>>  advection_term(
                                scratch.n_q_points,
                                Tensor<1,dim>());
  std::vector<Tensor<1,dim>>  pseudo_pressure_gradient_term(
                                scratch.n_q_points,
                                Tensor<1,dim>());
  std::vector<curl_type>      diffusion_term(
                                scratch.n_q_points,
                                curl_type());
  // Vector extractor
  const FEValuesExtractors::Vector  vector_extractor(0);

  // Magnetic induction
  scratch.magnetic_field_fe_values.reinit(cell);

  scratch.magnetic_field_fe_values[vector_extractor].get_function_values(
    magnetic_field->old_solution,
    scratch.old_magnetic_field_values);

  scratch.magnetic_field_fe_values[vector_extractor].get_function_values(
    magnetic_field->old_old_solution,
    scratch.old_old_magnetic_field_values);

  scratch.magnetic_field_fe_values[vector_extractor].get_function_curls(
    magnetic_field->old_solution,
    scratch.old_magnetic_field_curls);

  scratch.magnetic_field_fe_values[vector_extractor].get_function_curls(
    magnetic_field->old_old_solution,
    scratch.old_old_magnetic_field_curls);

  // Pseudo-pressure and auxiliary scalar
  typename DoFHandler<dim>::active_cell_iterator
    pseudo_pressure_cell(&pseudo_pressure->get_triangulation(),
                         cell->level(),
                         cell->index(),
                         // Pointer to the velocity's DoFHandler
                         &pseudo_pressure->get_dof_handler());

  scratch.pseudo_pressure_fe_values.reinit(pseudo_pressure_cell);

  scratch.pseudo_pressure_fe_values.get_function_gradients(
    pseudo_pressure->old_solution,
    scratch.old_pseudo_pressure_gradients);

  scratch.pseudo_pressure_fe_values.get_function_gradients(
    this->auxiliary_scalar->old_solution,
    scratch.old_auxiliary_scalar_gradients);

  scratch.pseudo_pressure_fe_values.get_function_gradients(
    this->auxiliary_scalar->old_old_solution,
    scratch.old_old_auxiliary_scalar_gradients);

  // Advection term
  if (parameters.convective_term_time_discretization ==
        RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit)
  {
    scratch.magnetic_field_fe_values[vector_extractor].get_function_gradients(
      magnetic_field->old_solution,
      scratch.old_magnetic_field_gradients);

    scratch.magnetic_field_fe_values[vector_extractor].get_function_gradients(
      magnetic_field->old_old_solution,
      scratch.old_old_magnetic_field_gradients);

    if (velocity != nullptr)
    {
      typename DoFHandler<dim>::active_cell_iterator
        velocity_cell(&velocity->get_triangulation(),
                      cell->level(),
                      cell->index(),
                      // Pointer to the velocity's DoFHandler
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

      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        advection_term[q] =
          beta[0] *
          (scratch.old_velocity_gradients[q] *
          scratch.old_magnetic_field_values[q]
          -
          scratch.old_magnetic_field_gradients[q] *
          scratch.old_velocity_values[q])
          +
          beta[1] *
          (scratch.old_old_velocity_gradients[q] *
          scratch.old_old_magnetic_field_values[q]
          -
          scratch.old_old_magnetic_field_gradients[q] *
          scratch.old_old_velocity_values[q]);
    }
    else if (ptr_velocity_function != nullptr)
    {
      ptr_velocity_function->value_list(
        scratch.magnetic_field_fe_values.get_quadrature_points(),
        scratch.velocity_values);

      ptr_velocity_function->gradient_list(
        scratch.magnetic_field_fe_values.get_quadrature_points(),
        scratch.velocity_gradients);

      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        advection_term[q] =
          beta[0] *
          (scratch.velocity_gradients[q] *
          scratch.old_magnetic_field_values[q]
          -
          scratch.old_magnetic_field_gradients[q] *
          scratch.velocity_values[q])
          +
          beta[1] *
          (scratch.velocity_gradients[q] *
          scratch.old_old_magnetic_field_values[q]
          -
          scratch.old_old_magnetic_field_gradients[q] *
          scratch.velocity_values[q]);
    }
  }

  // Supply term
  if (this->ptr_supply_term != nullptr)
  {
    this->ptr_supply_term->set_time(this->time_stepping.get_next_time());

    this->ptr_supply_term->value_list(scratch.magnetic_field_fe_values.get_quadrature_points(),
                                      scratch.supply_term_values);
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Evaluate the weak form of the right-hand side's terms at
    // the quadrature point
    time_derivative_term[q] =
              alpha[1] / this->time_stepping.get_next_step_size() *
              scratch.old_magnetic_field_values[q]
              +
              alpha[2] / this->time_stepping.get_next_step_size() *
              scratch.old_old_magnetic_field_values[q];

    pseudo_pressure_gradient_term[q] =
              (scratch.old_pseudo_pressure_gradients[q]
               -
               this->previous_step_sizes[0] / this->time_stepping.get_next_step_size() *
               alpha[1] / this->previous_alpha_zeros[0] *
               scratch.old_auxiliary_scalar_gradients[q]
               -
               this->previous_step_sizes[1] / this->time_stepping.get_next_step_size() *
               alpha[2] / this->previous_alpha_zeros[1] *
               scratch.old_old_auxiliary_scalar_gradients[q]);

    diffusion_term[q] =
              parameters.C *
              (gamma[1] *
               scratch.old_magnetic_field_curls[q]
               +
               gamma[2] *
               scratch.old_old_magnetic_field_curls[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.magnetic_field_fe_values[vector_extractor].value(i,q);
      scratch.curl_phi[i] = scratch.magnetic_field_fe_values[vector_extractor].curl(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
                    (scratch.phi[i] *
                     (pseudo_pressure_gradient_term[q]
                      +
                      scratch.supply_term_values[q]
                      -
                      time_derivative_term[q]
                      -
                      advection_term[q])
                     -
                     scratch.curl_phi[i] *
                     diffusion_term[q]) *
                    scratch.magnetic_field_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (velocity->get_constraints().is_inhomogeneously_constrained(
            data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bc(j,i) +=
                (alpha[0] / this->time_stepping.get_next_step_size() *
                 scratch.phi[j] *
                 scratch.phi[i]
                 +
                 gamma[0] * parameters.C *
                 scratch.curl_phi[j] *
                 scratch.curl_phi[i]) *
                scratch.magnetic_field_fe_values.JxW(q);

          if (parameters.convective_term_time_discretization ==
              RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
          {
            if (velocity != nullptr)
              data.local_matrix_for_inhomogeneous_bc(j,i) +=
                scratch.phi[j] *
                ((eta[0] * scratch.old_velocity_gradients[q]
                  +
                  eta[1] * scratch.old_old_velocity_gradients[q]) *
                 scratch.phi[i]
                 -
                 scratch.grad_phi[i] *
                 (eta[0] * scratch.old_velocity_values[q]
                  +
                  eta[1] * scratch.old_old_velocity_values[q])
                ) *
                scratch.magnetic_field_fe_values.JxW(q);
            else if (ptr_velocity_function != nullptr)
              data.local_matrix_for_inhomogeneous_bc(j,i) +=
                scratch.phi[j] *
                (scratch.velocity_gradients[q] *
                 scratch.phi[i]
                 -
                 scratch.grad_phi[i] *
                 scratch.velocity_values[q]
                 ) *
                scratch.magnetic_field_fe_values.JxW(q);
          }
        } // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points
}



template <int dim>
void MagneticInduction<dim>::copy_local_to_global_diffusion_step_rhs(
  const AssemblyData::MagneticInduction::DiffusionStepRHS::Copy &data)
{
  magnetic_field->get_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    this->diffusion_step_rhs,
    data.local_matrix_for_inhomogeneous_bc);
}


} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_diffusion_step();
template void RMHD::Solvers::MagneticInduction<3>::assemble_diffusion_step();

template void RMHD::Solvers::MagneticInduction<2>::assemble_diffusion_step_rhs();
template void RMHD::Solvers::MagneticInduction<3>::assemble_diffusion_step_rhs();

template void RMHD::Solvers::MagneticInduction<2>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<2>::active_cell_iterator                   &,
 RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<2>  &,
 RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Copy        &);
template void RMHD::Solvers::MagneticInduction<3>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<3>::active_cell_iterator                   &,
 RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<3>  &,
 RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Copy        &);

template void RMHD::Solvers::MagneticInduction<2>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Copy  &);
template void RMHD::Solvers::MagneticInduction<3>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Copy  &);
