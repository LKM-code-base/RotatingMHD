#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/dofs/dof_tools.h>
#ifdef USE_PETSC_LA
  #include <deal.II/lac/dynamic_sparsity_pattern.h>
  #include <deal.II/lac/sparsity_tools.h>
#else
  #include <deal.II/lac/trilinos_sparsity_pattern.h>
#endif
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::setup()
{
  // The set up of @ref phi happens internally only once, unless
  // specified by the user. See the documentation of the flag for
  // more information.
  if (flag_setup_phi)
    setup_phi();

  setup_matrices();

  setup_vectors();

  assemble_constant_matrices();

  // If the pressure correction variable @ref phi only has Neumann
  // boundary conditions, its solution is defined only up to a constant.
  if (phi->get_dirichlet_boundary_conditions().empty())
    flag_normalize_pressure = true;

  // If the matrices and vector are assembled, the sum of the mass and
  // stiffness matrices has to be updated.
  flag_matrices_were_updated = true;

  if (time_stepping.get_step_number() == 0)
    poisson_prestep();
}



template <int dim>
void NavierStokesProjection<dim>::setup_phi()
{
  // Extract owned and relevant degrees of freedom and populate
  // AffineConstraint instance of the hanging nodes
  phi->setup_dofs();
  phi->setup_vectors();

  // Clears the boundary conditions
  phi->clear_boundary_conditions();
  phi->setup_boundary_conditions();

  // Copy the pressure boundary conditions
  Entities::BoundaryConditionsBase<dim> &phi_boundary_conditions =
      phi->get_boundary_conditions();
  const Entities::BoundaryConditionsBase<dim> &pressure_boundary_conditions =
      pressure->get_boundary_conditions();

  phi_boundary_conditions.copy(pressure_boundary_conditions);

  // Inhomogeneous Dirichlet boundary conditions in the pressure space
  // translate to homogeneous Dirichlet boundary conditions in the
  // phi space
  for (auto &dirichlet_bc : phi_boundary_conditions.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>();

  // The velocity field has to be constrained over all the boundary.
  // If it is not, the following sets homogeneous Neumann boundary
  // conditions on the unconstrained boundaries.
  std::vector<types::boundary_id> unconstrained_boundary_ids =
    velocity->get_boundary_conditions().get_unconstrained_boundary_ids();

  // An exception is made if there is a Dirichlet boundary condition
  // on the pressure.
  for (const auto &unconstrained_boundary_id: unconstrained_boundary_ids)
    if (pressure_boundary_conditions.dirichlet_bcs.find(unconstrained_boundary_id)
          != pressure_boundary_conditions.dirichlet_bcs.end())
      unconstrained_boundary_ids.erase(
        std::remove(unconstrained_boundary_ids.begin(),
                    unconstrained_boundary_ids.end(),
                    unconstrained_boundary_id),
        unconstrained_boundary_ids.end());

  if (unconstrained_boundary_ids.size() != 0)
  {
    *pcout << std::endl
           << "Warning: No boundary conditions for the velocity field were "
           << "assigned on the boundaries {";
    for (const auto &unconstrained_boundary_id: unconstrained_boundary_ids)
    {
      *pcout << unconstrained_boundary_id << ", ";
      velocity->set_neumann_boundary_condition(unconstrained_boundary_id);
    }
    *pcout << "\b\b}. Homogeneous Neumann boundary conditions will be"
              " assumed in order to properly constraint the problem.\n"
           << std::endl;
  }

  // Neumann boundary conditions in the velocity space translate into
  // homogeneous Dirichlet boundary conditions in the phi space
  for (const auto &neumann_bc : velocity->get_neumann_boundary_conditions())
    phi->set_dirichlet_boundary_condition(neumann_bc.first);

  // The remaining unconstrained boundaries in the phi space are set to
  // homogeneous Neumann boundary conditions
  for (const auto &unconstrained_boundary_id: phi_boundary_conditions.get_unconstrained_boundary_ids())
    phi->set_neumann_boundary_condition(unconstrained_boundary_id);

  // Close and apply boundary conditions
  phi->close_boundary_conditions();
  phi->apply_boundary_conditions();

  //Set all the solution vectors to zero
  phi->set_solution_vectors_to_zero();

  // The set up happens internally only once, unless specified by the
  // user. See the documentation of the flag for more information.
  flag_setup_phi = false;
}



template <int dim>
void NavierStokesProjection<dim>::setup_matrices()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Setting up matrices...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Setup - Matrices");

  // Clear all matrices related to the diffusion step
  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_system_matrix.clear();

  // Set ups the sparsity patterns and initiates all the matrices
  // related to the diffusion step.
  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      sparsity_pattern(velocity->get_locally_relevant_dofs());

      DoFTools::make_sparsity_pattern(velocity->get_dof_handler(),
                                      sparsity_pattern,
                                      velocity->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       velocity->get_locally_owned_dofs(),
       mpi_communicator,
       velocity->get_locally_relevant_dofs());

      velocity_mass_plus_laplace_matrix.reinit
      (velocity->get_locally_owned_dofs(),
       velocity->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      velocity_system_matrix.reinit
      (velocity->get_locally_owned_dofs(),
       velocity->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      velocity_mass_matrix.reinit
      (velocity->get_locally_owned_dofs(),
       velocity->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      velocity_laplace_matrix.reinit
      (velocity->get_locally_owned_dofs(),
       velocity->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      velocity_advection_matrix.reinit
      (velocity->get_locally_owned_dofs(),
       velocity->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(velocity->get_locally_owned_dofs(),
                       velocity->get_locally_owned_dofs(),
                       velocity->get_locally_relevant_dofs(),
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(velocity->get_dof_handler(),
                                      sparsity_pattern,
                                      velocity->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      sparsity_pattern.compress();

      velocity_mass_plus_laplace_matrix.reinit(sparsity_pattern);
      velocity_system_matrix.reinit(sparsity_pattern);
      velocity_mass_matrix.reinit(sparsity_pattern);
      velocity_laplace_matrix.reinit(sparsity_pattern);
      velocity_advection_matrix.reinit(sparsity_pattern);
   #endif
  }

  // Clear all matrices related to the pressure
  pressure_laplace_matrix.clear();  // Used in the poisson pre-step
  phi_laplace_matrix.clear();       // Used in the projection step
  projection_mass_matrix.clear();   // Used in the correction step

  // Set ups the sparsity patterns and initiates all the matrices
  // related to the pressure.
  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      pressure_sparsity_pattern(pressure->get_locally_relevant_dofs());

      DynamicSparsityPattern
      phi_sparsity_pattern(phi->get_locally_relevant_dofs());

      DynamicSparsityPattern
      projection_sparsity_pattern(pressure->get_locally_relevant_dofs());

      DoFTools::make_sparsity_pattern(pressure->get_dof_handler(),
                                      pressure_sparsity_pattern,
                                      pressure->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(phi->get_dof_handler(),
                                      phi_sparsity_pattern,
                                      phi->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(pressure->get_dof_handler(),
                                      projection_sparsity_pattern,
                                      pressure->get_hanging_node_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));


      SparsityTools::distribute_sparsity_pattern
      (pressure_sparsity_pattern,
       pressure->get_locally_owned_dofs(),
       mpi_communicator,
       pressure->get_locally_relevant_dofs());

      SparsityTools::distribute_sparsity_pattern
      (phi_sparsity_pattern,
       phi->get_locally_owned_dofs(),
       mpi_communicator,
       phi->locally_relevant_dofs);

      SparsityTools::distribute_sparsity_pattern
      (projection_sparsity_pattern,
       pressure->get_locally_owned_dofs(),
       mpi_communicator,
       pressure->get_locally_relevant_dofs());

      pressure_laplace_matrix.reinit
      (pressure->get_locally_owned_dofs(),
       pressure->get_locally_owned_dofs(),
       pressure_sparsity_pattern,
       mpi_communicator);
      phi_laplace_matrix.reinit
      (phi->get_locally_owned_dofs(),
       phi->get_locally_owned_dofs(),
       phi_sparsity_pattern,
       mpi_communicator);
      projection_mass_matrix.reinit
      (pressure->get_locally_owned_dofs(),
       pressure->get_locally_owned_dofs(),
       projection_sparsity_pattern,
       mpi_communicator);


    #else
      TrilinosWrappers::SparsityPattern
      pressure_sparsity_pattern(pressure->get_locally_owned_dofs(),
                                pressure->get_locally_owned_dofs(),
                                pressure->get_locally_relevant_dofs(),
                                mpi_communicator);

      TrilinosWrappers::SparsityPattern
      phi_sparsity_pattern(phi->get_locally_owned_dofs(),
                           phi->get_locally_owned_dofs(),
                           phi->get_locally_relevant_dofs(),
                           mpi_communicator);

      TrilinosWrappers::SparsityPattern
      projection_sparsity_pattern(pressure->get_locally_owned_dofs(),
                                  pressure->get_locally_owned_dofs(),
                                  pressure->get_locally_relevant_dofs(),
                                  mpi_communicator);

      DoFTools::make_sparsity_pattern(pressure->get_dof_handler(),
                                      pressure_sparsity_pattern,
                                      pressure->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(phi->get_dof_handler(),
                                      phi_sparsity_pattern,
                                      phi->get_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(pressure->get_dof_handler(),
                                      projection_sparsity_pattern,
                                      pressure->get_hanging_node_constraints(),
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      pressure_sparsity_pattern.compress();
      phi_sparsity_pattern.compress();
      projection_sparsity_pattern.compress();

      pressure_laplace_matrix.reinit(pressure_sparsity_pattern);
      phi_laplace_matrix.reinit(phi_sparsity_pattern);
      projection_mass_matrix.reinit(projection_sparsity_pattern);
    #endif
  }

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void NavierStokesProjection<dim>::
setup_vectors()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Setting up vectors...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Setup - Vectors");

  poisson_prestep_rhs.reinit(pressure->distributed_vector);

  diffusion_step_rhs.reinit(velocity->distributed_vector);

  projection_step_rhs.reinit(phi->distributed_vector);
  correction_step_rhs.reinit(pressure->distributed_vector);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void NavierStokesProjection<dim>::
assemble_constant_matrices()
{
  assemble_velocity_matrices();

  assemble_pressure_matrices();
}



template <int dim>
void NavierStokesProjection<dim>::set_body_force(
  RMHD::EquationData::VectorFunction<dim> &body_force)
{
  body_force_ptr = &body_force;
}



template <int dim>
void NavierStokesProjection<dim>::set_gravity_vector(
  RMHD::EquationData::VectorFunction<dim> &gravity_vector)
{
  gravity_vector_ptr = &gravity_vector;
}



template <int dim>
void NavierStokesProjection<dim>::set_angular_velocity_vector(
  RMHD::EquationData::AngularVelocity<dim> &angular_velocity_vector)
{
  angular_velocity_vector_ptr = &angular_velocity_vector;
}



template <int dim>
void NavierStokesProjection<dim>::clear()
{
  // Body force density pointers
  body_force_ptr              = nullptr;
  gravity_vector_ptr          = nullptr;
  angular_velocity_vector_ptr = nullptr;

  // Preconditioners
  correction_step_preconditioner.reset();
  diffusion_step_preconditioner.reset();
  projection_step_preconditioner.reset();
  poisson_prestep_preconditioner.reset();

  // Velocity matrices
  velocity_system_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_mass_matrix.clear();

  // Velocity vectors
  diffusion_step_rhs.clear();

  // Pressure matrices
  pressure_laplace_matrix.clear();
  projection_mass_matrix.clear();
  phi_laplace_matrix.clear();

  // Pressure vectors
  correction_step_rhs.clear();
  poisson_prestep_rhs.clear();
  projection_step_rhs.clear();

  // Internal entity
  phi->clear();

  // Norms
  norm_diffusion_rhs  = std::numeric_limits<double>::min();
  norm_projection_rhs = std::numeric_limits<double>::min();

  // Internal flags
  flag_setup_phi              = true;
  flag_matrices_were_updated  = true;
  flag_normalize_pressure     = false;
}



template <int dim>
void NavierStokesProjection<dim>::reset_phi()
{
  phi->set_solution_vectors_to_zero();
  flag_setup_phi = true;
}



template <int dim>
void NavierStokesProjection<dim>::reset()
{
  velocity_system_matrix.clear();
  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  diffusion_step_rhs.clear();
  projection_mass_matrix.clear();
  pressure_laplace_matrix.clear();
  phi_laplace_matrix.clear();
  projection_step_rhs.clear();
  poisson_prestep_rhs.clear();
  correction_step_rhs.clear();
  norm_diffusion_rhs  = 0.;
  norm_projection_rhs = 0.;
  flag_setup_phi              = true;
  flag_matrices_were_updated  = true;
}



template <int dim>
void NavierStokesProjection<dim>::
poisson_prestep()
{
  // Sets the body force's internal time to the simulation's start time
  if (body_force_ptr != nullptr)
    body_force_ptr->set_time(time_stepping.get_start_time());

  // Assemble linear system
  assemble_poisson_prestep();

  // Solve linear system
  solve_poisson_prestep();
}


}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::setup();
template void RMHD::NavierStokesProjection<3>::setup();

template void RMHD::NavierStokesProjection<2>::setup_matrices();
template void RMHD::NavierStokesProjection<3>::setup_matrices();

template void RMHD::NavierStokesProjection<2>::setup_vectors();
template void RMHD::NavierStokesProjection<3>::setup_vectors();

template void RMHD::NavierStokesProjection<2>::assemble_constant_matrices();
template void RMHD::NavierStokesProjection<3>::assemble_constant_matrices();

template void RMHD::NavierStokesProjection<2>::set_body_force(RMHD::EquationData::VectorFunction<2> &);
template void RMHD::NavierStokesProjection<3>::set_body_force(RMHD::EquationData::VectorFunction<3> &);

template void RMHD::NavierStokesProjection<2>::set_gravity_vector(RMHD::EquationData::VectorFunction<2> &);
template void RMHD::NavierStokesProjection<3>::set_gravity_vector(RMHD::EquationData::VectorFunction<3> &);

template void RMHD::NavierStokesProjection<2>::set_angular_velocity_vector(RMHD::EquationData::AngularVelocity<2> &);
template void RMHD::NavierStokesProjection<3>::set_angular_velocity_vector(RMHD::EquationData::AngularVelocity<3> &);

template void RMHD::NavierStokesProjection<2>::clear();
template void RMHD::NavierStokesProjection<3>::clear();

template void RMHD::NavierStokesProjection<2>::reset_phi();
template void RMHD::NavierStokesProjection<3>::reset_phi();

template void RMHD::NavierStokesProjection<2>::reset();
template void RMHD::NavierStokesProjection<3>::reset();

template void RMHD::NavierStokesProjection<2>::poisson_prestep();
template void RMHD::NavierStokesProjection<3>::poisson_prestep();
