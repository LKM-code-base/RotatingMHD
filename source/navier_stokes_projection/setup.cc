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
  if (phi->boundary_conditions.dirichlet_bcs.empty())
    flag_normalize_pressure = true;

  // If the matrices and vector are assembled, the sum of the mass and
  // stiffness matrices has to be updated.
  flag_add_mass_and_stiffness_matrices = true;
}

template <int dim>
void NavierStokesProjection<dim>::setup_phi()
{
  // Extract owned and relevant degrees of freedom and populate
  // AffineConstraint instance of the hanging nodes
  phi->setup_dofs();

  // Initiate the solution vectors
  phi->reinit();

  // Copy the pressure boundary conditions 
  phi->boundary_conditions.copy(pressure->boundary_conditions);

  // Inhomogeneous Dirichlet boundary conditions in the pressure space 
  // translate into homogeneous Dirichlet boundary conditions in the 
  // phi space
  for (auto &dirichlet_bc : phi->boundary_conditions.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>();

  // Neumann boundary conditions in the velocity space translate into
  // homogeneous Dirichlet boundary conditions in the phi space
  for (auto &neumann_bc : velocity->boundary_conditions.neumann_bcs)
    phi->boundary_conditions.set_dirichlet_bcs(neumann_bc.first);

  // Apply boundary conditions
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
      sparsity_pattern(velocity->locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(*(velocity->dof_handler),
                                      sparsity_pattern,
                                      velocity->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       velocity->locally_owned_dofs,
       mpi_communicator,
       velocity->locally_relevant_dofs);

      velocity_mass_plus_laplace_matrix.reinit
      (velocity->locally_owned_dofs,
       velocity->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_system_matrix.reinit
      (velocity->locally_owned_dofs,
       velocity->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_mass_matrix.reinit
      (velocity->locally_owned_dofs,
       velocity->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_laplace_matrix.reinit
      (velocity->locally_owned_dofs,
       velocity->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_advection_matrix.reinit
      (velocity->locally_owned_dofs,
       velocity->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(velocity->locally_owned_dofs,
                       velocity->locally_owned_dofs,
                       velocity->locally_relevant_dofs,
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(*(velocity->dof_handler),
                                      sparsity_pattern,
                                      velocity->constraints,
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
      pressure_sparsity_pattern(pressure->locally_relevant_dofs);

      DynamicSparsityPattern
      phi_sparsity_pattern(phi->locally_relevant_dofs);

      DynamicSparsityPattern
      projection_sparsity_pattern(pressure->locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      pressure_sparsity_pattern,
                                      pressure->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(*(phi->dof_handler),
                                      phi_sparsity_pattern,
                                      phi->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      projection_sparsity_pattern,
                                      pressure->hanging_nodes,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));


      SparsityTools::distribute_sparsity_pattern
      (pressure_sparsity_pattern,
       pressure->locally_owned_dofs,
       mpi_communicator,
       pressure->locally_relevant_dofs);

      SparsityTools::distribute_sparsity_pattern
      (phi_sparsity_pattern,
       phi->locally_owned_dofs,
       mpi_communicator,
       phi->locally_relevant_dofs);

      SparsityTools::distribute_sparsity_pattern
      (projection_sparsity_pattern,
       pressure->locally_owned_dofs,
       mpi_communicator,
       pressure->locally_relevant_dofs);

      pressure_laplace_matrix.reinit
      (pressure->locally_owned_dofs,
       pressure->locally_owned_dofs,
       pressure_sparsity_pattern,
       mpi_communicator);
      phi_laplace_matrix.reinit
      (phi->locally_owned_dofs,
       phi->locally_owned_dofs,
       phi_sparsity_pattern,
       mpi_communicator);
      projection_mass_matrix.reinit
      (pressure->locally_owned_dofs,
       pressure->locally_owned_dofs,
       projection_sparsity_pattern,
       mpi_communicator);


    #else
      TrilinosWrappers::SparsityPattern
      pressure_sparsity_pattern(pressure->locally_owned_dofs,
                                pressure->locally_owned_dofs,
                                pressure->locally_relevant_dofs,
                                mpi_communicator);

      TrilinosWrappers::SparsityPattern
      phi_sparsity_pattern(phi->locally_owned_dofs,
                           phi->locally_owned_dofs,
                           phi->locally_relevant_dofs,
                           mpi_communicator);

      TrilinosWrappers::SparsityPattern
      projection_sparsity_pattern(pressure->locally_owned_dofs,
                                  pressure->locally_owned_dofs,
                                  pressure->locally_relevant_dofs,
                                  mpi_communicator);

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      pressure_sparsity_pattern,
                                      pressure->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(*(phi->dof_handler),
                                      phi_sparsity_pattern,
                                      phi->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      projection_sparsity_pattern,
                                      pressure->hanging_nodes,
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

  // Initialize the vectors related to the velocity.
  diffusion_step_rhs.reinit(velocity->distributed_vector);
  extrapolated_velocity.reinit(velocity->solution);
  velocity_tmp.reinit(velocity->solution);

  // Initialze the vectors related to the pressure.
  poisson_prestep_rhs.reinit(pressure->distributed_vector);
  pressure_tmp.reinit(pressure->solution);
  projection_step_rhs.reinit(phi->distributed_vector);
  correction_step_rhs.reinit(pressure->distributed_vector);

  // Initialize the vector related to the temperature.
  if (!flag_ignore_bouyancy_term)
    extrapolated_temperature.reinit(temperature->solution);

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
  RMHD::EquationData::BodyForce<dim> &body_force)
{
  body_force_ptr  = &body_force;
}

template <int dim>
void NavierStokesProjection<dim>::reset_phi()
{
  phi->set_solution_vectors_to_zero();
  flag_setup_phi = true;
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

template void RMHD::NavierStokesProjection<2>::set_body_force(RMHD::EquationData::BodyForce<2> &);
template void RMHD::NavierStokesProjection<3>::set_body_force(RMHD::EquationData::BodyForce<3> &);

template void RMHD::NavierStokesProjection<2>::reset_phi();
template void RMHD::NavierStokesProjection<3>::reset_phi();
