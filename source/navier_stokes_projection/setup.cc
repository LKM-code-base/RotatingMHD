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
  if (flag_setup_phi)
    setup_phi();

  setup_matrices();

  setup_vectors();

  assemble_constant_matrices();

  if (pressure->boundary_conditions.dirichlet_bcs.empty())
    flag_normalize_pressure = true;

  flag_add_mass_and_stiffness_matrices = true;
}

template <int dim>
void NavierStokesProjection<dim>::setup_phi()
{
  /*! Extract owned and relevant degrees of freedom and populate
   *  AffineConstraint instance of the hanging nodes
   */
  phi->setup_dofs();

  /*!
   * Initiate the solution vectors
   */
  phi->reinit();

  /* Copy the pressure boundary conditions */
  phi->boundary_conditions.copy(pressure->boundary_conditions);

  /*! 
   * Inhomogeneous Dirichlet boundary conditions in the pressure space 
   * translate into homogeneous Dirichlet boundary conditions in the 
   * phi space
   */
  for (auto &dirichlet_bc : phi->boundary_conditions.dirichlet_bcs)
    dirichlet_bc.second = std::make_shared<Functions::ZeroFunction<dim>>();

  /*!
   * Neumann boundary conditions in the velocity space translate into
   * homogeneous Dirichlet boundary conditions in the phi space
   */
  for (auto &neumann_bc : velocity->boundary_conditions.neumann_bcs)
    phi->boundary_conditions.set_dirichlet_bcs(neumann_bc.first);

  /* Apply boundary conditions */
  phi->apply_boundary_conditions();

  /* Set all the solution vectors to zero */
  phi->set_solution_vectors_to_zero();

  flag_setup_phi = false;
  /*!
   * @todo Replace pressure.constraint calls with phi.constraints
   */ 
}

template <int dim>
void NavierStokesProjection<dim>::setup_matrices()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Setting up matrices...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Setup - Matrices");

  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_system_matrix.clear();

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

  pressure_mass_matrix.clear();
  pressure_laplace_matrix.clear();
  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      sparsity_pattern(pressure->locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      sparsity_pattern,
                                      pressure->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       pressure->locally_owned_dofs,
       mpi_communicator,
       pressure->locally_relevant_dofs);

      pressure_laplace_matrix.reinit
      (pressure->locally_owned_dofs,
       pressure->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      pressure_mass_matrix.reinit
      (pressure->locally_owned_dofs,
       pressure->locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(pressure->locally_owned_dofs,
                       pressure->locally_owned_dofs,
                       pressure->locally_relevant_dofs,
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(*(pressure->dof_handler),
                                      sparsity_pattern,
                                      pressure->constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));
      sparsity_pattern.compress();

      pressure_laplace_matrix.reinit(sparsity_pattern);
      pressure_mass_matrix.reinit(sparsity_pattern);
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

  pressure_rhs.reinit(pressure->distributed_vector);
  poisson_prestep_rhs.reinit(pressure->distributed_vector);
  velocity_rhs.reinit(velocity->distributed_vector);


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
