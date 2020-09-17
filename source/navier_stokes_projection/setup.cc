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
  setup_matrices();

  setup_vectors();

  assemble_constant_matrices();

  reinit_internal_entities();
}

template <int dim>
void NavierStokesProjection<dim>::setup_matrices()
{
  if (parameters.verbose)
    *pcout << "  Setup matrices..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Matrix setup");

  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_system_matrix.clear();

  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      sparsity_pattern(velocity.locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(velocity.dof_handler,
                                      sparsity_pattern,
                                      velocity.constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       velocity.locally_owned_dofs,
       mpi_communicator,
       velocity.locally_relevant_dofs);

      velocity_mass_plus_laplace_matrix.reinit
      (velocity.locally_owned_dofs,
       velocity.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_system_matrix.reinit
      (velocity.locally_owned_dofs,
       velocity.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_mass_matrix.reinit
      (velocity.locally_owned_dofs,
       velocity.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_laplace_matrix.reinit
      (velocity.locally_owned_dofs,
       velocity.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      velocity_advection_matrix.reinit
      (velocity.locally_owned_dofs,
       velocity.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(velocity.locally_owned_dofs,
                       velocity.locally_owned_dofs,
                       velocity.locally_relevant_dofs,
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(velocity.dof_handler,
                                      sparsity_pattern,
                                      velocity.constraints,
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
      sparsity_pattern(pressure.locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(pressure.dof_handler,
                                      sparsity_pattern,
                                      pressure.constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       pressure.locally_owned_dofs,
       mpi_communicator,
       pressure.locally_relevant_dofs);

      pressure_laplace_matrix.reinit
      (pressure.locally_owned_dofs,
       pressure.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);
      pressure_mass_matrix.reinit
      (pressure.locally_owned_dofs,
       pressure.locally_owned_dofs,
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(pressure.locally_owned_dofs,
                       pressure.locally_owned_dofs,
                       pressure.locally_relevant_dofs,
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(pressure.dof_handler,
                                      sparsity_pattern,
                                      pressure.constraints,
                                      false,
                                      Utilities::MPI::this_mpi_process(mpi_communicator));
      sparsity_pattern.compress();

      pressure_laplace_matrix.reinit(sparsity_pattern);
      pressure_mass_matrix.reinit(sparsity_pattern);
    #endif
  }

  if (parameters.verbose)
    *pcout << "    done." << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::
setup_vectors()
{
  if (parameters.verbose)
    *pcout << "  Setup vectors..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Setup vectors");

  #ifdef USE_PETSC_LA
    pressure_rhs.reinit(pressure.locally_owned_dofs,
                        mpi_communicator);
  #else
    pressure_rhs.reinit(pressure.locally_owned_dofs,
                        pressure.locally_relevant_dofs,
                        mpi_communicator,
                        true);
  #endif
  poisson_prestep_rhs.reinit(pressure_rhs);
  pressure_tmp.reinit(pressure.solution);
  
  phi.reinit(pressure.solution);
  old_phi.reinit(pressure.solution);
  old_old_phi.reinit(pressure.solution);

  #ifdef USE_PETSC_LA
    velocity_rhs.reinit(velocity.locally_owned_dofs,
                        mpi_communicator);
  #else
    velocity_rhs.reinit(velocity.locally_owned_dofs,
                        velocity.locally_relevant_dofs,
                        mpi_communicator,
                        true);
  #endif

  extrapolated_velocity.reinit(velocity.solution);
  velocity_tmp.reinit(velocity.solution);

  if (parameters.verbose)
    *pcout << "     done." << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_constant_matrices()
{
  assemble_velocity_matrices();

  assemble_pressure_matrices();
}

template <int dim>
void NavierStokesProjection<dim>::reinit_internal_entities()
{
  phi         = 0.;
  old_phi     = 0.;
  old_old_phi = 0.;
  flag_diffusion_matrix_assembled = false;
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

template void RMHD::NavierStokesProjection<2>::reinit_internal_entities();
template void RMHD::NavierStokesProjection<3>::reinit_internal_entities();
