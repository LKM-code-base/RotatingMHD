#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
setup()
{
  setup_matrices();
  setup_vectors();
  assemble_constant_matrices();
  reinit_internal_variables();
}

template <int dim>
void NavierStokesProjection<dim>::
setup_matrices()
{
  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_system_matrix.clear();

  {
  TrilinosWrappers::SparsityPattern sp(velocity.locally_owned_dofs,
                                       velocity.locally_owned_dofs,
                                       velocity.locally_relevant_dofs,
                                       MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(velocity.dof_handler, 
                                  sp,
                                  velocity.constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD));
  sp.compress();

  velocity_mass_plus_laplace_matrix.reinit(sp);
  velocity_system_matrix.reinit(sp);
  velocity_mass_matrix.reinit(sp);
  velocity_laplace_matrix.reinit(sp);
  velocity_advection_matrix.reinit(sp);
  }

  pressure_mass_matrix.clear();
  pressure_laplace_matrix.clear();
  {
  TrilinosWrappers::SparsityPattern sp(pressure.locally_owned_dofs,
                                       pressure.locally_owned_dofs,
                                       pressure.locally_relevant_dofs,
                                       MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(pressure.dof_handler, 
                                  sp,
                                  pressure.constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD));
  sp.compress();
  pressure_laplace_matrix.reinit(sp);
  pressure_mass_matrix.reinit(sp);
  }
}

template <int dim>
void NavierStokesProjection<dim>::
setup_vectors()
{
  pressure_rhs.reinit(pressure.locally_owned_dofs,
                      pressure.locally_relevant_dofs,
                      MPI_COMM_WORLD,
                      true);
  poisson_prestep_rhs.reinit(pressure_rhs);
  pressure_tmp.reinit(pressure.solution);
  
  phi.reinit(pressure.solution);
  old_phi.reinit(pressure.solution);
  old_old_phi.reinit(pressure.solution);

  velocity_rhs.reinit(velocity.locally_owned_dofs,
                      velocity.locally_relevant_dofs,
                      MPI_COMM_WORLD,
                      true);
  extrapolated_velocity.reinit(velocity.solution);
  velocity_tmp.reinit(velocity.solution);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_constant_matrices()
{
  assemble_velocity_matrices();
  assemble_pressure_matrices();
}

template <int dim>
void NavierStokesProjection<dim>::
reinit_internal_variables()
{
  phi         = 0.;
  old_phi     = 0.;
  old_old_phi = 0.;
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
template void RMHD::NavierStokesProjection<2>::reinit_internal_variables();
template void RMHD::NavierStokesProjection<3>::reinit_internal_variables();