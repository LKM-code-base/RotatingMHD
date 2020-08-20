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
  pressure_tmp.reinit(pressure.solution_n);
  
  phi_n.reinit(pressure.solution_n);
  phi_n_minus_1.reinit(pressure.solution_n);

  velocity_rhs.reinit(velocity.locally_owned_dofs,
                      velocity.locally_relevant_dofs,
                      MPI_COMM_WORLD,
                      true);
  extrapolated_velocity.reinit(velocity.solution_n);
  velocity_tmp.reinit(velocity.solution_n);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_constant_matrices()
{
  assemble_velocity_matrices();
  assemble_pressure_matrices();

  if (!flag_adpative_time_step)
  {
    velocity_mass_plus_laplace_matrix = 0.;
    velocity_mass_plus_laplace_matrix.add(1.0 / Re, 
                                          velocity_laplace_matrix);
    velocity_mass_plus_laplace_matrix.add(VSIMEX.alpha[2], 
                                          velocity_mass_matrix);
  }
}
/*
template <int dim>
void NavierStokesProjection<dim>::
initialize()
{
  phi_n         = 0.;
  phi_n_minus_1 = 0.;
 {
    TrilinosWrappers::MPI::Vector tmp_velocity_n_minus_1(
                                          locally_owned_velocity_dofs);
    TrilinosWrappers::MPI::Vector tmp_velocity_n(
                                          locally_owned_velocity_dofs);
    
    velocity_initial_conditions.set_time(t_0);    
    VectorTools::project(velocity.dof_handler,
                         velocity.constraints,
                         QGauss<dim>(velocity_fe_degree + 2),
                         velocity_initial_conditions,
                         tmp_velocity_n_minus_1);

    velocity_initial_conditions.advance_time(dt_n);
    VectorTools::project(velocity.dof_handler,
                         velocity.constraints,
                         QGauss<dim>(velocity_fe_degree + 2),
                         velocity_initial_conditions,
                         tmp_velocity_n);

    velocity_n_minus_1  = tmp_velocity_n_minus_1;
    velocity_n          = tmp_velocity_n;
 }

 {
    TrilinosWrappers::MPI::Vector tmp_pressure_n_minus_1(
                                          pressure.locally_owned_dofs);
    TrilinosWrappers::MPI::Vector tmp_pressure_n(
                                          pressure.locally_owned_dofs);
    
    pressure_initial_conditions.set_time(t_0);    
    VectorTools::project(pressure.dof_handler,
                          pressure.constraints,
                          QGauss<dim>(pressure_fe_degree + 2),
                          pressure_initial_conditions,
                          tmp_pressure_n_minus_1);

    pressure_initial_conditions.advance_time(dt_n);
    VectorTools::project(pressure.dof_handler,
                          pressure.constraints,
                          QGauss<dim>(pressure_fe_degree + 2),
                          pressure_initial_conditions,
                          tmp_pressure_n);

    pressure_n_minus_1  = tmp_pressure_n_minus_1;
    pressure_n          = tmp_pressure_n;
 }
}
*/
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
/*template void RMHD::NavierStokesProjection<2>::initialize();
template void RMHD::NavierStokesProjection<3>::initialize();*/