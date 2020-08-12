#include <rotatingMHD/projection_solver.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
setup_matrices_and_vectors()
{
  setup_velocity_matrices();
  setup_pressure_matrices();

  pressure_rhs.reinit(locally_owned_pressure_dofs,
                      locally_relevant_pressure_dofs,
                      mpi_communicator,
                      true);
  pressure_n.reinit(locally_relevant_pressure_dofs,
                    mpi_communicator);
  pressure_n_minus_1.reinit(pressure_n);
  pressure_tmp.reinit(pressure_n);
  
  phi_n.reinit(pressure_n);
  phi_n_minus_1.reinit(pressure_n);

  velocity_rhs.reinit(locally_owned_velocity_dofs,
                      locally_relevant_velocity_dofs,
                      mpi_communicator,
                      true);
  velocity_n.reinit(locally_relevant_velocity_dofs,
                    mpi_communicator);
  velocity_n_minus_1.reinit(velocity_n);
  extrapolated_velocity.reinit(velocity_n);
  velocity_tmp.reinit(velocity_n);
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

template <int dim>
void NavierStokesProjection<dim>::
setup_velocity_matrices()
{
  velocity_mass_matrix.clear();
  velocity_laplace_matrix.clear();
  velocity_mass_plus_laplace_matrix.clear();
  velocity_advection_matrix.clear();
  velocity_system_matrix.clear();

  TrilinosWrappers::SparsityPattern sp(locally_owned_velocity_dofs,
                                       locally_owned_velocity_dofs,
                                       locally_relevant_velocity_dofs,
                                       mpi_communicator);
  DoFTools::make_sparsity_pattern(velocity_dof_handler, 
                                  sp,
                                  velocity_constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    mpi_communicator));
  sp.compress();

  velocity_mass_plus_laplace_matrix.reinit(sp);
  velocity_system_matrix.reinit(sp);
  velocity_mass_matrix.reinit(sp);
  velocity_laplace_matrix.reinit(sp);
  velocity_advection_matrix.reinit(sp);
}

template <int dim>
void NavierStokesProjection<dim>::
setup_pressure_matrices()
{
  pressure_mass_matrix.clear();
  pressure_laplace_matrix.clear();
  TrilinosWrappers::SparsityPattern sp(locally_owned_pressure_dofs,
                                       locally_owned_pressure_dofs,
                                       locally_relevant_pressure_dofs,
                                       mpi_communicator);
  DoFTools::make_sparsity_pattern(pressure_dof_handler, 
                                  sp,
                                  pressure_constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    mpi_communicator));
  sp.compress();
  pressure_laplace_matrix.reinit(sp);
  pressure_mass_matrix.reinit(sp);
}

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
    VectorTools::project(velocity_dof_handler,
                         velocity_constraints,
                         QGauss<dim>(velocity_fe_degree + 2),
                         velocity_initial_conditions,
                         tmp_velocity_n_minus_1);

    velocity_initial_conditions.advance_time(dt_n);
    VectorTools::project(velocity_dof_handler,
                         velocity_constraints,
                         QGauss<dim>(velocity_fe_degree + 2),
                         velocity_initial_conditions,
                         tmp_velocity_n);

    velocity_n_minus_1  = tmp_velocity_n_minus_1;
    velocity_n          = tmp_velocity_n;
 }

 {
    TrilinosWrappers::MPI::Vector tmp_pressure_n_minus_1(
                                          locally_owned_pressure_dofs);
    TrilinosWrappers::MPI::Vector tmp_pressure_n(
                                          locally_owned_pressure_dofs);
    
    pressure_initial_conditions.set_time(t_0);    
    VectorTools::project(pressure_dof_handler,
                          pressure_constraints,
                          QGauss<dim>(pressure_fe_degree + 2),
                          pressure_initial_conditions,
                          tmp_pressure_n_minus_1);

    pressure_initial_conditions.advance_time(dt_n);
    VectorTools::project(pressure_dof_handler,
                          pressure_constraints,
                          QGauss<dim>(pressure_fe_degree + 2),
                          pressure_initial_conditions,
                          tmp_pressure_n);

    pressure_n_minus_1  = tmp_pressure_n_minus_1;
    pressure_n          = tmp_pressure_n;
 }
}

}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::setup_matrices_and_vectors();
template void Step35::NavierStokesProjection<3>::setup_matrices_and_vectors();
template void Step35::NavierStokesProjection<2>::assemble_constant_matrices();
template void Step35::NavierStokesProjection<3>::assemble_constant_matrices();
template void Step35::NavierStokesProjection<2>::setup_velocity_matrices();
template void Step35::NavierStokesProjection<3>::setup_velocity_matrices();
template void Step35::NavierStokesProjection<2>::setup_pressure_matrices();
template void Step35::NavierStokesProjection<3>::setup_pressure_matrices();
template void Step35::NavierStokesProjection<2>::initialize();
template void Step35::NavierStokesProjection<3>::initialize();