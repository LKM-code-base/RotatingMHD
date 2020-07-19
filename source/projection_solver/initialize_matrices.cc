/*
 * initialize_matrices.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>

#include <fstream>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::initialize_velocity_matrices()
{
  {
    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(),
                               dof_handler_velocity.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp);
    sparsity_pattern_velocity.copy_from(dsp);
    std::ofstream out ("results_mod/v_sparsity_pattern.gpl");
    sparsity_pattern_velocity.print_gnuplot(out);
  }
  vel_Laplace_plus_Mass.reinit(sparsity_pattern_velocity);

  //for (unsigned int d = 0; d < dim; ++d)
  vel_it_matrix.reinit(sparsity_pattern_velocity);
  vel_Mass.reinit(sparsity_pattern_velocity);
  vel_Laplace.reinit(sparsity_pattern_velocity);
  vel_Advection.reinit(sparsity_pattern_velocity);

  MatrixCreator::create_mass_matrix(dof_handler_velocity,
                                    quadrature_velocity,
                                    vel_Mass);
  MatrixCreator::create_laplace_matrix(dof_handler_velocity,
                                       quadrature_velocity,
                                       vel_Laplace);
}


template <int dim>
void NavierStokesProjection<dim>::initialize_pressure_matrices()
{
  {
    DynamicSparsityPattern dsp(dof_handler_pressure.n_dofs(),
                               dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp);
    sparsity_pattern_pressure.copy_from(dsp);
    std::ofstream out ("results_original/p_sparsity_pattern.gpl");
    sparsity_pattern_pressure.print_gnuplot(out);
  }

  pres_Laplace.reinit(sparsity_pattern_pressure);
  pres_iterative.reinit(sparsity_pattern_pressure);
  pres_Mass.reinit(sparsity_pattern_pressure);

  MatrixCreator::create_laplace_matrix(dof_handler_pressure,
                                       quadrature_pressure,
                                       pres_Laplace);
  MatrixCreator::create_mass_matrix(dof_handler_pressure,
                                    quadrature_pressure,
                                    pres_Mass);
}

}  // namespace Step35


// explicit instantiations

template void Step35::NavierStokesProjection<2>::initialize_velocity_matrices();
template void Step35::NavierStokesProjection<3>::initialize_velocity_matrices();

template void Step35::NavierStokesProjection<2>::initialize_pressure_matrices();
template void Step35::NavierStokesProjection<3>::initialize_pressure_matrices();
