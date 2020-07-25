#include <rotatingMHD/projection_solver.h>

#include <deal.II/numerics/matrix_tools.h>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
assemble_velocity_matrices()
{
  MatrixCreator::create_mass_matrix(velocity_dof_handler,
                                    velocity_quadrature_formula,
                                    velocity_mass_matrix);
  MatrixCreator::create_laplace_matrix(velocity_dof_handler,
                                       velocity_quadrature_formula,
                                       velocity_laplace_matrix);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_pressure_matrices()
{
  MatrixCreator::create_laplace_matrix(pressure_dof_handler,
                                       pressure_quadrature_formula,
                                       pressure_laplace_matrix);
  MatrixCreator::create_mass_matrix(pressure_dof_handler,
                                    pressure_quadrature_formula,
                                    pressure_mass_matrix);
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::assemble_velocity_matrices();
template void Step35::NavierStokesProjection<3>::assemble_velocity_matrices();
template void Step35::NavierStokesProjection<2>::assemble_pressure_matrices();
template void Step35::NavierStokesProjection<3>::assemble_pressure_matrices();
