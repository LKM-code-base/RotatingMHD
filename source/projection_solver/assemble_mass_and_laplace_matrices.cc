#include <rotatingMHD/projection_solver.h>

#include <deal.II/numerics/matrix_tools.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_v_matrices()
  {
    MatrixCreator::create_mass_matrix(v_dof_handler,
                                      v_quadrature_formula,
                                      v_mass_matrix);
    MatrixCreator::create_laplace_matrix(v_dof_handler,
                                         v_quadrature_formula,
                                         v_laplace_matrix);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_p_matrices()
  {
    MatrixCreator::create_laplace_matrix(p_dof_handler,
                                         p_quadrature_formula,
                                         p_laplace_matrix);
    MatrixCreator::create_mass_matrix(p_dof_handler,
                                      p_quadrature_formula,
                                      p_mass_matrix);
  }
}

template void Step35::NavierStokesProjection<2>::assemble_v_matrices();
template void Step35::NavierStokesProjection<3>::assemble_v_matrices();
template void Step35::NavierStokesProjection<2>::assemble_p_matrices();
template void Step35::NavierStokesProjection<3>::assemble_p_matrices();
