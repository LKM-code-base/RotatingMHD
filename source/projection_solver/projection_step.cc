#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  projection_step_assembly(const bool reinit_prec)
  {
    /* System matrix setup */
    p_system_matrix.copy_from(p_laplace_matrix);

    /* Right hand side setup */
    p_rhs = 0.;
    p_gradient_matrix.Tvmult_add(p_rhs, v_n);

    /* Update for the next time step */
    phi_n_m1 = phi_n;

    static std::map<types::global_dof_index, double> bval;
    if (reinit_prec)
      VectorTools::interpolate_boundary_values(p_dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(),
                                               bval);

    p_constraints.condense(p_system_matrix, p_rhs);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  projection_step_solve(const bool reinit_prec)
  {
    if (reinit_prec)
      p_preconditioner.initialize(p_system_matrix,
                                   SparseILU<double>::AdditionalData(
                                     solver_diag_strength, 
                                     solver_off_diagonals));

    SolverControl solvercontrol(solver_max_iterations, 
                                solver_tolerance * p_rhs.l2_norm());
    SolverCG<>    cg(solvercontrol);
    cg.solve(p_system_matrix, 
             phi_n, 
             p_rhs, 
             p_preconditioner);
    p_constraints.distribute(phi_n);
    phi_n *= ((2.0 * dt_n + dt_n_m1) / (dt_n * (dt_n + dt_n_m1)));
  }
}

template void Step35::NavierStokesProjection<2>::projection_step_assembly(const bool);
template void Step35::NavierStokesProjection<3>::projection_step_assembly(const bool);
template void Step35::NavierStokesProjection<2>::projection_step_solve(const bool);
template void Step35::NavierStokesProjection<3>::projection_step_solve(const bool);