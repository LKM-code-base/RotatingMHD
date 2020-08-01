#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
projection_step(const bool reinit_prec)
{
  /* Assemble linear system */
  assemble_projection_step();

  /* Update for the next time step */
  phi_n_minus_1 = phi_n;

  /* Solve linear system */
  solve_projection_step(reinit_prec);
}

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step()
{
  /* System matrix setup */
  // System matrix is constant and assembled in the
  // NavierStokesProjection constructor.

  /* Right hand side setup */
  assemble_projection_step_rhs();

  /* Apply boundary conditions and hanging node constraints */
  pressure_constraints.condense(pressure_laplace_matrix, pressure_rhs);
}

template <int dim>
void NavierStokesProjection<dim>::
solve_projection_step(const bool reinit_prec)
{
  if (reinit_prec)
    projection_step_preconditioner.initialize(pressure_laplace_matrix,
                                  SparseILU<double>::AdditionalData(
                                    solver_diag_strength, 
                                    solver_off_diagonals));

  SolverControl solvercontrol(solver_max_iterations, 
                              solver_tolerance * pressure_rhs.l2_norm());
  SolverCG<>    cg(solvercontrol);
  cg.solve(pressure_laplace_matrix, 
           phi_n, 
           pressure_rhs, 
           projection_step_preconditioner);
  pressure_constraints.distribute(phi_n);
  phi_n *= ((2.0 * dt_n + dt_n_minus_1) /     //
            (dt_n * (dt_n + dt_n_minus_1)));
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::projection_step(const bool);
template void Step35::NavierStokesProjection<3>::projection_step(const bool);
template void Step35::NavierStokesProjection<2>::assemble_projection_step();
template void Step35::NavierStokesProjection<3>::assemble_projection_step();
template void Step35::NavierStokesProjection<2>::solve_projection_step(const bool);
template void Step35::NavierStokesProjection<3>::solve_projection_step(const bool);