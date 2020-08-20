#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step()
{
  /* System matrix setup */
  // System matrix is constant and assembled in the
  // NavierStokesProjection constructor.

  /* Right hand side setup */
  assemble_projection_step_rhs();
}

template <int dim>
void NavierStokesProjection<dim>::
solve_projection_step(const bool reinit_prec)
{
  TrilinosWrappers::MPI::Vector distributed_phi_n(pressure_rhs);
  distributed_phi_n = phi_n;

  if (reinit_prec)
    projection_step_preconditioner.initialize(pressure_laplace_matrix);

  SolverControl solvercontrol(pressure_laplace_matrix.m(), 
                              solver_tolerance * pressure_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector>    cg(solvercontrol);
  cg.solve(pressure_laplace_matrix, 
           distributed_phi_n, 
           pressure_rhs, 
           projection_step_preconditioner);
  pressure.constraints.distribute(distributed_phi_n);
  distributed_phi_n *= VSIMEX.alpha[2];
  phi_n = distributed_phi_n;
}
}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step();
template void RMHD::NavierStokesProjection<2>::solve_projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_projection_step(const bool);