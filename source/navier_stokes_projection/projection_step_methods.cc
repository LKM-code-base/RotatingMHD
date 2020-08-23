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
  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  TrilinosWrappers::MPI::Vector distributed_phi(pressure_rhs);
  distributed_phi = phi;

  if (reinit_prec)
    projection_step_preconditioner.initialize(pressure_laplace_matrix);

  SolverControl solvercontrol(pressure_laplace_matrix.m(), 
                              solver_tolerance * pressure_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector>    cg(solvercontrol);
  cg.solve(pressure_laplace_matrix, 
           distributed_phi, 
           pressure_rhs, 
           projection_step_preconditioner);
  pressure.constraints.distribute(distributed_phi);
  distributed_phi *= VSIMEX.alpha[2];
  phi = distributed_phi;
}
}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step();
template void RMHD::NavierStokesProjection<2>::solve_projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_projection_step(const bool);