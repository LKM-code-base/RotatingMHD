#include <rotatingMHD/projection_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  diffusion_step_assembly()
  {
    /*Extrapolate velocity by a Taylor expansion
      v^{\textrm{k}+1} \approx 2 * v^\textrm{k} - v^{\textrm{k}-1 */

    v_extrapolated.equ(1.0 + dt_n / dt_n_m1, v_n);
    v_extrapolated.add(-dt_n / dt_n_m1, v_n_m1);
    
    /*Define auxiliary pressure
      p^{\#} = p^\textrm{k} + 4/3 * \phi^\textrm{k} 
                - 1/3 * \phi^{\textrm{k}-1} 
      Note: The signs are inverted since p_gradient_matrix is
      defined as negative */

    p_tmp.equ(-1., p_n);
    p_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_m1);

    /* System matrix setup */
    assemble_v_advection_matrix();
    if (flag_adpative_time_step)
    {
      v_mass_plus_laplace_matrix = 0.;
      v_mass_plus_laplace_matrix.add(1.0 / Re, v_laplace_matrix);
      v_mass_plus_laplace_matrix.add( (2.0 * dt_n + dt_n_m1) / (dt_n * (dt_n + dt_n_m1)), v_mass_matrix);
    }
    v_system_matrix.copy_from(v_mass_plus_laplace_matrix);
    v_system_matrix.add(1., v_advection_matrix);

    /* Right hand side setup */
    v_rhs = 0.;
    v_tmp.equ( ( dt_n + dt_n_m1 ) / (dt_n * dt_n_m1), v_n);
    v_tmp.add(-(dt_n * dt_n) / (dt_n * dt_n_m1 * (dt_n + dt_n_m1)), v_n_m1);
    v_mass_matrix.vmult_add(v_rhs, v_tmp);
    p_gradient_matrix.vmult_add(v_rhs, p_tmp);

    /* Update for the next time step */
    v_n_m1 = v_n;

    v_constraints.condense(v_system_matrix, v_rhs);
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::
  diffusion_step_solve(const bool reinit_prec)
  {
    if (reinit_prec)
      v_preconditioner.initialize(v_system_matrix,
                                      SparseILU<double>::AdditionalData(
                                        solver_diag_strength, 
                                        solver_off_diagonals));

    SolverControl solver_control(solver_max_iterations, 
                                 solver_tolerance * v_rhs.l2_norm());
    SolverGMRES<> gmres(solver_control,
                        SolverGMRES<>::AdditionalData(solver_krylov_size));
    gmres.solve(v_system_matrix, 
                v_n, 
                v_rhs, 
                v_preconditioner);
    v_constraints.distribute(v_n);
  }
}

template void Step35::NavierStokesProjection<2>::diffusion_step_assembly();
template void Step35::NavierStokesProjection<3>::diffusion_step_assembly();
template void Step35::NavierStokesProjection<2>::diffusion_step_solve(const bool);
template void Step35::NavierStokesProjection<3>::diffusion_step_solve(const bool);