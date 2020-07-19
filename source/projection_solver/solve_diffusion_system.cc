/*
 * solve_diffusion_system.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
{
    pres_tmp.equ(-1., pres_n);
    pres_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_minus_1);

    assemble_advection_term();

    //for (unsigned int d = 0; d < dim; ++d)
    //  {

    force = 0.;
    v_tmp.equ(2. / dt, u_n);
    v_tmp.add(-.5 / dt, u_n_minus_1);
    vel_Mass.vmult_add(force, v_tmp);

    pres_Diff.vmult_add(force, pres_tmp);
    u_n_minus_1 = u_n;

    vel_it_matrix.copy_from(vel_Laplace_plus_Mass);
    vel_it_matrix.add(1., vel_Advection);

    boundary_values.clear();
    for (const auto &boundary_id : boundary_ids)
      {
        switch (boundary_id)
        {
          case 1:
            VectorTools::interpolate_boundary_values(
                dof_handler_velocity,
                boundary_id,
                Functions::ZeroFunction<dim>(dim),
                boundary_values);
            break;
          case 2:
            VectorTools::interpolate_boundary_values(dof_handler_velocity,
                                                     boundary_id,
                                                     vel_exact,
                                                     boundary_values);
            break;
          case 3:
            {
              ComponentMask   component_mask(dim, true);
              component_mask.set(0, false);
              //if (d != 0)
                VectorTools::interpolate_boundary_values(
                    dof_handler_velocity,
                    boundary_id,
                    Functions::ZeroFunction<dim>(dim),
                    boundary_values,
                    component_mask);
                break;
            }
          case 4:
            VectorTools::interpolate_boundary_values(
                dof_handler_velocity,
                boundary_id,
                Functions::ZeroFunction<dim>(dim),
                boundary_values);
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      }
    MatrixTools::apply_boundary_values(boundary_values,
                                       vel_it_matrix,
                                       u_n,
                                       force);
    //  }

    /*
    Threads::TaskGroup<void> tasks;
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (reinit_prec)
          prec_velocity[d].initialize(vel_it_matrix[d],
                                      SparseILU<double>::AdditionalData(
                                        vel_diag_strength, vel_off_diagonals));
        tasks += Threads::new_task(
          &NavierStokesProjection<dim>::diffusion_component_solve, *this, d);
      }
    tasks.join_all();
     */

    if (reinit_prec)
      prec_velocity.initialize(vel_it_matrix,
                               SparseILU<double>::AdditionalData(
                                   vel_diag_strength, vel_off_diagonals));
    diffusion_component_solve();
}

template <int dim>
void NavierStokesProjection<dim>::diffusion_component_solve()
{
  SolverControl solver_control(vel_max_its, vel_eps * force.l2_norm());
  SolverGMRES<> gmres(solver_control,
                      SolverGMRES<>::AdditionalData(vel_Krylov_size));
  gmres.solve(vel_it_matrix, u_n, force, prec_velocity);
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::diffusion_step(const bool);
template void Step35::NavierStokesProjection<3>::diffusion_step(const bool);

template void Step35::NavierStokesProjection<2>::diffusion_component_solve();
template void Step35::NavierStokesProjection<3>::diffusion_component_solve();
