/*
 * projection_solver.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>

namespace Step35
{

  template <int dim>
  NavierStokesProjection<dim>::
  NavierStokesProjection(const RunTimeParameters::ParameterSet &data)
    : projection_method(data.projection_method),
      p_fe_degree(data.p_fe_degree),
      v_fe_degree(p_fe_degree + 1),
      dt(data.dt),
      t_0(data.t_0),
      T(data.T),
      Re(data.Re),
      inflow_bc(data.t_0),
      v_fe(FE_Q<dim>(v_fe_degree), dim),
      p_fe(p_fe_degree),
      v_dof_handler(triangulation),
      p_dof_handler(triangulation),
      p_quadrature_formula(p_fe_degree + 1),
      v_quadrature_formula(v_fe_degree + 1),
      solver_max_iterations(data.solver_max_iterations),
      solver_krylov_size(data.solver_krylov_size),
      solver_off_diagonals(data.solver_off_diagonals),
      solver_update_preconditioner(data.solver_update_preconditioner),
      solver_tolerance(data.solver_tolerance),
      solver_diag_strength(data.solver_diag_strength)
  {
    if (p_fe_degree < 1)
      std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;

    AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

    make_grid(data.n_global_refinements);
    setup_dofs();
    initialize();
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  run(const bool  flag_verbose_output,
      const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout(std::cout, flag_verbose_output);

    const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);

    inflow_bc.set_time(2. * dt);
    output_results(1);

    for (unsigned int n = 2; n <= n_steps; ++n)
      {
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n);
          }
        Point<dim> evaluation_point;
        evaluation_point(0) = 2.0;
        evaluation_point(1) = 3.0;

        Vector<double> point_value_velocity(dim);
        VectorTools::point_value(v_dof_handler,
                                v_n,
                                evaluation_point,
                                point_value_velocity);

        const double point_value_pressure
        = VectorTools::point_value(p_dof_handler,
                                  p_n,
                                  evaluation_point);
        std::cout << "Step = " 
                  << std::setw(2) 
                  << n 
                  << " Time = " 
                  << std::noshowpos << std::scientific
                  << (n * dt)
                  << " Velocity = (" 
                  << std::showpos << std::scientific
                  << point_value_velocity[0] 
                  << ", "
                  << std::showpos << std::scientific
                  << point_value_velocity[1] 
                  << ") Pressure = "
                  << std::showpos << std::scientific
                  << point_value_pressure << std::endl;

        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % solver_update_preconditioner == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;
        diffusion_step_assembly();
        diffusion_step_solve((n % solver_update_preconditioner == 0) || (n == 2));

        verbose_cout << "  Projection Step" << std::endl;
        projection_step_assembly((n == 2));
        projection_step_solve((n==2));
        
        verbose_cout << "  Updating the Pressure" << std::endl;
        correction_step((n == 2));

        inflow_bc.advance_time(dt);
      }
    output_results(n_steps);
  }
}  // namespace Step35

// explicit instantiations

template class Step35::NavierStokesProjection<2>;
template class Step35::NavierStokesProjection<3>;

