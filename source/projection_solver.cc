/*
 * projection_solver.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

namespace Step35
{

template <int dim>
NavierStokesProjection<dim>::
NavierStokesProjection(const RunTimeParameters::ParameterSet &data)
  : projection_method(data.projection_method),
    dt_n(data.dt),
    dt_n_minus_1(data.dt),
    t_0(data.t_0),
    T(data.T),
    Re(data.Re),
    inflow_boundary_condition(data.t_0),
    pressure_fe_degree(data.p_fe_degree),
    pressure_fe(pressure_fe_degree),
    pressure_dof_handler(triangulation),
    pressure_quadrature_formula(pressure_fe_degree + 1),
    velocity_fe_degree(pressure_fe_degree + 1),
    velocity_fe(FE_Q<dim>(velocity_fe_degree), dim),
    velocity_dof_handler(triangulation),
    velocity_quadrature_formula(velocity_fe_degree + 1),
    solver_max_iterations(data.solver_max_iterations),
    solver_krylov_size(data.solver_krylov_size),
    solver_off_diagonals(data.solver_off_diagonals),
    solver_update_preconditioner(data.solver_update_preconditioner),
    solver_tolerance(data.solver_tolerance),
    solver_diag_strength(data.solver_diag_strength),
    flag_adpative_time_step(data.flag_adaptive_time_step)
{
  if (pressure_fe_degree < 1)
    std::cout
      << " WARNING: The chosen pair of finite element spaces is not stable."
      << std::endl
      << " The obtained results will be nonsense" << std::endl;

  AssertThrow(!((dt_n <= 0.) || (dt_n > .5 * T)), 
              ExcInvalidTimeStep(dt_n, .5 * T));

  make_grid(data.n_global_refinements);
  setup_dofs();
  setup_matrices_and_vectors();
  assemble_constant_matrices();
  initialize();
}

template <int dim>
void NavierStokesProjection<dim>::
run(const bool  flag_verbose_output,
    const unsigned int output_interval)
{
  ConditionalOStream verbose_cout(std::cout, flag_verbose_output);

  //const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);

  inflow_boundary_condition.set_time(2. * dt_n);
  output_results(1);
  unsigned int n = 2;
  
  for (DiscreteTime time(2 * dt_n, T, dt_n);
        time.get_current_time() <= time.get_end_time();
        time.advance_time())
  {
    verbose_cout << "  Diffusion Step" << std::endl;
    if (n % solver_update_preconditioner == 0)
      verbose_cout << "    With reinitialization of the preconditioner"
                    << std::endl;
    diffusion_step((n % solver_update_preconditioner == 0) || (n == 2));

    verbose_cout << "  Projection Step" << std::endl;
    projection_step((n == 2));
    
    verbose_cout << "  Updating the Pressure" << std::endl;
    pressure_correction((n == 2));

    if ((n % output_interval == 0) || time.is_at_end())
      {
        verbose_cout << "Plotting Solution" << std::endl;
        output_results(n);
      }

    if ((flag_adpative_time_step) && (n > 20))
      update_time_step();

    Point<dim> evaluation_point;
    evaluation_point(0) = 2.0;
    evaluation_point(1) = 3.0;

    Vector<double> point_value_velocity(dim);
    VectorTools::point_value(velocity_dof_handler,
                            velocity_n,
                            evaluation_point,
                            point_value_velocity);

    const double point_value_pressure
    = VectorTools::point_value(pressure_dof_handler,
                              pressure_n,
                              evaluation_point);
    std::cout << "Step = " 
              << std::setw(2) 
              << n 
              << " Time = " 
              << std::noshowpos << std::scientific
              << time.get_current_time()
              << " Velocity = (" 
              << std::showpos << std::scientific
              << point_value_velocity[0] 
              << ", "
              << std::showpos << std::scientific
              << point_value_velocity[1] 
              << ") Pressure = "
              << std::showpos << std::scientific
              << point_value_pressure 
              << " Time step = " 
              << std::showpos << std::scientific
              << dt_n << std::endl;

    time.set_desired_next_step_size(dt_n);
    inflow_boundary_condition.advance_time(dt_n);
    ++n;
    if (time.is_at_end())
      break;
  }
}
}  // namespace Step35

// explicit instantiations

template class Step35::NavierStokesProjection<2>;
template class Step35::NavierStokesProjection<3>;

