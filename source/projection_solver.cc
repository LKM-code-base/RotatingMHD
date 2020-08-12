/*
 * projection_solver.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>
#include <rotatingMHD/time_discretization.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <iostream>

namespace Step35
{

template <int dim>
NavierStokesProjection<dim>::
NavierStokesProjection(const RunTimeParameters::ParameterSet &data)
  : mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    projection_method(data.projection_method),
    dt_n(data.dt),
    dt_n_minus_1(data.dt),
    t_0(data.t_0),
    T(data.T),
    Re(data.Re),
    VSIMEX(2),
    time_stepping(2, 
                  {data.vsimex_input_gamma, data.vsimex_input_c}, 
                  dt_n, 
                  T, 
                  dt_n),
    inflow_boundary_condition(data.t_0),
    triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
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
    pcout
      << " WARNING: The chosen pair of finite element spaces is not stable."
      << std::endl
      << " The obtained results will be nonsense" << std::endl;

  AssertThrow(!((dt_n <= 0.) || (dt_n > .5 * T)), 
              ExcInvalidTimeStep(dt_n, .5 * T));
  /* Time is started one time step earlier and advance in order to
     populate a private entity of the DiscreteTime class */
  time_stepping.advance_time();
  time_stepping.update_coefficients();
  time_stepping.get_coefficients(VSIMEX);
  //VSIMEX.output();
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
  ConditionalOStream verbose_cout(
    std::cout, 
    ((flag_verbose_output) && 
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) ));

  Point<dim> evaluation_point(2.0, 3.0);
  
  inflow_boundary_condition.set_time(2. * dt_n);
  output_results(1);
  unsigned int n = 2;
  
  for (;time_stepping.get_current_time() <= time_stepping.get_end_time();
        time_stepping.advance_time())
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

    if ((n % output_interval == 0) || time_stepping.is_at_end())
      {
        verbose_cout << "Plotting Solution" << std::endl;
        output_results(n);
      }

    if ((flag_adpative_time_step) && (n > 20))
      update_time_step();

    time_stepping.set_desired_next_step_size(dt_n);
    time_stepping.update_coefficients();
    time_stepping.get_coefficients(VSIMEX);
    //VSIMEX.output();
    inflow_boundary_condition.advance_time(
                                    time_stepping.get_next_step_size());

    if ((n % output_interval == 0) || time_stepping.is_at_end())
      point_evaluation(evaluation_point, n, time_stepping);

    ++n;

    if (time_stepping.is_at_end())
      break;
  }
}

template <int dim>
void NavierStokesProjection<dim>::
point_evaluation(const Point<dim>   &point,
                 unsigned int       time_step,
                 DiscreteTime       time) const
{
const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> cell_point =
    GridTools::find_active_cell_around_point(StaticMappingQ1<dim, dim>::mapping, 
                                              velocity_dof_handler, 
                                              point);
if (cell_point.first->is_locally_owned())
{
  Vector<double> point_value_velocity(dim);
  VectorTools::point_value(velocity_dof_handler,
                          velocity_n,
                          point,
                          point_value_velocity);

  const double point_value_pressure
  = VectorTools::point_value(pressure_dof_handler,
                            pressure_n,
                            point);
  std::cout << "Step = " 
            << std::setw(2) 
            << time_step 
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
}
}

}  // namespace Step35

// explicit instantiations

template class Step35::NavierStokesProjection<2>;
template class Step35::NavierStokesProjection<3>;

