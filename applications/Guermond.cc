/*!
 *@file Guermond
 *@brief The .cc file replicating the numerical test of section
  3.7.2 of the Guermond paper.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include <string>
#include <cmath>

namespace RMHD
{
  using namespace dealii;

template <int dim>
struct ConvergenceAnalysisData
{
  ConvergenceTable                convergence_table;

  const std::string               entity_name;

  const Entities::EntityBase<dim> &entity;

  const Function<dim>             &exact_solution;

  ConvergenceAnalysisData(const Entities::EntityBase<dim> &entity,
                          const Function<dim>             &exact_solution,
                          const std::string entity_name = "Entity")
  :
  entity_name(entity_name),
  entity(entity),
  exact_solution(exact_solution)
  {
    convergence_table.declare_column("level");
    convergence_table.declare_column("dt");
    convergence_table.declare_column("cells");
    convergence_table.declare_column("dofs");
    convergence_table.declare_column("hmax");
    convergence_table.declare_column("L2");
    convergence_table.declare_column("H1");
    convergence_table.declare_column("Linfty");
    convergence_table.set_scientific("dt", true);
    convergence_table.set_scientific("hmax", true);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_precision("dt", 2);
    convergence_table.set_precision("hmax", 2);
    convergence_table.set_precision("L2", 6);
    convergence_table.set_precision("H1", 6);
    convergence_table.set_precision("Linfty", 6);
  }

  void update_table(const unsigned int  &level,
                    const double        &time_step,
                    const bool          &flag_spatial_convergence)
  {
  Vector<double> cellwise_difference(
    entity.dof_handler.get_triangulation().n_active_cells());

  QGauss<dim>    quadrature_formula(entity.fe_degree + 2);
  const QTrapez<1>     trapezoidal_rule;
  const QIterated<dim> iterated_quadrature_rule(trapezoidal_rule,
                                                entity.fe_degree * 2 + 1);
  
  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::L2_norm);
  
  const double L2_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::L2_norm);

  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::H1_norm);
  
  const double H1_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::H1_norm);

  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    iterated_quadrature_rule,
                                    VectorTools::Linfty_norm);
  
  const double Linfty_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::Linfty_norm);

  convergence_table.add_value("level", level);
  convergence_table.add_value("dt", time_step);
  convergence_table.add_value("cells", entity.dof_handler.get_triangulation().n_active_cells());
  convergence_table.add_value("dofs", entity.dof_handler.n_dofs());
  convergence_table.add_value("hmax", GridTools::maximal_cell_diameter(entity.dof_handler.get_triangulation()));
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

  std::string reference_column = (flag_spatial_convergence) ? 
                                  "hmax" : "dt";

  convergence_table.evaluate_convergence_rates(
                              "L2",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  convergence_table.evaluate_convergence_rates(
                              "H1",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  convergence_table.evaluate_convergence_rates(
                              "Linfty",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  }

  void print_table()
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
    std::cout << std::endl;
    std::cout 
      << "                               " << entity_name 
      << " convergence table" << std::endl
      << "==============================================="
      << "===============================================" 
      << std::endl;
    convergence_table.write_text(std::cout);
    }
  }
};

/*!
 * @class Guermond
 * @brief This class solves the Taylor-Green vortex benchmark
 * @todo Add documentation
 */  
template <int dim>
class Guermond : public Problem<dim>
{
public:
  Guermond(const RunTimeParameters::ParameterSet &parameters);
  void run(const bool flag_convergence_test);
private:
  const RunTimeParameters::ParameterSet       &prm;

  ConditionalOStream                          pcout;

  parallel::distributed::Triangulation<dim>   triangulation;

  std::vector<types::boundary_id>             boundary_ids;

  Entities::VectorEntity<dim>                 velocity;

  Entities::ScalarEntity<dim>                 pressure;

  LinearAlgebra::MPI::Vector                  velocity_error;

  LinearAlgebra::MPI::Vector                  pressure_error;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;

  EquationData::Guermond::VelocityExactSolution<dim>         
                                      velocity_exact_solution;

  EquationData::Guermond::PressureExactSolution<dim>         
                                      pressure_exact_solution;

  EquationData::Guermond::BodyForce<dim>      body_force;

  ConvergenceAnalysisData<dim>                velocity_convergence_table;

  ConvergenceAnalysisData<dim>                pressure_convergence_table;

  const bool                                  flag_set_exact_pressure_constant;

  const bool                                  flag_square_domain;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_entities();


  void solve(const unsigned int &level);
};

template <int dim>
Guermond<dim>::Guermond(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(),
prm(parameters),
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
triangulation(MPI_COMM_WORLD,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
velocity(parameters.p_fe_degree + 1, triangulation),
pressure(parameters.p_fe_degree, triangulation),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters, velocity, pressure, time_stepping),
velocity_exact_solution(parameters.time_stepping_parameters.start_time),
pressure_exact_solution(parameters.time_stepping_parameters.start_time),
body_force(parameters.Re, parameters.time_stepping_parameters.start_time),
velocity_convergence_table(velocity, velocity_exact_solution, "Velocity"),
pressure_convergence_table(pressure, pressure_exact_solution, "Pressure"),
flag_set_exact_pressure_constant(true),
flag_square_domain(true)
{
  navier_stokes.set_body_force(body_force);
}

template <int dim>
void Guermond<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  if (flag_square_domain)
    GridGenerator::hyper_cube(triangulation,
                              0.0,
                              1.0,
                              true);
  else
  {
    const double radius = 0.5;
    GridGenerator::hyper_ball(triangulation,
                              Point<dim>(),
                              radius,
                              true);
  }

  triangulation.refine_global(n_global_refinements);
  boundary_ids = triangulation.get_boundary_ids();
}

template <int dim>
void Guermond<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  pcout     << "  Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;
  pcout     << "  Number of velocity degrees of freedom = " 
            << velocity.dof_handler.n_dofs()
            << std::endl
            << "  Number of pressure degrees of freedom = " 
            << pressure.dof_handler.n_dofs()
            << std::endl;
}

template <int dim>
void Guermond<dim>::setup_constraints()
{
  for (const auto& boundary_id : boundary_ids)
    velocity.boundary_conditions.set_dirichlet_bcs(
      boundary_id,
      std::shared_ptr<Function<dim>> 
        (new EquationData::Guermond::VelocityExactSolution<dim>(
          prm.time_stepping_parameters.start_time)),
      true);
  
  velocity.apply_boundary_conditions();
  pressure.apply_boundary_conditions();
}

template <int dim>
void Guermond<dim>::initialize()
{
  this->set_initial_conditions(velocity, 
                               velocity_exact_solution, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_exact_solution, 
                               time_stepping);
  //navier_stokes.initialize();
}

template <int dim>
void Guermond<dim>::postprocessing(const bool flag_point_evaluation)
{
  if (flag_set_exact_pressure_constant)
  {
    LinearAlgebra::MPI::Vector  analytical_pressure(pressure.solution);
    {
      #ifdef USE_PETSC_LA
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure.locally_owned_dofs, MPI_COMM_WORLD);
      #else
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure.locally_owned_dofs);
      #endif
      VectorTools::project(pressure.dof_handler,
                          pressure.constraints,
                          QGauss<dim>(pressure.fe_degree + 2),
                          pressure_exact_solution,
                          tmp_analytical_pressure);

      analytical_pressure = tmp_analytical_pressure;
    }
    {
      LinearAlgebra::MPI::Vector distributed_analytical_pressure;
      LinearAlgebra::MPI::Vector distributed_numerical_pressure;
      #ifdef USE_PETSC_LA
        distributed_analytical_pressure.reinit(pressure.locally_owned_dofs,
                                        MPI_COMM_WORLD);
      #else
        distributed_analytical_pressure.reinit(pressure.locally_owned_dofs,
                                        pressure.locally_relevant_dofs,
                                        MPI_COMM_WORLD,
                                        true);
      #endif
      distributed_numerical_pressure.reinit(distributed_analytical_pressure);

      distributed_analytical_pressure = analytical_pressure;
      distributed_numerical_pressure  = pressure.solution;

      distributed_numerical_pressure.add(  
        distributed_analytical_pressure.mean_value() -
        distributed_numerical_pressure.mean_value());

      pressure.solution = distributed_numerical_pressure;
    }
  }

  if (flag_point_evaluation)
  {
    std::cout.precision(1);
    pcout << "  Step = " 
          << std::setw(4) 
          << time_stepping.get_step_number() 
          << " Time = " 
          << std::noshowpos << std::scientific
          << time_stepping.get_next_time()
          << " Progress ["
          << std::setw(5) 
          << std::fixed
          << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
          << "%] \r";
  }
}

template <int dim>
void Guermond<dim>::output()
{
  this->compute_error(velocity_error,
                       velocity,
                       velocity_exact_solution);
  this->compute_error(pressure_error,
                       pressure,
                       pressure_exact_solution);

  std::vector<std::string> names(dim, "velocity");
  std::vector<std::string> error_name(dim, "velocity_error");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(velocity.dof_handler,
                           velocity.solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(velocity.dof_handler,
                           velocity_error,
                           error_name, 
                           component_interpretation);
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure.solution, 
                           "pressure");
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure_error, 
                           "pressure_error");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

template <int dim>
void Guermond<dim>::update_entities()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
  navier_stokes.update_internal_entities();
}

template <int dim>
void Guermond<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  velocity_error.reinit(velocity.solution);
  pressure_error.reinit(pressure.solution);
  navier_stokes.setup(true);
  initialize();

  // Advances the time to t^{k-1}, either t^0 or t^1
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  // Outputs the fields at t_0, i.e. the initial conditions.
  { 
    velocity.solution = velocity.old_old_solution;
    pressure.solution = pressure.old_old_solution;
    velocity_exact_solution.set_time(time_stepping.get_start_time());
    pressure_exact_solution.set_time(time_stepping.get_start_time());
    output();
    velocity.solution = velocity.old_solution;
    pressure.solution = pressure.old_solution;
    velocity_exact_solution.set_time(time_stepping.get_start_time() + 
                                     time_stepping.get_next_step_size());
    pressure_exact_solution.set_time(time_stepping.get_start_time() + 
                                     time_stepping.get_next_step_size());
    output();   
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The whole while-scope is at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
                              navier_stokes.compute_next_time_step());
    
    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();
    
    // Updates the functions and the constraints to t^{k}
    velocity_exact_solution.set_time(time_stepping.get_next_time());
    pressure_exact_solution.set_time(time_stepping.get_next_time());
    body_force.set_time(time_stepping.get_next_time());
    velocity.boundary_conditions.set_time(time_stepping.get_next_time());
    velocity.update_boundary_conditions();
    //update_boundary_values();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve(time_stepping.get_step_number());

    // Snapshot stage, all time calls should be done with get_next_time()
    postprocessing((time_stepping.get_step_number() %
                    prm.terminal_output_interval == 0) ||
                    (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          prm.graphical_output_interval == 0) ||
        (time_stepping.get_next_time() == 
          time_stepping.get_end_time()))
      output();
    
    // Advances the time to t^{k}
    update_entities();
    time_stepping.advance_time();
  }

  Assert(time_stepping.get_current_time() == velocity_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the velocity function"));
  Assert(time_stepping.get_current_time() == pressure_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the pressure function"));
  
  velocity_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), prm.flag_spatial_convergence_test);
  pressure_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), prm.flag_spatial_convergence_test);
  
  pcout << std::endl;
  pcout << std::endl;
}

template <int dim>
void Guermond<dim>::run(const bool flag_convergence_test)
{
  make_grid(prm.initial_refinement_level);
  if (flag_convergence_test)
    for (unsigned int level = prm.initial_refinement_level; 
          level <= prm.final_refinement_level; ++level)
    {
      std::cout.precision(1);
      pcout << "Solving until t = " 
            << std::fixed << time_stepping.get_end_time()
            << " with a refinement level of " << level << std::endl;
      time_stepping.restart();
      solve(level);
      triangulation.refine_global();
    }
  else
  {
    for (unsigned int cycle = 0; 
         cycle < prm.temporal_convergence_cycles; ++cycle)
    {
      double time_step = prm.time_stepping_parameters.initial_time_step *
                         pow(prm.time_step_scaling_factor, cycle);
      std::cout.precision(1);
      pcout << "Solving until t = " 
            << std::fixed << time_stepping.get_end_time()
            << " with a refinement level of " << prm.initial_refinement_level << std::endl;
      time_stepping.restart();
      time_stepping.set_desired_next_step_size(time_step);
      solve(prm.initial_refinement_level);
    }
  }
  
  velocity_convergence_table.print_table();
  pressure_convergence_table.print_table();
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

      RunTimeParameters::ParameterSet parameter_set("Guermond.prm");

      deallog.depth_console(parameter_set.verbose ? 2 : 0);

      Guermond<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_spatial_convergence_test);
  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}
