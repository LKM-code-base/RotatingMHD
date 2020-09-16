#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <string>

namespace RMHD
{

using namespace dealii;

template <int dim>
class Step35 : public Problem<dim>
{
public:
  Step35(const RunTimeParameters::ParameterSet &parameters);

  void run(const bool         flag_verbose_output           = false,
           const unsigned int terminal_output_periodicity   = 10,
           const unsigned int graphical_output_periodicity  = 10);
private:

  std::vector<types::boundary_id>             boundary_ids;

  Entities::VectorEntity<dim>                 velocity;

  Entities::ScalarEntity<dim>                 pressure;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;
  
  EquationData::Step35::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::Step35::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;
  EquationData::Step35::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;

  void make_grid(const unsigned int n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();
  void postprocessing(const bool flag_point_evaluation);
  void output();
  void update_solution_vectors();
  void point_evaluation(const Point<dim>   &point) const;
};

template <int dim>
Step35<dim>::Step35(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(),
velocity(parameters.p_fe_degree + 1, this->triangulation),
pressure(parameters.p_fe_degree, this->triangulation),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters, velocity, pressure, time_stepping),
inflow_boundary_condition(parameters.time_stepping_parameters.start_time),
velocity_initial_conditions(parameters.time_stepping_parameters.start_time),
pressure_initial_conditions(parameters.time_stepping_parameters.start_time)
{
  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup();
  initialize();

  pcout << "Time step: " << time_stepping.get_next_step_size() << std::endl;
}

template <int dim>
void Step35<dim>::make_grid(const unsigned int n_global_refinements)
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);

  {
    std::string   filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  this->triangulation.refine_global(n_global_refinements);

  boundary_ids = this->triangulation.get_boundary_ids();

  this->pcout << "Number of refines                     = "
              << n_global_refinements << std::endl;
  this->pcout << "Number of active cells                = "
              << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void Step35<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  
  this->pcout << "Number of velocity degrees of freedom = "
              << velocity.dof_handler.n_dofs()
              << std::endl
              << "Number of pressure degrees of freedom = "
              << pressure.dof_handler.n_dofs()
              << std::endl;
}

template <int dim>
void Step35<dim>::setup_constraints()
{
  velocity.constraints.clear();
  velocity.constraints.reinit(velocity.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(velocity.dof_handler,
                                          velocity.constraints);
  for (const auto &boundary_id : boundary_ids)
    switch (boundary_id)
    {
      case 1:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    Functions::ZeroFunction<dim>(dim),
                                    velocity.constraints);
        break;
      case 2:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    inflow_boundary_condition,
                                    velocity.constraints);
        break;
      case 3:
      {
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert(boundary_id);
        VectorTools::compute_normal_flux_constraints(
                                    velocity.dof_handler,
                                    0,
                                    no_normal_flux_boundaries,
                                    velocity.constraints);
        break;
      }
      case 4:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    Functions::ZeroFunction<dim>(dim),
                                    velocity.constraints);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  velocity.constraints.close();

  pressure.constraints.clear();
  pressure.constraints.reinit(pressure.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(pressure.dof_handler,
                                          pressure.constraints);
  VectorTools::interpolate_boundary_values(
                                      pressure.dof_handler,
                                      3,
                                      Functions::ZeroFunction<dim>(),
                                      pressure.constraints);
  pressure.constraints.close();
}

template <int dim>
void Step35<dim>::initialize()
{
  this->set_initial_conditions(velocity, 
                               velocity_initial_conditions, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_conditions, 
                               time_stepping);
  velocity.solution = velocity.old_solution;
  pressure.solution = pressure.old_solution;
  output();
}

template <int dim>
void Step35<dim>::postprocessing(const bool flag_point_evaluation)
{
  static Point<dim> evaluation_point(2.0, 3.0);
  if (flag_point_evaluation)
  {
    point_evaluation(evaluation_point);
  }
}

template <int dim>
void Step35<dim>::output()
{
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(velocity.dof_handler,
                           velocity.solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure.solution, 
                           "Pressure");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

template <int dim>
void Step35<dim>::update_solution_vectors()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
  navier_stokes.update_internal_entities();
}

template <int dim>
void Step35<dim>::run(
              const bool          flag_verbose_output,
              const unsigned int  terminal_output_periodicity,
              const unsigned int  graphical_output_periodicity)
{
  (void)flag_verbose_output;

  /*
   * What is going on here? The fact that the initial time step is first order
   * time step
   */
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
      // snapshot stage
    pcout << "Desired time step: " << navier_stokes.compute_next_time_step() << std::endl;

    time_stepping.set_desired_next_step_size(
                              navier_stokes.compute_next_time_step());

    // update stage
    time_stepping.update_coefficients();

    navier_stokes.solve(time_stepping.get_step_number());

    // snapshot stage
    postprocessing((time_stepping.get_step_number() % 
                    terminal_output_periodicity == 0) ||
                   (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() % 
          graphical_output_periodicity == 0) ||
        (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()))
      output();

    update_solution_vectors();
    
    time_stepping.advance_time();
  }
}

template <int dim>
void Step35<dim>::
point_evaluation(const Point<dim>   &point) const
{
  /*
   * Is the point evaluation not duplicate? You have implemented
   * a function mpi_point_value in auxiliary_functions.cc?
   */
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,Point<dim>>
  cell_point =
  GridTools::find_active_cell_around_point(StaticMappingQ1<dim, dim>::mapping,
                                           velocity.dof_handler,
                                           point);
  if (cell_point.first->is_locally_owned())
  {
    Vector<double> point_value_velocity(dim);
    VectorTools::point_value(velocity.dof_handler,
                            velocity.solution,
                            point,
                            point_value_velocity);

    const double point_value_pressure
    = VectorTools::point_value(pressure.dof_handler,
                              pressure.solution,
                              point);
    std::cout << "Step = " << std::setw(2)
              << time_stepping.get_step_number()
              << " Time = " << std::noshowpos << std::scientific
              << time_stepping.get_next_time()
              << " Velocity = (" << std::showpos << std::scientific
              << point_value_velocity[0]
              << ", " << std::showpos << std::scientific
              << point_value_velocity[1]
              << ") Pressure = " << std::showpos << std::scientific
              << point_value_pressure
              << " Time step = " << std::showpos << std::scientific
              << time_stepping.get_next_step_size() << std::endl;
  }
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

      RunTimeParameters::ParameterSet parameter_set("step-35.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      Step35<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_verbose_output, 
                     parameter_set.terminal_output_interval,
                     parameter_set.graphical_output_interval);
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
