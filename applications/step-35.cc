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
#include <memory>
#include <string>

namespace RMHD
{

using namespace dealii;

template <int dim>
class Step35 : public Problem<dim>
{
public:
  Step35(const RunTimeParameters::ParameterSet &parameters);

  void run();
private:
  const RunTimeParameters::ParameterSet       &params;

  std::vector<types::boundary_id>             boundary_ids;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;
  
  EquationData::Step35::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::Step35::VelocityInitialCondition<dim>         
                                      velocity_initial_condition;
  EquationData::Step35::PressureInitialCondition<dim>         
                                      pressure_initial_condition;

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
Problem<dim>(parameters),
params(parameters),
velocity(std::make_shared<Entities::VectorEntity<dim>>(parameters.p_fe_degree + 1,
                                                       this->triangulation,
                                                       "velocity")),
pressure(std::make_shared<Entities::ScalarEntity<dim>>(parameters.p_fe_degree,
                                                       this->triangulation,
                                                       "pressure")),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters,
              velocity,
              pressure,
              time_stepping,
              this->pcout,
              this->computing_timer),
inflow_boundary_condition(parameters.time_stepping_parameters.start_time),
velocity_initial_condition(dim),
pressure_initial_condition()
{
  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity->reinit();
  pressure->reinit();
  initialize();

  this->container.add_entity(velocity);
  this->container.add_entity(pressure, false);
  this->container.add_entity(navier_stokes.phi, false);
}

template <int dim>
void Step35<dim>::make_grid(const unsigned int n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

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

  *(this->pcout) << "Number of refines                     = "
                 << n_global_refinements << std::endl;
  *(this->pcout) << "Number of active cells                = "
                 << this->triangulation.n_global_active_cells() << std::endl;
}

template <int dim>
void Step35<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();
  
  *(this->pcout) << "Number of velocity degrees of freedom = "
                 << (velocity->dof_handler)->n_dofs()
                 << std::endl
                 << "Number of pressure degrees of freedom = "
                 << pressure->dof_handler->n_dofs()
                 << std::endl;
}

template <int dim>
void Step35<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->boundary_conditions.set_dirichlet_bcs(1);
  velocity->boundary_conditions.set_dirichlet_bcs(2,
    std::shared_ptr<Function<dim>> 
    (new EquationData::Step35::VelocityInflowBoundaryCondition<dim>()));
  velocity->boundary_conditions.set_dirichlet_bcs(4);
  velocity->boundary_conditions.set_tangential_flux_bcs(3);
  
  pressure->boundary_conditions.set_dirichlet_bcs(3);

  velocity->apply_boundary_conditions();

  pressure->apply_boundary_conditions();

}

template <int dim>
void Step35<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity, 
                               velocity_initial_condition, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_condition, 
                               time_stepping);
  //navier_stokes.initialize();
  velocity->solution = velocity->old_solution;
  pressure->solution = pressure->old_solution;
  output();
}

template <int dim>
void Step35<dim>::postprocessing(const bool flag_point_evaluation)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  static Point<dim> evaluation_point(2.0, 3.0);
  if (flag_point_evaluation)
  {
    point_evaluation(evaluation_point);
  }
}

template <int dim>
void Step35<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(*velocity->dof_handler,
                           velocity->solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(*pressure->dof_handler, 
                           pressure->solution, 
                           "Pressure");
  data_out.build_patches(velocity->fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record("./",
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

template <int dim>
void Step35<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void Step35<dim>::run()
{
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage
    postprocessing((time_stepping.get_step_number() % 
                    this->prm.terminal_output_interval == 0) ||
                   (time_stepping.get_current_time() == 
                   time_stepping.get_end_time()));

    if (time_stepping.get_step_number() % 
        this->prm.adaptive_meshing_interval == 0)
      this->adaptive_mesh_refinement();

    if ((time_stepping.get_step_number() % 
          this->prm.graphical_output_interval == 0) ||
        (time_stepping.get_current_time() == 
                   time_stepping.get_end_time()))
      output();
  }
}

template <int dim>
void Step35<dim>::point_evaluation(const Point<dim> &point) const
{
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,Point<dim>>
  cell_point =
  GridTools::find_active_cell_around_point(StaticMappingQ1<dim, dim>::mapping,
                                           *velocity->dof_handler,
                                           point);
  if (cell_point.first->is_locally_owned())
  {
    Vector<double> point_value_velocity(dim);
    VectorTools::point_value(*velocity->dof_handler,
                            velocity->solution,
                            point,
                            point_value_velocity);

    const double point_value_pressure
    = VectorTools::point_value(*pressure->dof_handler,
                              pressure->solution,
                              point);
    std::cout << time_stepping
              << " Velocity = (" << std::showpos << std::scientific
              << point_value_velocity[0]
              << ", " << std::showpos << std::scientific
              << point_value_velocity[1]
              << ") Pressure = " << std::showpos << std::scientific
              << point_value_pressure  << std::endl;
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

      deallog.depth_console(parameter_set.verbose ? 2 : 0);

      Step35<2> simulation(parameter_set);
      simulation.run();
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
            << std::endl;
  return 0;
}
