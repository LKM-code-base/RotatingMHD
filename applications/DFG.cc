#include <rotatingMHD/benchmark_data.h>
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
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include <string>

namespace RMHD
{
  using namespace dealii;

template <int dim>
class DFG : public Problem<dim>
{
public:
  DFG(const RunTimeParameters::ParameterSet &parameters);
  void run(const bool         flag_verbose_output           = false,
           const unsigned int terminal_output_periodicity   = 10,
           const unsigned int graphical_output_periodicity  = 10);
private:
  ConditionalOStream                          pcout;
  parallel::distributed::Triangulation<dim>   triangulation;  
  std::vector<types::boundary_id>             boundary_ids;
  Entities::VectorEntity<dim>                 velocity;
  Entities::ScalarEntity<dim>                 pressure;
  TimeDiscretization::VSIMEXMethod            time_stepping;
  TimeDiscretization::VSIMEXCoefficients      VSIMEX;
  NavierStokesProjection<dim>                 navier_stokes;
  BenchmarkData::DFG<dim>                     dfg_benchmark;

  EquationData::DFG::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::DFG::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;
  EquationData::DFG::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;

  void make_grid();
  void setup_dofs();
  void setup_constraints();
  void initialize();
  void postprocessing(const bool flag_point_evaluation);
  void output();
  void update_solution_vectors();
};

template <int dim>
DFG<dim>::DFG(const RunTimeParameters::ParameterSet &parameters)
  : Problem<dim>(),
    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    triangulation(MPI_COMM_WORLD,
                  typename Triangulation<dim>::MeshSmoothing(
                  Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
    velocity(parameters.p_fe_degree + 1, triangulation),
    pressure(parameters.p_fe_degree, triangulation),
    time_stepping((TimeDiscretization::VSIMEXScheme) (int) parameters.vsimex_scheme,
                  parameters.t_0, parameters.T, parameters.dt,
                  parameters.timestep_lower_bound,
                  parameters.timestep_upper_bound),
    VSIMEX(time_stepping.get_order()),
    navier_stokes(parameters, velocity, pressure, VSIMEX, time_stepping),
    dfg_benchmark(),
    inflow_boundary_condition(parameters.t_0),
    velocity_initial_conditions(parameters.t_0),
    pressure_initial_conditions(parameters.t_0)
{
  make_grid();
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup();
  initialize();
}

template <int dim>
void DFG<dim>::
make_grid()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);

  {
    std::string   filename = "dfg.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  boundary_ids = triangulation.get_boundary_ids();

  const PolarManifold<dim> inner_boundary;
  triangulation.set_all_manifold_ids_on_boundary(2, 1);
  triangulation.set_manifold(1, inner_boundary);

  pcout     << "Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void DFG<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  
  pcout     << "Number of velocity degrees of freedom = " 
            << velocity.dof_handler.n_dofs()
            << std::endl
            << "Number of pressure degrees of freedom = " 
            << pressure.dof_handler.n_dofs()
            << std::endl;
}

template <int dim>
void DFG<dim>::setup_constraints()
{
  velocity.constraints.clear();
  velocity.constraints.reinit(velocity.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(velocity.dof_handler,
                                          velocity.constraints);
  for (const auto &boundary_id : boundary_ids)
    switch (boundary_id)
    {
      case 0:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    inflow_boundary_condition,
                                    velocity.constraints);
        break;
      case 1:
        break;
      case 2:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    Functions::ZeroFunction<dim>(dim),
                                    velocity.constraints);
        break;
      case 3:
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
                                      1,
                                      Functions::ZeroFunction<dim>(),
                                      pressure.constraints);
  pressure.constraints.close();
}

template <int dim>
void DFG<dim>::initialize()
{
  this->set_initial_conditions(velocity, 
                               velocity_initial_conditions, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_conditions, 
                               time_stepping);
  
  navier_stokes.initialize();
}

template <int dim>
void DFG<dim>::postprocessing(const bool flag_point_evaluation)
{
  if (flag_point_evaluation)
  {
    dfg_benchmark.compute_pressure_difference(pressure);
    dfg_benchmark.compute_drag_and_lift_forces_and_coefficients(
                                                            velocity,
                                                            pressure);
    dfg_benchmark.print_step_data(time_stepping);
    dfg_benchmark.update_table(time_stepping);
  }
}

template <int dim>
void DFG<dim>::output()
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
void DFG<dim>::update_solution_vectors()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
  navier_stokes.update_internal_entities();
}

template <int dim>
void DFG<dim>::run(
              const bool          flag_verbose_output,
              const unsigned int  terminal_output_periodicity,
              const unsigned int  graphical_output_periodicity)
{
(void)flag_verbose_output;

for (unsigned int k = 0; k < time_stepping.get_order(); ++k)
  time_stepping.advance_time();

time_stepping.get_coefficients(VSIMEX);

pcout << "Solving until t = 3.5..." << std::endl;
while (time_stepping.get_current_time() <= 3.5)
{
  navier_stokes.solve(time_stepping.get_step_number());

  postprocessing((time_stepping.get_step_number() % 
                  terminal_output_periodicity == 0) ||
                  time_stepping.is_at_end());

  time_stepping.set_proposed_step_size(
                            navier_stokes.compute_next_time_step());
  update_solution_vectors();

  time_stepping.get_coefficients(VSIMEX);
  time_stepping.advance_time();
}

pcout << "Restarting..." << std::endl;
time_stepping.restart();
velocity.old_old_solution = velocity.solution;
navier_stokes.reinit_internal_entities();
navier_stokes.initialize();
velocity.solution = velocity.old_solution;
pressure.solution = pressure.old_solution;
output();

for (unsigned int k = 0; k < time_stepping.get_order(); ++k)
  time_stepping.advance_time();

time_stepping.get_coefficients(VSIMEX);

pcout << "Solving until t = 40..." << std::endl;

while (time_stepping.get_current_time() <= time_stepping.get_end_time())
{    
  navier_stokes.solve(time_stepping.get_step_number());

  postprocessing((time_stepping.get_step_number() % 
                  terminal_output_periodicity == 0) ||
                  time_stepping.is_at_end());

  if ((time_stepping.get_step_number() % 
        graphical_output_periodicity == 0) ||
      time_stepping.is_at_end())
    output();

  time_stepping.set_proposed_step_size(
                            navier_stokes.compute_next_time_step());
  update_solution_vectors();
  
  if (time_stepping.is_at_end())
    break;
  time_stepping.get_coefficients(VSIMEX);
  time_stepping.advance_time();
}
dfg_benchmark.write_table_to_file("dfg_benchmark.tex");
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

      RunTimeParameters::ParameterSet parameter_set;
      parameter_set.read_data_from_file("DFG.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      DFG<2> simulation(parameter_set);
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