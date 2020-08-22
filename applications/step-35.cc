#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
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
class Step35
{
public:
  Step35(const RunTimeParameters::ParameterSet &parameters);
  void run(const bool         flag_verbose_output           = false,
           const unsigned int terminal_output_periodicity   = 10,
           const unsigned int graphical_output_periodicity  = 10);
private:
  ConditionalOStream                          pcout;
  parallel::distributed::Triangulation<dim>   triangulation;  
  std::vector<types::boundary_id>             boundary_ids;
  Entities::Velocity<dim>                     velocity;
  Entities::Pressure<dim>                     pressure;
  TimeDiscretization::VSIMEXMethod            time_stepping;
  TimeDiscretization::VSIMEXCoefficients      VSIMEX;
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
  void set_initial_conditions(
                        Entities::EntityBase<dim>        &entity,
                        Function<dim>                    &function,
                        TimeDiscretization::VSIMEXMethod &time_stepping);
  void postprocessing();
  void output();
  void point_evaluation(const Point<dim>   &point,
                        unsigned int       time_step,
                        DiscreteTime       time) const;
};

template <int dim>
Step35<dim>::Step35(const RunTimeParameters::ParameterSet &parameters)
  : pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    triangulation(MPI_COMM_WORLD,
                  typename Triangulation<dim>::MeshSmoothing(
                  Triangulation<dim>::smoothing_on_refinement |
                  Triangulation<dim>::smoothing_on_coarsening)),
    velocity(parameters.p_fe_degree + 1, triangulation),
    pressure(parameters.p_fe_degree, triangulation),
    time_stepping((TimeDiscretization::VSIMEXScheme) (int) parameters.vsimex_scheme,
                  -parameters.dt, parameters.T, parameters.dt,
                  parameters.timestep_lower_bound,
                  parameters.timestep_upper_bound),
    VSIMEX(time_stepping.get_order()),
    navier_stokes(parameters, velocity, pressure, VSIMEX, time_stepping),
    inflow_boundary_condition(parameters.t_0),
    velocity_initial_conditions(parameters.t_0),
    pressure_initial_conditions(parameters.t_0)
{
  // The VSIMEXMethod class is initialized with t_0 = -dt and then
  // advanced in order to populate a private member of the class, which
  // is needed to calculate the coefficients for the first step
  time_stepping.advance_time();
  time_stepping.get_coefficients(VSIMEX);

  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup();
  initialize();
}

template <int dim>
void Step35<dim>::
make_grid(const unsigned int n_global_refinements)
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);

  {
    std::string   filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  triangulation.refine_global(n_global_refinements);

  boundary_ids = triangulation.get_boundary_ids();

  pcout     << "Number of refines                     = " 
            << n_global_refinements << std::endl;
  pcout     << "Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void Step35<dim>::setup_dofs()
{
  velocity.dof_handler.distribute_dofs(velocity.fe);
  velocity.locally_owned_dofs = velocity.dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(velocity.dof_handler,
                                          velocity.locally_relevant_dofs);

  pressure.dof_handler.distribute_dofs(pressure.fe);
  pressure.locally_owned_dofs = pressure.dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(pressure.dof_handler,
                                          pressure.locally_relevant_dofs);

  pcout     << "Number of velocity degrees of freedom = " 
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
  set_initial_conditions(velocity, 
                         velocity_initial_conditions, 
                         time_stepping);
  set_initial_conditions(pressure,
                         pressure_initial_conditions, 
                         time_stepping);
}

template <int dim>
void Step35<dim>::set_initial_conditions(
                        Entities::EntityBase<dim>         &entity,
                        Function<dim>                     &function,
                        TimeDiscretization::VSIMEXMethod  &time_stepping)
{
  switch (time_stepping.get_order())
  {
    case 1 :
      {
      TrilinosWrappers::MPI::Vector tmp_solution_n(
                                            entity.locally_owned_dofs);
      function.set_time(time_stepping.get_start_time() + 
                        time_stepping.get_next_step_size());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_solution_n);

      entity.solution_n          = tmp_solution_n;
      break;
      }
    case 2 :
      {
      TrilinosWrappers::MPI::Vector tmp_solution_n_minus_1(
                                            entity.locally_owned_dofs);
      TrilinosWrappers::MPI::Vector tmp_solution_n(
                                            entity.locally_owned_dofs);
      function.set_time(time_stepping.get_start_time() + 
                        time_stepping.get_next_step_size());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_solution_n_minus_1);

      function.advance_time(time_stepping.get_next_step_size());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_solution_n);

      entity.solution_n_minus_1  = tmp_solution_n_minus_1;
      entity.solution_n          = tmp_solution_n;
      break;
      }
    default:
      Assert(false, ExcNotImplemented());
  };

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
                           velocity.solution_n, 
                           names, 
                           component_interpretation);
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure.solution_n, 
                           "Pressure");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

template <int dim>
void Step35<dim>::run(
              const bool          flag_verbose_output,
              const unsigned int  terminal_output_periodicity,
              const unsigned int  graphical_output_periodicity)
{
(void)flag_verbose_output;

Point<dim> evaluation_point(2.0, 3.0);

if (time_stepping.get_order() == 2)
  time_stepping.advance_time();

unsigned int step = time_stepping.get_order();

output();

while (!time_stepping.is_at_end())
  {
    time_stepping.get_coefficients(VSIMEX);
    time_stepping.advance_time();
    navier_stokes.solve(step);

    if ((step % graphical_output_periodicity == 0) ||
        time_stepping.is_at_end())
      output();

    if ((step % terminal_output_periodicity == 0) ||
        time_stepping.is_at_end())
      point_evaluation(evaluation_point, step, time_stepping);

    ++step;
  }
}

template <int dim>
void Step35<dim>::
point_evaluation(const Point<dim>   &point,
                 unsigned int       time_step,
                 DiscreteTime       time) const
{
const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> cell_point =
    GridTools::find_active_cell_around_point(
                                    StaticMappingQ1<dim, dim>::mapping,
                                    velocity.dof_handler, 
                                    point);
if (cell_point.first->is_locally_owned())
{
  Vector<double> point_value_velocity(dim);
  VectorTools::point_value(velocity.dof_handler,
                          velocity.solution_n,
                          point,
                          point_value_velocity);

  const double point_value_pressure
  = VectorTools::point_value(pressure.dof_handler,
                            pressure.solution_n,
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

      RunTimeParameters::ParameterSet parameter_set;
      parameter_set.read_data_from_file("parameter_file.prm");

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