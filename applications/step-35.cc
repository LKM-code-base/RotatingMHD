#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_in.h>
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
  TimeDiscretization::VSIMEXCoefficients      VSIMEX;
  TimeDiscretization::VSIMEXMethod            time_stepping;
  NavierStokesProjection<dim>                 navier_stokes;
  
  EquationData::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;
  EquationData::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;

  void make_grid(const unsigned int n_global_refinements);
  void setup_dofs();
  void setup_constraints();
  void initialize();
  void assembly();
  void solve();
  void postprocessing();
  void output();
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
    VSIMEX(2),
    time_stepping(2,
                  {parameters.vsimex_input_gamma, 
                    parameters.vsimex_input_c}, 
                  -parameters.dt, parameters.T, parameters.dt),
    navier_stokes(parameters, velocity, pressure, VSIMEX, time_stepping),
    inflow_boundary_condition(false, parameters.t_0),
    velocity_initial_conditions(parameters.t_0),
    pressure_initial_conditions(false, parameters.t_0)
{
  time_stepping.advance_time();
  time_stepping.update_coefficients();
  time_stepping.get_coefficients(VSIMEX);

  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup();
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
void Step35<dim>::run(
              const bool          flag_verbose_output,
              const unsigned int  terminal_output_periodicity,
              const unsigned int  graphical_output_periodicity)
{
(void)flag_verbose_output;
(void)terminal_output_periodicity;
(void)graphical_output_periodicity;

std::cout << velocity.fe_degree << std::endl;

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
                     parameter_set.graphical_output_interval,
                     parameter_set.terminal_output_interval);
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
