#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/heat_equation.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>


#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


#include <memory>

namespace RMHD
{

using namespace dealii;

template <int dim>
class ThermalTGV : public Problem<dim>
{
public:
  ThermalTGV(const RunTimeParameters::ParameterSet &parameters);

  void run(const bool flag_convergence_test);

  std::ofstream outputFile;

private:
  std::vector<types::boundary_id>             boundary_ids;

  Entities::ScalarEntity<dim>                 temperature;

  Entities::VectorEntity<dim>                 velocity;

  LinearAlgebra::MPI::Vector                  error;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  EquationData::ThermalTGV::TemperatureExactSolution<dim>         
                                              exact_solution;

  EquationData::ThermalTGV::VelocityField<dim> velocity_field;

  std::shared_ptr<Mapping<dim>>               mapping;

  HeatEquation<dim>                           heat_equation;

  ConvergenceAnalysisData<dim>                convergence_table;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_entities();

  void solve(const unsigned int &level);
};

template <int dim>
ThermalTGV<dim>::ThermalTGV(
  const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(parameters),
outputFile("ThermalTGV_Log.csv"),
temperature(parameters.temperature_fe_degree, this->triangulation),
velocity(parameters.p_fe_degree + 1, this->triangulation),
time_stepping(parameters.time_stepping_parameters),
exact_solution(parameters.Re,
               parameters.Pr,
               parameters.time_stepping_parameters.start_time),
velocity_field(parameters.time_stepping_parameters.start_time),
mapping(new MappingQ<dim>(1)),
heat_equation(parameters,
              time_stepping,
              temperature,
              velocity,
              mapping,
              this->pcout,
              this->computing_timer),
convergence_table(temperature, exact_solution, "Temperature")
{
outputFile << "Step" << "," << "Time" << ","
           << "Norm_diffusion" << "," << "Norm_projection"
           << "," << "dt" << "," << "CFL" << std::endl;
}

template <int dim>
void ThermalTGV<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  FullMatrix<double> rotation_matrix(dim);
  rotation_matrix[0][0] = 1.;
  rotation_matrix[1][1] = 1.;    

  Tensor<1,dim> offset_x;
  offset_x[0] = 1.0;
  offset_x[1] = 0.0;

  Tensor<1,dim> offset_y;
  offset_y[0] = 0.0;
  offset_y[1] = 1.0;

  GridTools::collect_periodic_faces(this->triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector,
                                    offset_x,
                                    rotation_matrix);
  GridTools::collect_periodic_faces(this->triangulation,
                                    2,
                                    3,
                                    1,
                                    periodicity_vector,
                                    offset_y,
                                    rotation_matrix);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(n_global_refinements);
  boundary_ids = this->triangulation.get_boundary_ids();
}

template <int dim>
void ThermalTGV<dim>::setup_dofs()
{
  temperature.setup_dofs();
  velocity.setup_dofs();
  *(this->pcout)  << "  Number of active cells                   = " 
                  << this->triangulation.n_global_active_cells() 
                  << std::endl;
  *(this->pcout)  << "  Number of temperature degrees of freedom = " 
                  << temperature.dof_handler.n_dofs()
                  << std::endl;
}

template <int dim>
void ThermalTGV<dim>::setup_constraints()
{
  temperature.boundary_conditions.clear();
  velocity.boundary_conditions.clear();
  
  temperature.boundary_conditions.set_periodic_bcs(0, 1, 0);
  temperature.boundary_conditions.set_periodic_bcs(2, 3, 1);

  velocity.boundary_conditions.set_periodic_bcs(0, 1, 0);
  velocity.boundary_conditions.set_periodic_bcs(2, 3, 1);

  temperature.apply_boundary_conditions();

  velocity.apply_boundary_conditions();
}

template <int dim>
void ThermalTGV<dim>::initialize()
{
  this->set_initial_conditions(temperature, 
                               exact_solution, 
                               time_stepping);
  this->set_initial_conditions(velocity,
                               velocity_field,
                               time_stepping);
}

template <int dim>
void ThermalTGV<dim>::postprocessing()
{
  std::cout.precision(1);
  *(this->pcout)  << time_stepping
                  << " Norm = "
                  << std::noshowpos << std::scientific
                  << heat_equation.get_rhs_norm()
                  << " Progress ["
                  << std::setw(5) 
                  << std::fixed
                  << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                  << "%] \r";
}

template <int dim>
void ThermalTGV<dim>::output()
{
  this->compute_error(error,
                      temperature,
                      exact_solution);

  DataOut<dim>        data_out;

  data_out.add_data_vector(temperature.dof_handler, 
                           temperature.solution, 
                           "temperature");
  data_out.add_data_vector(temperature.dof_handler, 
                           error, 
                           "error");
  data_out.build_patches(temperature.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record("./",
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

template <int dim>
void ThermalTGV<dim>::update_entities()
{
  temperature.update_solution_vectors();
}

template <int dim>
void ThermalTGV<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  temperature.reinit();
  velocity.reinit();
  error.reinit(temperature.solution);
  initialize();

  // Advances the time to t^{k-1}, either t^0 or t^1
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();
  
  // Outputs the fields at t_0, i.e. the initial conditions.
  { 
    temperature.solution = temperature.old_old_solution;
    exact_solution.set_time(time_stepping.get_start_time());
    output();
    temperature.solution = temperature.old_solution;
    exact_solution.set_time(time_stepping.get_start_time() + 
                            time_stepping.get_next_step_size());
    output();   
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}
    /*time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));*/
    
    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();
    
    // Updates the functions and the constraints to t^{k}
    exact_solution.set_time(time_stepping.get_next_time());

    temperature.boundary_conditions.set_time(time_stepping.get_next_time());
    temperature.update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    heat_equation.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();


    // Snapshot stage, all time calls should be done with get_next_time()
    postprocessing();

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_interval == 0) ||
        (time_stepping.get_current_time() == 
          time_stepping.get_end_time()))
      output();
  }
  
  Assert(time_stepping.get_current_time() == exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the temperature function"));

  convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), this->prm.flag_spatial_convergence_test);

  temperature.boundary_conditions.clear();

  outputFile << "\n";

  *(this->pcout) << std::endl;
  *(this->pcout) << std::endl;
}

template <int dim>
void ThermalTGV<dim>::run(const bool flag_convergence_test)
{
  make_grid(this->prm.initial_refinement_level);
  if (flag_convergence_test)
  { 
    *(this->pcout) << "Performing a spatial convergence test"
                   << std::endl << std::endl;
    for (unsigned int level = this->prm.initial_refinement_level; 
          level <= this->prm.final_refinement_level; ++level)
    {
      std::cout.precision(1);
      *(this->pcout)  << "Solving until t = " 
                      << std::fixed << time_stepping.get_end_time()
                      << " with a refinement level of " << level 
                      << std::endl;
      time_stepping.restart();
      solve(level);
      this->triangulation.refine_global();
      heat_equation.set_linear_algebra_to_reset();
    }
  }
  else
  {
    *(this->pcout) << "Performing a temporal convergence test"
                   << std::endl << std::endl;
    for (unsigned int cycle = 0; 
         cycle < this->prm.temporal_convergence_cycles; ++cycle)
    {
      double time_step = this->prm.time_stepping_parameters.initial_time_step *
                         pow(this->prm.time_step_scaling_factor, cycle);
      std::cout.precision(1);
      *(this->pcout)  << "Solving until t = " 
                      << std::fixed << time_stepping.get_end_time()
                      << " with a refinement level of " 
                      << this->prm.initial_refinement_level << std::endl;
      time_stepping.restart();
      time_stepping.set_desired_next_step_size(time_step);
      solve(this->prm.initial_refinement_level);
      heat_equation.set_linear_algebra_to_reset();
    }
  }

  std::string tablefilename = (this->prm.flag_spatial_convergence_test) ?
                              "ThermalTGVSpatialTest" : 
                              "ThermalTGVTemporalTest_Level" + 
                              std::to_string(this->prm.initial_refinement_level);
  tablefilename += "_Pr" + std::to_string((int)this->prm.Pr);

  convergence_table.print_table_to_terminal();
  convergence_table.print_table_to_file(tablefilename);
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

      RunTimeParameters::ParameterSet parameter_set("ThermalTGV.prm");

      deallog.depth_console(parameter_set.verbose ? 2 : 0);

      ThermalTGV<2> simulation(parameter_set);
      
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
            << std::endl;
  return 0;
}