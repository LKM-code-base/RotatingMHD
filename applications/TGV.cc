/*!
 *@file TGV
 *@brief The .cc file solving the TGV benchmark.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

namespace RMHD
{
  using namespace dealii;

/*!
 * @class TGV
 * @brief This class solves the Taylor-Green vortex benchmark
 * @todo Add documentation
 */  
template <int dim>
class TGV : public Problem<dim>
{
public:

  TGV(const RunTimeParameters::ParameterSet &parameters);

  void run(const bool flag_convergence_test);

  std::ofstream outputFile;

private:

  std::vector<types::boundary_id>             boundary_ids;

  Entities::VectorEntity<dim>                 velocity;

  Entities::ScalarEntity<dim>                 pressure;

  LinearAlgebra::MPI::Vector                  velocity_error;

  LinearAlgebra::MPI::Vector                  pressure_error;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;

  EquationData::TGV::VelocityExactSolution<dim>         
                                      velocity_exact_solution;

  EquationData::TGV::PressureExactSolution<dim>         
                                      pressure_exact_solution;

  ConvergenceAnalysisData<dim>                velocity_convergence_table;

  ConvergenceAnalysisData<dim>                pressure_convergence_table;

  const bool                                  flag_set_exact_pressure_constant;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_entities();

  void solve(const unsigned int &level);

  void compute_error_vector(LinearAlgebra::MPI::Vector  &error_vector,
                            Entities::EntityBase<dim>   &entity,
                            Function<dim>               &exact_solution);
};

template <int dim>
TGV<dim>::TGV(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(parameters),
outputFile("TGV_Log.csv"),
velocity(parameters.p_fe_degree + 1, this->triangulation),
pressure(parameters.p_fe_degree, this->triangulation),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters,
              velocity,
              pressure,
              time_stepping,
              this->pcout,
              this->computing_timer),
velocity_exact_solution(parameters.Re, parameters.time_stepping_parameters.start_time),
pressure_exact_solution(parameters.Re, parameters.time_stepping_parameters.start_time),
velocity_convergence_table(velocity, velocity_exact_solution, "Velocity"),
pressure_convergence_table(pressure, pressure_exact_solution, "Pressure"),
flag_set_exact_pressure_constant(true)
{
outputFile << "Step" << "," << "Time" << ","
           << "Norm_diffusion" << "," << "Norm_projection"
           << "," << "dt" << "," << "CFL" << std::endl;
}

template <int dim>
void TGV<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

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
void TGV<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity.setup_dofs();
  pressure.setup_dofs();
  *(this->pcout)  << "  Number of active cells                = " 
                  << this->triangulation.n_global_active_cells() << std::endl;
  *(this->pcout)  << "  Number of velocity degrees of freedom = " 
                  << velocity.dof_handler->n_dofs()
                  << std::endl
                  << "  Number of pressure degrees of freedom = " 
                  << pressure.dof_handler->n_dofs()
                  << std::endl;
}

template <int dim>
void TGV<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity.boundary_conditions.clear();
  pressure.boundary_conditions.clear();
  
  velocity.boundary_conditions.set_periodic_bcs(0, 1, 0);
  velocity.boundary_conditions.set_periodic_bcs(2, 3, 1);
  pressure.boundary_conditions.set_periodic_bcs(0, 1, 0);
  pressure.boundary_conditions.set_periodic_bcs(2, 3, 1);
  
  velocity.apply_boundary_conditions();
  pressure.apply_boundary_conditions();
}

template <int dim>
void TGV<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity, 
                               velocity_exact_solution, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_exact_solution, 
                               time_stepping);
  // The diffusion prestep does not produce physical results.
  // Could this be due to the boundary conditions? Since there is no
  // body force in this problem the problem reduces to a boundary integral
  // whhose integrand is projected in the normal to the boundary direction.
  //navier_stokes.initialize();
}

template <int dim>
void TGV<dim>::postprocessing(const bool flag_point_evaluation)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  if (flag_set_exact_pressure_constant)
  {
    LinearAlgebra::MPI::Vector  analytical_pressure(pressure.solution);
    {
      #ifdef USE_PETSC_LA
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure.locally_owned_dofs,
                                this->mpi_communicator);
      #else
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure.locally_owned_dofs);
      #endif
      VectorTools::project(*(pressure.dof_handler),
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
                                               this->mpi_communicator);
      #else
        distributed_analytical_pressure.reinit(pressure.locally_owned_dofs,
                                               pressure.locally_relevant_dofs,
                                               this->mpi_communicator,
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
    *(this->pcout)  << time_stepping
                    << " Norms = ("
                    << std::noshowpos << std::scientific
                    << navier_stokes.get_diffusion_step_rhs_norm()
                    << ", "
                    << navier_stokes.get_projection_step_rhs_norm()
                    << ") CFL = "
                    << navier_stokes.get_cfl_number()
                    << " ["
                    << std::setw(5) 
                    << std::fixed
                    << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                    << "%] \r";
    outputFile << time_stepping.get_step_number() << ","
               << time_stepping.get_current_time() << ","
               << navier_stokes.get_diffusion_step_rhs_norm() << ","
               << navier_stokes.get_projection_step_rhs_norm() << ","
               << time_stepping.get_next_step_size() << ","
               << navier_stokes.get_cfl_number() << std::endl;
  }
}

template <int dim>
void TGV<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

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
  data_out.add_data_vector(*(velocity.dof_handler),
                           velocity.solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(*(velocity.dof_handler),
                           velocity_error,
                           error_name, 
                           component_interpretation);
  data_out.add_data_vector(*(pressure.dof_handler), 
                           pressure.solution, 
                           "pressure");
  data_out.add_data_vector(*(pressure.dof_handler), 
                           pressure_error, 
                           "pressure_error");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record("./",
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

template <int dim>
void TGV<dim>::update_entities()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
}

template <int dim>
void TGV<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  velocity_error.reinit(velocity.solution);
  pressure_error.reinit(pressure.solution);
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
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));
    
    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();
    
    // Updates the functions and the constraints to t^{k}
    velocity_exact_solution.set_time(time_stepping.get_next_time());
    pressure_exact_solution.set_time(time_stepping.get_next_time());

    velocity.boundary_conditions.set_time(time_stepping.get_next_time());
    velocity.update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_current_time()
    postprocessing((time_stepping.get_step_number() %
                    this->prm.terminal_output_interval == 0) ||
                    (time_stepping.get_current_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_interval == 0) ||
        (time_stepping.get_current_time() == 
          time_stepping.get_end_time()))
      output();
  }
  
  Assert(time_stepping.get_current_time() == velocity_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the velocity function"));
  Assert(time_stepping.get_current_time() == pressure_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the pressure function"));
  
  velocity_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), this->prm.flag_spatial_convergence_test);
  pressure_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), this->prm.flag_spatial_convergence_test);
  
  velocity.boundary_conditions.clear();
  pressure.boundary_conditions.clear();

  outputFile << "\n";

  *(this->pcout) << std::endl;
  *(this->pcout) << std::endl;
}

template <int dim>
void TGV<dim>::run(const bool flag_convergence_test)
{
  make_grid(this->prm.initial_refinement_level);
  if (flag_convergence_test)
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
      navier_stokes.reset_phi();
    }
  else
  {
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
      navier_stokes.reset_phi();
    }
  }
  
  std::string tablefilename = (this->prm.flag_spatial_convergence_test) ?
                              "TGVSpatialTest" : 
                              "TGVTemporalTest_Level" + 
                              std::to_string(this->prm.initial_refinement_level);
  tablefilename += "_Re" + std::to_string((int)this->prm.Re);

  velocity_convergence_table.print_table_to_terminal();
  velocity_convergence_table.print_table_to_file(tablefilename + "_Velocity");
  pressure_convergence_table.print_table_to_terminal();
  pressure_convergence_table.print_table_to_file(tablefilename + "_Pressure");
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

      RunTimeParameters::ParameterSet parameter_set("TGV.prm");

      deallog.depth_console(parameter_set.verbose ? 2 : 0);

      TGV<2> simulation(parameter_set);
      
      simulation.run(parameter_set.flag_spatial_convergence_test);
  
      //simulation.outputFile.close();

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
