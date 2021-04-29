/*!
 *@file Guermond
 *@brief The .cc file replicating the numerical test of section
  10.3 of the Guermond paper.
 */
#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
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

/*!
 * @class Guermond
 *
 * @todo Add documentation
 *
 */
template <int dim>
class Guermond : public Problem<dim>
{
public:
  Guermond(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  std::ofstream                                 log_file;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  LinearAlgebra::MPI::Vector                    velocity_error;

  LinearAlgebra::MPI::Vector                    pressure_error;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  std::shared_ptr<EquationData::GuermondNeumannBC::VelocityExactSolution<dim>>
                                                velocity_exact_solution;

  std::shared_ptr<EquationData::GuermondNeumannBC::PressureExactSolution<dim>>
                                                pressure_exact_solution;

  EquationData::GuermondNeumannBC::BodyForce<dim>
                                                body_force;

  ConvergenceAnalysisData<dim>                  velocity_convergence_table;

  ConvergenceAnalysisData<dim>                  pressure_convergence_table;

  double                                        cfl_number;

  const bool                                    flag_set_exact_pressure_constant;

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
Guermond<dim>::Guermond(const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
log_file("GuermondNeumannBC_Log.csv"),
velocity(std::make_shared<Entities::VectorEntity<dim>>(parameters.fe_degree_velocity,
                                                       this->triangulation,
                                                       "Velocity")),
pressure(std::make_shared<Entities::ScalarEntity<dim>>(parameters.fe_degree_pressure,
                                                       this->triangulation,
                                                       "Pressure")),
time_stepping(parameters.time_discretization_parameters),
navier_stokes(parameters.navier_stokes_parameters,
              time_stepping,
              velocity,
              pressure,
              this->mapping,
              this->pcout,
              this->computing_timer),
velocity_exact_solution(
  std::make_shared<EquationData::GuermondNeumannBC::VelocityExactSolution<dim>>(
    parameters.time_discretization_parameters.start_time)),
pressure_exact_solution(
  std::make_shared<EquationData::GuermondNeumannBC::PressureExactSolution<dim>>(
    parameters.time_discretization_parameters.start_time)),
body_force(parameters.Re, parameters.time_discretization_parameters.start_time),
velocity_convergence_table(velocity, *velocity_exact_solution),
pressure_convergence_table(pressure, *pressure_exact_solution),
flag_set_exact_pressure_constant(false)
{
  navier_stokes.set_body_force(body_force);

  *this->pcout << parameters << std::endl << std::endl;

  log_file << "Step" << ","
           << "Time" << ","
           << "Norm_diffusion" << ","
           << "Norm_projection" << ","
           << "dt" << ","
           << "CFL" << std::endl;
}

template <int dim>
void Guermond<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  this->triangulation.refine_global(n_global_refinements);
}

template <int dim>
void Guermond<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();

  *this->pcout  << "  Number of active cells                = "
                << this->triangulation.n_global_active_cells() << std::endl;
  *this->pcout  << "  Number of velocity degrees of freedom = "
                << (velocity->dof_handler)->n_dofs()
                << std::endl
                << "  Number of pressure degrees of freedom = "
                << (pressure->dof_handler)->n_dofs()
                << std::endl
               << "  Number of total degrees of freedom    = "
               << (pressure->dof_handler->n_dofs() +
                   velocity->dof_handler->n_dofs())
               << std::endl;
}

template <int dim>
void Guermond<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->clear_boundary_conditions();
  pressure->clear_boundary_conditions();

  velocity_exact_solution->set_time(time_stepping.get_start_time());

  // left boundary
  velocity->boundary_conditions.set_neumann_bcs(0);
  // right boundary
  velocity->boundary_conditions.set_dirichlet_bcs(
    1,
    velocity_exact_solution,
    true);
  pressure->boundary_conditions.set_dirichlet_bcs(
  	1,
		pressure_exact_solution,
		true);
  // bottom boundary
  velocity->boundary_conditions.set_dirichlet_bcs(
    2,
    velocity_exact_solution,
    true);
  pressure->boundary_conditions.set_dirichlet_bcs(
  	2,
		pressure_exact_solution,
		true);
  // top boundary
  velocity->boundary_conditions.set_dirichlet_bcs(
    3,
    velocity_exact_solution,
    true);
  pressure->boundary_conditions.set_dirichlet_bcs(
  	3,
		pressure_exact_solution,
		true);

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void Guermond<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity,
                               *velocity_exact_solution,
                               time_stepping,
                               true);
  this->set_initial_conditions(pressure,
                               *pressure_exact_solution,
                               time_stepping,
                               true);
}

template <int dim>
void Guermond<dim>::postprocessing(const bool flag_point_evaluation)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  if (flag_set_exact_pressure_constant)
  {
    LinearAlgebra::MPI::Vector  analytical_pressure(pressure->solution);
    {
      #ifdef USE_PETSC_LA
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure->locally_owned_dofs,
                                this->mpi_communicator);
      #else
        LinearAlgebra::MPI::Vector
        tmp_analytical_pressure(pressure->locally_owned_dofs);
      #endif
      VectorTools::project(*(pressure->dof_handler),
                          pressure->constraints,
                          QGauss<dim>(pressure->fe_degree + 2),
                          *pressure_exact_solution,
                          tmp_analytical_pressure);

      analytical_pressure = tmp_analytical_pressure;
    }
    {
      LinearAlgebra::MPI::Vector distributed_analytical_pressure;
      LinearAlgebra::MPI::Vector distributed_numerical_pressure;
      #ifdef USE_PETSC_LA
        distributed_analytical_pressure.reinit(pressure->locally_owned_dofs,
                                        this->mpi_communicator);
      #else
        distributed_analytical_pressure.reinit(pressure->locally_owned_dofs,
                                               pressure->locally_relevant_dofs,
                                               this->mpi_communicator,
                                               true);
      #endif
      distributed_numerical_pressure.reinit(distributed_analytical_pressure);

      distributed_analytical_pressure = analytical_pressure;
      distributed_numerical_pressure  = pressure->solution;

      distributed_numerical_pressure.add(
        distributed_analytical_pressure.mean_value() -
        distributed_numerical_pressure.mean_value());

      pressure->solution = distributed_numerical_pressure;
    }
  }

  if (flag_point_evaluation)
  {
    std::cout.precision(1);
    *this->pcout << time_stepping
                 << " Norms = ("
                 << std::noshowpos << std::scientific
                 << navier_stokes.get_diffusion_step_rhs_norm()
                 << ", "
                 << navier_stokes.get_projection_step_rhs_norm()
                 << ") CFL = "
                 << cfl_number
                 << " ["
                 << std::setw(5)
                 << std::fixed
                 << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                 << "%] \r";

    log_file << time_stepping.get_step_number() << ","
             << time_stepping.get_current_time() << ","
             << navier_stokes.get_diffusion_step_rhs_norm() << ","
             << navier_stokes.get_projection_step_rhs_norm() << ","
             << time_stepping.get_next_step_size() << ","
             << cfl_number << std::endl;
  }
}

template <int dim>
void Guermond<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  this->compute_error(velocity_error,
                       velocity,
                       *velocity_exact_solution);
  this->compute_error(pressure_error,
                       pressure,
                       *pressure_exact_solution);

  std::vector<std::string> names(dim, "velocity");
  std::vector<std::string> error_name(dim, "velocity_error");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim>        data_out;

  data_out.add_data_vector(*(velocity->dof_handler),
                           velocity->solution,
                           names,
                           component_interpretation);

  data_out.add_data_vector(*(velocity->dof_handler),
                           velocity_error,
                           error_name,
                           component_interpretation);

  data_out.add_data_vector(*(pressure->dof_handler),
                           pressure->solution,
                           "pressure");

  data_out.add_data_vector(*(pressure->dof_handler),
                           pressure_error,
                           "pressure_error");

  data_out.build_patches(velocity->fe_degree);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}

template <int dim>
void Guermond<dim>::update_entities()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void Guermond<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  velocity->reinit();
  pressure->reinit();
  velocity_error.reinit(velocity->solution);
  pressure_error.reinit(pressure->solution);
  initialize();

  time_stepping.advance_time();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    velocity->solution = velocity->old_old_solution;
    pressure->solution = pressure->old_old_solution;
    velocity_exact_solution->set_time(time_stepping.get_start_time());
    pressure_exact_solution->set_time(time_stepping.get_start_time());
    output();
    velocity->solution = velocity->old_solution;
    pressure->solution = pressure->old_solution;
    velocity_exact_solution->set_time(time_stepping.get_start_time() +
                                     time_stepping.get_next_step_size());
    pressure_exact_solution->set_time(time_stepping.get_start_time() +
                                     time_stepping.get_next_step_size());
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(time_stepping, cfl_number));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    velocity_exact_solution->set_time(time_stepping.get_next_time());
    pressure_exact_solution->set_time(time_stepping.get_next_time());

    velocity->boundary_conditions.set_time(time_stepping.get_next_time());
    velocity->update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();

    // Snapshot stage
    postprocessing((time_stepping.get_step_number() %
                    this->prm.terminal_output_frequency == 0) ||
                    (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  Assert(time_stepping.get_current_time() == velocity_exact_solution->get_time(),
    ExcMessage("Time mismatch between the time stepping class and the velocity function"));
  Assert(time_stepping.get_current_time() == pressure_exact_solution->get_time(),
    ExcMessage("Time mismatch between the time stepping class and the pressure function"));

  velocity_convergence_table.update_table(
    level,
    time_stepping.get_previous_step_size(),
    this->prm.convergence_test_parameters.convergence_test_type ==
      RunTimeParameters::ConvergenceTestType::spatial);
  pressure_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(),
    this->prm.convergence_test_parameters.convergence_test_type ==
      RunTimeParameters::ConvergenceTestType::spatial);

  log_file << "\n";

  *this->pcout << std::endl << std::endl;
}

template <int dim>
void Guermond<dim>::run()
{
  make_grid(this->prm.convergence_test_parameters.n_global_initial_refinements);

  switch (this->prm.convergence_test_parameters.convergence_test_type)
  {
  case RunTimeParameters::ConvergenceTestType::spatial:
    for (unsigned int level = this->prm.convergence_test_parameters.n_global_initial_refinements;
         level < (this->prm.convergence_test_parameters.n_global_initial_refinements +
                  this->prm.convergence_test_parameters.n_spatial_convergence_cycles);
         ++level)
    {
      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of " << level
                    << std::endl;

      time_stepping.restart();

      solve(level);

      this->triangulation.refine_global();

      navier_stokes.reset_phi();
    }
    break;
  case RunTimeParameters::ConvergenceTestType::temporal:
    for (unsigned int cycle = 0;
         cycle < this->prm.convergence_test_parameters.n_temporal_convergence_cycles;
         ++cycle)
    {
      double time_step = this->prm.time_discretization_parameters.initial_time_step *
                         pow(this->prm.convergence_test_parameters.timestep_reduction_factor,
                             cycle);

      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of "
                    << this->prm.convergence_test_parameters.n_global_initial_refinements
                    << std::endl;

      time_stepping.restart();

      time_stepping.set_desired_next_step_size(time_step);

      solve(this->prm.convergence_test_parameters.n_global_initial_refinements);

      navier_stokes.reset_phi();
    }
    break;
  default:
    break;
  }

  *this->pcout << velocity_convergence_table;
  *this->pcout << pressure_convergence_table;

  std::ostringstream tablefilename;
  tablefilename << ((this->prm.convergence_test_parameters.convergence_test_type ==
                      RunTimeParameters::ConvergenceTestType::spatial)
                     ? "GuermondNeumannBC_SpatialTest"
                     : ("GuermondNeumannBC_TemporalTest_Level" + std::to_string(this->prm.convergence_test_parameters.n_global_initial_refinements)))
                << "_Re"
                << this->prm.Re;

  velocity_convergence_table.write_text(tablefilename.str() + "_Velocity");
  pressure_convergence_table.write_text(tablefilename.str() + "_Pressure");
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

      RunTimeParameters::ProblemParameters parameter_set("GuermondNeumannBC.prm",
                                                         true);

      Guermond<2> simulation(parameter_set);

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
  return 0;
}
