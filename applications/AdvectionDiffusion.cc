#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/convergence_test.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
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



template <int dim>
class AdvectionDiffusion : public Problem <dim>
{
public:
  AdvectionDiffusion(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  const RunTimeParameters::ProblemParameters    &parameters;

  std::shared_ptr<Entities::ScalarEntity<dim>>  scalar_field;

  std::shared_ptr<Function<dim>>                exact_solution;

  LinearAlgebra::MPI::Vector                    error_field;

  std::shared_ptr<TensorFunction<1,dim>>        velocity_field;

  EquationData::AdvectionDiffusion::SourceTerm<dim> source_term;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  HeatEquation<dim>                             advection_diffusion;

  ConvergenceAnalysisData<dim>                  convergence_table;

  double                                        cfl_number;

  std::ofstream                                 log_file;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void solve(const unsigned int &level);

  void postprocessing();

  void output();

  void update_entities();
};



template <int dim>
AdvectionDiffusion<dim>::AdvectionDiffusion(
  const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
scalar_field(std::make_shared<Entities::ScalarEntity<dim>>(
  parameters.fe_degree_temperature,
  this->triangulation,
  "Scalar field")),
exact_solution(std::make_shared<EquationData::AdvectionDiffusion::TemperatureExactSolution<dim>>(
  parameters.time_discretization_parameters.start_time)),
velocity_field(std::make_shared<EquationData::AdvectionDiffusion::VelocityExactSolution<dim>>(
  parameters.time_discretization_parameters.start_time)),
source_term(parameters.Pe, parameters.time_discretization_parameters.start_time),
time_stepping(parameters.time_discretization_parameters),
advection_diffusion(parameters.heat_equation_parameters,
                    time_stepping,
                    scalar_field,
                    velocity_field,
                    this->mapping,
                    this->pcout,
                    this->computing_timer),
convergence_table(scalar_field, *exact_solution),
cfl_number(std::numeric_limits<double>::min()),
log_file("AdvectionDiffusion_Log.csv")
{}



template <int dim>
void AdvectionDiffusion<dim>::make_grid(
  const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            false);

  this->triangulation.refine_global(n_global_refinements);
}



template <int dim>
void AdvectionDiffusion<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  scalar_field->setup_dofs();
  *this->pcout  << "  Number of active cells                   = "
                << this->triangulation.n_global_active_cells()
                << std::endl
                << "  Number of temperature degrees of freedom = "
                << (scalar_field->dof_handler)->n_dofs()
                << std::endl;
}



template <int dim>
void AdvectionDiffusion<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  scalar_field->clear_boundary_conditions();

  exact_solution->set_time(time_stepping.get_start_time());

  scalar_field->boundary_conditions.set_dirichlet_bcs(0);

  scalar_field->close_boundary_conditions();
  scalar_field->apply_boundary_conditions();
}



template <int dim>
void AdvectionDiffusion<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(scalar_field,
                               *exact_solution,
                               time_stepping);
}



template <int dim>
void AdvectionDiffusion<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  std::cout.precision(1);
  *this->pcout  << time_stepping
                << " Norm = "
                << std::noshowpos << std::scientific
                << advection_diffusion.get_rhs_norm()
                << " Progress ["
                << std::setw(5)
                << std::fixed
                << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                << "%] \r";

  log_file << time_stepping.get_step_number()     << ","
           << time_stepping.get_current_time()    << ","
           << advection_diffusion.get_rhs_norm()        << ","
           << time_stepping.get_next_step_size()  << std::endl;
}



template <int dim>
void AdvectionDiffusion<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  this->compute_error(error_field,
                      scalar_field,
                      *exact_solution);

  DataOut<dim>  data_out;

  data_out.add_data_vector(*scalar_field->dof_handler,
                           scalar_field->solution,
                           "Scalar");
  data_out.add_data_vector(*scalar_field->dof_handler,
                           error_field,
                           "Error");

  data_out.build_patches(scalar_field->fe_degree);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "Solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}



template <int dim>
void AdvectionDiffusion<dim>::solve(const unsigned int &level)
{
  advection_diffusion.set_source_term(source_term);
  setup_dofs();
  setup_constraints();
  scalar_field->reinit();
  error_field.reinit(scalar_field->solution);
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    scalar_field->solution = scalar_field->old_solution;
    exact_solution->set_time(time_stepping.get_start_time());
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    exact_solution->set_time(time_stepping.get_next_time());

    scalar_field->boundary_conditions.set_time(time_stepping.get_next_time());
    scalar_field->update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    advection_diffusion.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    scalar_field->update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_next_time()

    if ((time_stepping.get_step_number() %
          this->prm.terminal_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      postprocessing();

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  Assert(time_stepping.get_current_time() == exact_solution->get_time(),
    ExcMessage("Time mismatch between the time stepping class and the temperature function"));

  convergence_table.update_table(
    level,
    time_stepping.get_previous_step_size(),
    parameters.convergence_test_parameters.test_type ==
    		ConvergenceTest::ConvergenceTestType::spatial);

  log_file << "\n";

  *this->pcout << std::endl << std::endl;
}



template <int dim>
void AdvectionDiffusion<dim>::run()
{
  make_grid(parameters.spatial_discretization_parameters.n_initial_global_refinements);

  switch (parameters.convergence_test_parameters.test_type)
  {
  case ConvergenceTest::ConvergenceTestType::spatial:
    for (unsigned int level = parameters.spatial_discretization_parameters.n_initial_global_refinements;
         level < (parameters.spatial_discretization_parameters.n_initial_global_refinements +
                  parameters.convergence_test_parameters.n_spatial_cycles);
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
    }
    break;
  case ConvergenceTest::ConvergenceTestType::temporal:
    for (unsigned int cycle = 0;
         cycle < parameters.convergence_test_parameters.n_temporal_cycles;
         ++cycle)
    {
      double time_step = parameters.time_discretization_parameters.initial_time_step *
                         pow(parameters.convergence_test_parameters.step_size_reduction_factor,
                             cycle);

      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of "
                    << parameters.spatial_discretization_parameters.n_initial_global_refinements
                    << std::endl;

      time_stepping.restart();

      time_stepping.set_desired_next_step_size(time_step);

      solve(parameters.spatial_discretization_parameters.n_initial_global_refinements);
    }
    break;
  default:
    break;
  }

  *this->pcout << convergence_table;

  std::ostringstream tablefilename;
  tablefilename << ((parameters.convergence_test_parameters.test_type ==
  									 ConvergenceTest::ConvergenceTestType::spatial)
                     ? "AdvectionDiffusion_SpatialTest"
                     : ("AdvectionDiffusion_TemporalTest_Level" + std::to_string(parameters.spatial_discretization_parameters.n_initial_global_refinements)))
                << "_Pe"
                << parameters.Pe;

  convergence_table.write_text(tablefilename.str());
}



} // namespace RMHD



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace RMHD;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                        argv,
                                                        1);

    RunTimeParameters::ProblemParameters parameter_set(
      "AdvectionDiffusion.prm",
      true);

    AdvectionDiffusion<2> simulation(parameter_set);

    simulation.run();
  }
  catch(std::exception& exc)
  {
    std::cerr << std::endl << std::endl
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
    std::cerr << std::endl << std::endl
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