#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
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
 * @class ThermalTGV
 * @todo Add documentation
 */
template <int dim>
class ThermalTGV : public Problem<dim>
{
public:
  ThermalTGV(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:

  const RunTimeParameters::ProblemParameters   &parameters;

  std::ofstream                                 log_file;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  temperature;

  LinearAlgebra::MPI::Vector                    error;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  std::shared_ptr<Function<dim>>								temperature_exact_solution;

  std::shared_ptr<TensorFunction<1,dim>> 				velocity_exact_solution;

  HeatEquation<dim>                             heat_equation;

  ConvergenceTable                              convergence_table;

  double                                        cfl_number;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void process_solution(const unsigned int cycle);

  void output();

  void update_entities();

  void solve(const unsigned int level);
};

template <int dim>
ThermalTGV<dim>::ThermalTGV(
  const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
log_file("ThermalTGV_Log.csv"),
temperature(std::make_shared<Entities::FE_ScalarField<dim>>(parameters.fe_degree_temperature,
                                                          this->triangulation,
                                                          "Temperature")),
time_stepping(parameters.time_discretization_parameters),
temperature_exact_solution(
  std::make_shared<EquationData::ThermalTGV::TemperatureExactSolution<dim>>(
    parameters.Pe,
    parameters.time_discretization_parameters.start_time)),
velocity_exact_solution(
  std::make_shared<EquationData::ThermalTGV::VelocityExactSolution<dim>>(
    parameters.time_discretization_parameters.start_time)),
heat_equation(parameters.heat_equation_parameters,
              time_stepping,
              temperature,
              velocity_exact_solution,
              this->mapping,
              this->pcout,
              this->computing_timer)
{
  *this->pcout << parameters << std::endl << std::endl;

  log_file << "Step" << ","
           << "Time" << ","
           << "Norm" << ","
           << "dt"   << std::endl;
}

template <int dim>
void ThermalTGV<dim>::
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

  GridTools::collect_periodic_faces(this->triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector);
  GridTools::collect_periodic_faces(this->triangulation,
                                    2,
                                    3,
                                    1,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(n_global_refinements);
}

template <int dim>
void ThermalTGV<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  temperature->setup_dofs();
  *this->pcout  << "  Number of active cells                   = "
                << this->triangulation.n_global_active_cells()
                << std::endl
                << "  Number of temperature degrees of freedom = "
                << temperature->n_dofs()
                << std::endl;
}

template <int dim>
void ThermalTGV<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  temperature->clear_boundary_conditions();

  temperature->setup_boundary_conditions();

  temperature_exact_solution->set_time(time_stepping.get_start_time());

  temperature->set_periodic_boundary_condition(0, 1, 0);
  temperature->set_periodic_boundary_condition(2, 3, 1);

  temperature->close_boundary_conditions();
  temperature->apply_boundary_conditions();
}

template <int dim>
void ThermalTGV<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(temperature,
                               *temperature_exact_solution,
                               time_stepping);
}

template <int dim>
void ThermalTGV<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  std::cout.precision(1);
  *this->pcout  << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                << " Norm = "
                << std::noshowpos << std::scientific
                << heat_equation.get_rhs_norm()
                << " Progress ["
                << std::setw(5)
                << std::fixed
                << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                << "%] \r";

  log_file << time_stepping.get_step_number()     << ","
           << time_stepping.get_current_time()    << ","
           << heat_equation.get_rhs_norm()        << ","
           << time_stepping.get_next_step_size()  << std::endl;
}


template <int dim>
void ThermalTGV<dim>::process_solution(const unsigned int cycle)
{
  const double current_time{time_stepping.get_current_time()};

  using ExactSolution = EquationData::ThermalTGV::TemperatureExactSolution<dim>;

  Vector<float> difference_per_cell(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(temperature->get_dof_handler(),
                                    temperature->solution,
                                    ExactSolution(parameters.Pe, current_time),
                                    difference_per_cell,
                                    QGauss<dim>(temperature->fe_degree() + 1),
                                    VectorTools::L2_norm);
  const double L2_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  VectorTools::integrate_difference(temperature->get_dof_handler(),
                                    temperature->solution,
                                    ExactSolution(parameters.Pe, current_time),
                                    difference_per_cell,
                                    QGauss<dim>(temperature->fe_degree() + 1),
                                    VectorTools::H1_seminorm);
  const double H1_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);
  const QTrapez<1>     q_trapez;
  const QIterated<dim> q_iterated(q_trapez, temperature->fe_degree() * 2 + 1);
  VectorTools::integrate_difference(temperature->get_dof_handler(),
                                    temperature->solution,
                                    ExactSolution(parameters.Pe, current_time),
                                    difference_per_cell,
                                    q_iterated,
                                    VectorTools::Linfty_norm);
  const double Linfty_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::Linfty_norm);

  const unsigned int n_active_cells = this->triangulation.n_global_active_cells();
  const unsigned int n_dofs         = temperature->n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("time_step", time_stepping.get_previous_step_size());
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

}


template <int dim>
void ThermalTGV<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  DataOut<dim>        data_out;

  data_out.add_data_vector(temperature->get_dof_handler(),
                           temperature->solution,
                           "temperature");
  data_out.add_data_vector(temperature->get_dof_handler(),
                           error,
                           "error");

  data_out.build_patches(temperature->fe_degree());

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}

template <int dim>
void ThermalTGV<dim>::update_entities()
{
  temperature->update_solution_vectors();
}

template <int dim>
void ThermalTGV<dim>::solve(const unsigned int /* level */)
{
  setup_dofs();
  setup_constraints();
  temperature->setup_vectors();
  error.reinit(temperature->solution);
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    temperature->solution = temperature->old_solution;
    temperature_exact_solution->set_time(time_stepping.get_start_time());
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    temperature_exact_solution->set_time(time_stepping.get_next_time());
    temperature->update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    heat_equation.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
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

  Assert(time_stepping.get_current_time() == temperature_exact_solution->get_time(),
         ExcMessage("Time mismatch between the time stepping class and the temperature function"));



  log_file << "\n";

  *this->pcout << std::endl << std::endl;
}

template <int dim>
void ThermalTGV<dim>::run()
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

      process_solution(level);

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

      process_solution(cycle);
    }
    break;
  default:
    break;
  }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("Linfty", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);
  convergence_table.omit_column_from_convergence_rate_evaluation("cycle");
  convergence_table.omit_column_from_convergence_rate_evaluation("cells");
  convergence_table.omit_column_from_convergence_rate_evaluation("dofs");
  convergence_table.omit_column_from_convergence_rate_evaluation("time_step");
  convergence_table.evaluate_all_convergence_rates("time_step", ConvergenceTable::RateMode::reduction_rate_log2);

  *this->pcout << std::endl;
  if (this->pcout->is_active())
    convergence_table.write_text(this->pcout->get_stream());
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

      RunTimeParameters::ProblemParameters parameter_set("ThermalTGV.prm",
                                                         true);

      ThermalTGV<2> simulation(parameter_set);

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
