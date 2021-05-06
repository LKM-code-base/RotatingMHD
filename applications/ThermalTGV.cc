#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/convection_diffusion_problem.h>
#include <rotatingMHD/convergence_test.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>
#include <sstream>

namespace RMHD
{

using namespace dealii;

/*!
 * @class ThermalTGV
 *
 * @todo Add documentation
 *
 */
template <int dim>
class ThermalTGV : public ConvectionDiffusionProblem<dim>
{
public:
  ThermalTGV(ConvectionDiffusionProblemParameters &parameters,
             const ConvergenceTest::ConvergenceTestParameters &convergence_parameters);

private:

  virtual void make_grid() override;

  virtual void postprocess_solution() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

  const ConvergenceTest::ConvergenceTestType  test_type;

  ConvergenceTest::ConvergenceTestData				convergence_data;

  std::map<typename VectorTools::NormType, double>  error_map;

  const types::boundary_id  left_bndry_id;

  const types::boundary_id  right_bndry_id;

  const types::boundary_id  bottom_bndry_id;

  const types::boundary_id  top_bndry_id;

};

template <int dim>
ThermalTGV<dim>::ThermalTGV
(ConvectionDiffusionProblemParameters &parameters,
 const ConvergenceTest::ConvergenceTestParameters &convergence_parameters)
:
ConvectionDiffusionProblem<dim>(parameters),
test_type(convergence_parameters.test_type),
convergence_data(test_type),
left_bndry_id(0),
right_bndry_id(1),
bottom_bndry_id(2),
top_bndry_id(3)
{
  AssertDimension(dim, 2);
}

template <>
void ThermalTGV<2>::make_grid()
{
  constexpr int dim{2};

  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);
  GridTools::collect_periodic_faces(this->triangulation,
                                    bottom_bndry_id,
                                    top_bndry_id,
                                    1,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(this->parameters.spatial_discretization_parameters.n_initial_global_refinements);
}



template <int dim>
void ThermalTGV<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  this->scalar_field->clear_boundary_conditions();

  this->scalar_field->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);
  this->scalar_field->boundary_conditions.set_periodic_bcs(bottom_bndry_id, top_bndry_id, 1);

  temperature->close_boundary_conditions();
  temperature->apply_boundary_conditions();
}

template <int dim>
void ThermalTGV<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  const double current_time{this->time_stepping.get_current_time()};
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Initial conditions are not setup at the start time."));

  using namespace EquationData::ThermalTGV;
  TemperatureExactSolution<dim> temperature_function(current_time);

  const double step_size{this->time_stepping.get_next_step_size()};
  Assert(step_size > 0.0, ExcLowerRangeType<double>(step_size, 0.0));

  this->initialize_from_function(temperature_function,
                                 step_size);
}

template <int dim>
void ThermalTGV<dim>::save_postprocessing_results()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  const double current_time{this->time_stepping.get_current_time()};

  using namespace EquationData::ThermalTGV;
  TemperatureExactSolution<dim> temperature_function(current_time);

  this->scalar_field->compute_error(temperature_function, this->mapping)
}

template <int dim>
void ThermalTGV<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  this->compute_error(error,
                      temperature,
                      *temperature_exact_solution);

  DataOut<dim>        data_out;

  data_out.add_data_vector(*temperature->dof_handler,
                           temperature->solution,
                           "temperature");
  data_out.add_data_vector(*temperature->dof_handler,
                           error,
                           "error");

  data_out.build_patches(temperature->fe_degree);

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
void ThermalTGV<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  temperature->reinit();
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

    temperature->boundary_conditions.set_time(time_stepping.get_next_time());
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

  {
    auto error_map = temperature->compute_error(*temperature_exact_solution, this->mapping);

    convergence_data.update_table(*this->temperature->dof_handler,
                                  time_stepping.get_previous_step_size(),
                                  error_map);
  }
  convergence_table.update_table(
    level,
    time_stepping.get_previous_step_size(),
    this->prm.convergence_test_parameters.test_type ==
    		ConvergenceTest::ConvergenceTestType::spatial);

  log_file << "\n";

  *this->pcout << std::endl << std::endl;
}

template <int dim>
void ThermalTGV<dim>::run()
{
  make_grid(this->prm.spatial_discretization_parameters.n_initial_global_refinements);

  switch (this->prm.convergence_test_parameters.test_type)
  {
  case ConvergenceTest::ConvergenceTestType::spatial:
    for (unsigned int level = this->prm.spatial_discretization_parameters.n_initial_global_refinements;
         level < (this->prm.spatial_discretization_parameters.n_initial_global_refinements +
                  this->prm.convergence_test_parameters.n_spatial_cycles);
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
         cycle < this->prm.convergence_test_parameters.n_temporal_cycles;
         ++cycle)
    {
      double time_step = this->prm.time_discretization_parameters.initial_time_step *
                         pow(this->prm.convergence_test_parameters.step_size_reduction_factor,
                             cycle);

      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of "
                    << this->prm.spatial_discretization_parameters.n_initial_global_refinements
                    << std::endl;

      time_stepping.restart();

      time_stepping.set_desired_next_step_size(time_step);

      solve(this->prm.spatial_discretization_parameters.n_initial_global_refinements);
    }
    break;
  default:
    break;
  }

  *this->pcout << convergence_table;

  std::ostringstream tablefilename;
  if (this->prm.convergence_test_parameters.test_type == ConvergenceTest::ConvergenceTestType::spatial)
    tablefilename << "ThermalTGV_SpatialTest";
  else
  {
    tablefilename << "ThermalTGV_TemporalTest_Level";
    const unsigned int n_initial_global_refinements{this->prm.spatial_discretization_parameters.n_initial_global_refinements};
    tablefilename << n_initial_global_refinements;
  }
  tablefilename << "_Re"
                << this->prm.Pe;

  convergence_table.write_text(tablefilename.str());

  *this->pcout << convergence_data;
  convergence_data.save("ThermalTGV_DumpTest.txt");
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
