/*!
 * @file TGV
 *
 * @brief The source code for solving the Taylor-Green vortex benchmark.
 */
#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/hydrodynamic_problem.h>
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

using namespace dealii;

/*!
 * @class TGV
 *
 * @brief This class solves the Taylor-Green vortex benchmark
 *
 * @todo Add documentation...
 *
 */
template <int dim>
class TGV : public HydrodynamicProblem<dim>
{
public:

  TGV(const RunTimeParameters::HydrodynamicProblemParameters  &parameters,
      const ConvergenceTest::ConvergenceTestParameters        &convergence_parameters);

  void run();

private:

  std::ofstream                 fstream;

  ConvergenceTest::ConvergenceTestData velocity_convergence_table;
  ConvergenceTest::ConvergenceTestData pressure_convergence_table;

  const ConvergenceTest::ConvergenceTestType  test_type;

  const unsigned int n_spatial_cycles;
  const unsigned int n_temporal_cycles;
  const double       step_size_reduction_factor;

  const types::boundary_id  left_bndry_id;
  const types::boundary_id  right_bndry_id;
  const types::boundary_id  bottom_bndry_id;
  const types::boundary_id  top_bndry_id;

  unsigned int n_additional_refinements;

  virtual void make_grid() override;

  virtual void postprocess_solution() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

};

template <int dim>
TGV<dim>::TGV
(const RunTimeParameters::HydrodynamicProblemParameters &parameters,
 const ConvergenceTest::ConvergenceTestParameters     &convergence_parameters)
:
HydrodynamicProblem<dim>(parameters),
fstream("TGV_Log.csv"),
test_type(convergence_parameters.test_type),
n_spatial_cycles(convergence_parameters.n_spatial_cycles),
n_temporal_cycles(convergence_parameters.n_temporal_cycles),
step_size_reduction_factor(convergence_parameters.step_size_reduction_factor),
left_bndry_id(0),
right_bndry_id(1),
bottom_bndry_id(2),
top_bndry_id(3),
n_additional_refinements(0)
{}

template <>
void TGV<2>::make_grid()
{
  constexpr int dim = 2;

  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  using cell_iterator = parallel::distributed::Triangulation<dim>::cell_iterator;

  std::vector<GridTools::PeriodicFacePair<cell_iterator>> periodicity_vector;

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

  unsigned int n_refinements
    = this->parameters.spatial_discretization_parameters.n_initial_global_refinements
    + n_additional_refinements;

  this->triangulation.refine_global(n_refinements);
}

template <int dim>
void TGV<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  this->velocity->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);
  this->velocity->boundary_conditions.set_periodic_bcs(bottom_bndry_id, top_bndry_id, 1);
  this->pressure->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);
  this->pressure->boundary_conditions.set_periodic_bcs(bottom_bndry_id, top_bndry_id, 1);

  this->velocity->apply_boundary_conditions();
  this->pressure->apply_boundary_conditions();
}

template <int dim>
void TGV<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  using namespace EquationData::TGV;

  const double current_time = this->time_stepping.get_current_time();
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Initial conditions are not setup at the start time."));
  const double step_size = this->time_stepping.get_next_step_size();
  Assert(step_size > 0.0, ExcLowerRangeType<double>(step_size, 0.0));

  // compute two ficitous previous times
  const double previous_time = current_time - step_size;
  const double previous_previous_time = previous_time - step_size;

  // initialize all solutions of the velocity
  {
    VelocityExactSolution<dim> velocity_function(this->parameters.Re);

    velocity_function.set_time(previous_previous_time);
    this->project_function(velocity_function,
                           this->velocity,
                           this->velocity->old_old_solution);

    velocity_function.set_time(previous_time);
    this->project_function(velocity_function,
                           this->velocity,
                           this->velocity->old_solution);

    velocity_function.set_time(current_time);
    this->project_function(velocity_function,
                           this->velocity,
                           this->velocity->solution);
  }
  // initialize all solutions of the pressure
  {
    PressureExactSolution<dim> pressure_function(this->parameters.Re);

    pressure_function.set_time(previous_previous_time);
    this->project_function(pressure_function,
                           this->pressure,
                           this->pressure->old_old_solution);

    pressure_function.set_time(previous_time);
    this->project_function(pressure_function,
                           this->pressure,
                           this->pressure->old_solution);

    pressure_function.set_time(current_time);
    this->project_function(pressure_function,
                           this->pressure,
                           this->pressure->solution);
  }
  // initialize the coefficients of the IMEX scheme
  // this->time_stepping.restart(step_size);

  // initialize the coefficients of the pressure correction variable
  // this->navier_stokes.initialize_phi()
}

template <int dim>
void TGV<dim>::postprocess_solution()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

//  if (flag_point_evaluation)
//  {
//    std::cout.precision(1);
//    *this->pcout  << time_stepping
//                  << " Norms = ("
//                  << std::noshowpos << std::scientific
//                  << navier_stokes.get_diffusion_step_rhs_norm()
//                  << ", "
//                  << navier_stokes.get_projection_step_rhs_norm()
//                  << ") CFL = "
//                  << cfl_number
//                  << " ["
//                  << std::setw(5)
//                  << std::fixed
//                  << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
//                  << "%] \r";
//
//    log_file << time_stepping.get_step_number() << ","
//             << time_stepping.get_current_time() << ","
//             << navier_stokes.get_diffusion_step_rhs_norm() << ","
//             << navier_stokes.get_projection_step_rhs_norm() << ","
//             << time_stepping.get_next_step_size() << ","
//             << cfl_number << std::endl;
//  }
}


template <int dim>
void TGV<dim>::run()
{
  using TestType = ConvergenceTest::ConvergenceTestType;

  switch (test_type)
  {
    case TestType::spatial:
    {
      for (unsigned int cycle=0; cycle < n_spatial_cycles; ++cycle, ++n_additional_refinements)
      {
        HydrodynamicProblem<dim>::run();

        this->clear();
      }
      break;
    }
    case TestType::temporal:
    {
      for (unsigned int cycle=0; cycle < n_temporal_cycles; ++cycle)
      {
        const double time_step
        = this->prm.time_discretization_parameters.initial_time_step *
          std::pow(step_size_reduction_factor,
                   cycle);

        this->time_stepping.set_desired_next_step_size(time_step);

        {
          unsigned int n_refinements = n_additional_refinements;
          n_refinements += this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

          *this->pcout << "On temporal cycle "
                       << Utilities::to_string(cycle)
                       << " with a refinement level of "
                       << Utilities::int_to_string(n_refinements)
                       << std::endl;
        }

        HydrodynamicProblem<dim>::run();

        this->clear();
      }
      break;
    }
    case TestType::spatio_temporal:
    {
      for (unsigned int spatial_cycle=0; spatial_cycle < n_spatial_cycles;
           ++spatial_cycle, ++n_additional_refinements)
        for (unsigned int temporal_cycle=0; temporal_cycle < n_temporal_cycles; ++temporal_cycle)
        {
          const double time_step
          = this->prm.time_discretization_parameters.initial_time_step *
            std::pow(step_size_reduction_factor,
                     temporal_cycle);
          this->time_stepping.set_desired_next_step_size(time_step);

          {
            unsigned int n_refinements = n_additional_refinements;
            n_refinements += this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

            *this->pcout << std::setprecision(1)
                         << "Solving until t = "
                         << Utilities::to_string(this->time_stepping.get_end_time(), 6)
                         << " with a refinement level of "
                         << Utilities::int_to_string(n_refinements)
                         << std::endl;
          }

          HydrodynamicProblem<dim>::run();

          this->clear();
        }
      break;
    }
    default:
      Assert(false, ExcMessage("Convergence test type is undefined."));
      break;
  }
//  // Advances the time to t^{k-1}, either t^0 or t^1
//  // This is needed since the boundary integral of the Poisson pre-step
//  // is not defined for this problem.
//  time_stepping.advance_time();
//
//  // Outputs the fields at t_0, i.e. the initial conditions.
//  {
//    velocity->solution = velocity->old_old_solution;
//    pressure->solution = pressure->old_old_solution;
//    velocity_exact_solution->set_time(time_stepping.get_start_time());
//    pressure_exact_solution->set_time(time_stepping.get_start_time());
//    output();
//    velocity->solution = velocity->old_solution;
//    pressure->solution = pressure->old_solution;
//    velocity_exact_solution->set_time(time_stepping.get_start_time() +
//                                     time_stepping.get_next_step_size());
//    pressure_exact_solution->set_time(time_stepping.get_start_time() +
//                                     time_stepping.get_next_step_size());
//    output();
//  }
//
//  while (time_stepping.get_current_time() < time_stepping.get_end_time())
//  {
//    // The VSIMEXMethod instance starts each loop at t^{k-1}
//
//    // Compute CFL number
//    cfl_number = navier_stokes.get_cfl_number();
//
//    // Updates the time step, i.e sets the value of t^{k}
//    time_stepping.set_desired_next_step_size(
//      this->compute_next_time_step(time_stepping, cfl_number));
//
//    // Updates the coefficients to their k-th value
//    time_stepping.update_coefficients();
//
//    // Updates the functions and the constraints to t^{k}
//    velocity_exact_solution->set_time(time_stepping.get_next_time());
//    pressure_exact_solution->set_time(time_stepping.get_next_time());
//
//    velocity->boundary_conditions.set_time(time_stepping.get_next_time());
//    velocity->update_boundary_conditions();
//
//    // Solves the system, i.e. computes the fields at t^{k}
//    navier_stokes.solve();
//
//    // Advances the VSIMEXMethod instance to t^{k}
//    update_entities();
//    time_stepping.advance_time();
//
//    // Snapshot stage, all time calls should be done with get_current_time()
//    postprocessing((time_stepping.get_step_number() %
//                    this->prm.terminal_output_frequency == 0) ||
//                    (time_stepping.get_current_time() ==
//                   time_stepping.get_end_time()));
//
//    if ((time_stepping.get_step_number() %
//          this->prm.graphical_output_frequency == 0) ||
//        (time_stepping.get_current_time() ==
//          time_stepping.get_end_time()))
//      output();
//  }
//
//  Assert(time_stepping.get_current_time() == velocity_exact_solution->get_time(),
//    ExcMessage("Time mismatch between the time stepping class and the velocity function"));
//  Assert(time_stepping.get_current_time() == pressure_exact_solution->get_time(),
//    ExcMessage("Time mismatch between the time stepping class and the pressure function"));
//
//  velocity_convergence_table.update_table(
//    level,
//    time_stepping.get_previous_step_size(),
//    this->prm.convergence_test_parameters.convergence_test_type ==
//      RunTimeParameters::ConvergenceTestType::spatial);
//  pressure_convergence_table.update_table(
//    level, time_stepping.get_previous_step_size(),
//    this->prm.convergence_test_parameters.convergence_test_type ==
//      RunTimeParameters::ConvergenceTestType::spatial);
//
//  velocity->boundary_conditions.clear();
//  pressure->boundary_conditions.clear();
//
//  log_file << "\n";
//
//  *this->pcout << std::endl;
//  *this->pcout << std::endl;
}

//template <int dim>
//void TGV<dim>::run()
//{
//  make_grid(this->prm.convergence_test_parameters.n_global_initial_refinements);
//
//  switch (this->prm.convergence_test_parameters.convergence_test_type)
//  {
//  case ConvergenceTest::ConvergenceTestType::spatial:
//    for (unsigned int level = this->prm.convergence_test_parameters.n_global_initial_refinements;
//         level < (this->prm.convergence_test_parameters.n_global_initial_refinements +
//                  this->prm.convergence_test_parameters.n_spatial_convergence_cycles);
//         ++level)
//    {
//      *this->pcout  << std::setprecision(1)
//                    << "Solving until t = "
//                    << std::fixed << time_stepping.get_end_time()
//                    << " with a refinement level of " << level
//                    << std::endl;
//
//      time_stepping.restart();
//
//      solve(level);
//
//      this->triangulation.refine_global();
//
//      navier_stokes.reset_phi();
//    }
//    break;
//  case ConvergenceTest::::ConvergenceTestType::temporal:
//    for (unsigned int cycle = 0;
//         cycle < this->prm.convergence_test_parameters.n_temporal_convergence_cycles;
//         ++cycle)
//    {
//      double time_step = this->prm.time_discretization_parameters.initial_time_step *
//                         pow(this->prm.convergence_test_parameters.timestep_reduction_factor,
//                             cycle);
//
//      *this->pcout  << std::setprecision(1)
//                    << "Solving until t = "
//                    << std::fixed << time_stepping.get_end_time()
//                    << " with a refinement level of "
//                    << this->prm.convergence_test_parameters.n_global_initial_refinements
//                    << std::endl;
//
//      time_stepping.restart();
//
//      time_stepping.set_desired_next_step_size(time_step);
//
//      solve(this->prm.convergence_test_parameters.n_global_initial_refinements);
//
//      navier_stokes.reset_phi();
//    }
//    break;
//  default:
//    break;
//  }
//
//  *this->pcout << velocity_convergence_table;
//  *this->pcout << pressure_convergence_table;
//
//  std::ostringstream tablefilename;
//  tablefilename << ((this->prm.convergence_test_parameters.convergence_test_type ==
//                      RunTimeParameters::ConvergenceTestType::spatial)
//                     ? "TGV_SpatialTest"
//                     : ("TGV_TemporalTest_Level" + std::to_string(this->prm.convergence_test_parameters.n_global_initial_refinements)))
//                << "_Re"
//                << this->prm.Re;
//
//  velocity_convergence_table.write_text(tablefilename.str() + "_Velocity");
//  pressure_convergence_table.write_text(tablefilename.str() + "_Pressure");
//}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

      RunTimeParameters::HydrodynamicProblemParameters problem_parameters("TGV.prm");
      ConvergenceTest::ConvergenceTestParameters convergence_parameters("TGV.prm");

      TGV<2> simulation(problem_parameters, convergence_parameters);

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
