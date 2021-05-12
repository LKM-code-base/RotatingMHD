#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/convection_diffusion_problem.h>
#include <rotatingMHD/convergence_test.h>

#include <deal.II/base/function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <filesystem>
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

  void run();

private:

  virtual void make_grid() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

  virtual void setup_velocity_field() override;

  const ConvergenceTest::ConvergenceTestType  test_type;

  ConvergenceTest::ConvergenceTestData				convergence_data;

  const unsigned int n_spatial_cycles;

  const unsigned int n_temporal_cycles;

  const double step_size_reduction_factor;

  const types::boundary_id  left_bndry_id;

  const types::boundary_id  right_bndry_id;

  const types::boundary_id  bottom_bndry_id;

  const types::boundary_id  top_bndry_id;

  std::map<typename VectorTools::NormType, double>  error_map;

  unsigned int n_additional_refinements;

};



template <int dim>
ThermalTGV<dim>::ThermalTGV
(ConvectionDiffusionProblemParameters &parameters,
 const ConvergenceTest::ConvergenceTestParameters &convergence_parameters)
:
ConvectionDiffusionProblem<dim>(parameters),
test_type(convergence_parameters.test_type),
convergence_data(test_type),
n_spatial_cycles(convergence_parameters.n_spatial_cycles),
n_temporal_cycles(convergence_parameters.n_temporal_cycles),
step_size_reduction_factor(convergence_parameters.step_size_reduction_factor),
left_bndry_id(0),
right_bndry_id(1),
bottom_bndry_id(2),
top_bndry_id(3),
n_additional_refinements(0)
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

  unsigned int n_refinements{this->parameters.spatial_discretization_parameters.
                             n_initial_global_refinements};
  n_refinements += n_additional_refinements;
  this->triangulation.refine_global(n_refinements);
}



template <int dim>
void ThermalTGV<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  this->scalar_field->clear_boundary_conditions();

  this->scalar_field->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);
  this->scalar_field->boundary_conditions.set_periodic_bcs(bottom_bndry_id, top_bndry_id, 1);

  this->scalar_field->close_boundary_conditions(/* print summary ? */ false);
  this->scalar_field->apply_boundary_conditions();
}



template <int dim>
void ThermalTGV<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  const double current_time{this->time_stepping.get_current_time()};
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Initial conditions are not setup at the start time."));

  EquationData::ThermalTGV::TemperatureExactSolution<dim>
  temperature_function(this->parameters.peclet_number, current_time);

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

  EquationData::ThermalTGV::TemperatureExactSolution<dim>
  temperature_function(this->parameters.peclet_number, current_time);

  error_map = this->scalar_field->compute_error(temperature_function,
                                                *this->mapping);
}



template <int dim>
void ThermalTGV<dim>::setup_velocity_field()
{
  std::shared_ptr<TensorFunction<1,dim>> velocity_function_ptr
  = std::make_shared<EquationData::ThermalTGV::VelocityExactSolution<dim>>();

  this->solver.set_velocity(velocity_function_ptr);
}



template <int dim>
void ThermalTGV<dim>::run()
{

  std::filesystem::path path{this->parameters.graphical_output_directory};

  using TestType = ConvergenceTest::ConvergenceTestType;

  switch (test_type)
  {
    case TestType::spatial:
    {
      for (unsigned int cycle=0; cycle < n_spatial_cycles; ++cycle, ++n_additional_refinements)
      {
        {
          unsigned int n_refinements = cycle;
          n_refinements += this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

          *this->pcout << "On spatial cycle "
                       << Utilities::to_string(cycle)
                       << " with a refinement level of "
                       << Utilities::int_to_string(n_refinements)
                       << std::endl;
        }

        ConvectionDiffusionProblem<dim>::run();

        convergence_data.update_table(*this->scalar_field->dof_handler,
                                       error_map);
        error_map.clear();

        this->clear();
      }

      // print tabular output
      *this->pcout << "Convergence table" << std::endl
                   << convergence_data
                   << std::endl;

      // write tabular output to a file
      {
        std::filesystem::path filename = path / "spatial_convergence.txt";
        convergence_data.save(filename.string());
      }

      break;
    }
    case TestType::temporal:
    {
      for (unsigned int cycle=0; cycle < n_temporal_cycles; ++cycle)
      {
          const double time_step
          = this->parameters.time_discretization_parameters.initial_time_step *
          std::pow(step_size_reduction_factor, cycle);

          const double final_time
          = this->parameters.time_discretization_parameters.final_time;

          this->parameters.time_discretization_parameters.n_maximum_steps
          = static_cast<unsigned int>(std::ceil(final_time / time_step));;

          this->time_stepping.set_desired_next_step_size(time_step);

          {
            unsigned int n_refinements
            = this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

            *this->pcout << "On temporal cycle "
                         << Utilities::to_string(cycle)
                         << " with a refinement level of "
                         << Utilities::int_to_string(n_refinements)
            << std::endl;
          }

          ConvectionDiffusionProblem<dim>::run();

          convergence_data.update_table(time_step,
                                         error_map);
          error_map.clear();

          this->clear();
      }

      // print tabular output
      *this->pcout << "Convergence table" << std::endl
                   << convergence_data
                   << std::endl;

      // write tabular output to a file
      {
        std::filesystem::path filename = path / "temporal_convergence.txt";
        convergence_data.save(filename.string());
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
            = this->parameters.time_discretization_parameters.initial_time_step *
            std::pow(step_size_reduction_factor,
                     temporal_cycle);
            this->time_stepping.set_desired_next_step_size(time_step);

            {
              unsigned int n_refinements = spatial_cycle;
              n_refinements += this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

              *this->pcout << "On temporal cycle "
                           << Utilities::to_string(temporal_cycle)
                           << " with a refinement level of "
                           << Utilities::int_to_string(n_refinements)
                           << std::endl;
            }

            ConvectionDiffusionProblem<dim>::run();

            convergence_data.update_table(*this->scalar_field->dof_handler,
                                          time_step,
                                          error_map);
            error_map.clear();

            this->clear();

            // print tabular output
            *this->pcout << "Convergence table" << std::endl
                         << convergence_data
                         << std::endl;

            // write tabular output to a file
            {
              std::stringstream sstream;
              sstream << "convergence_on_level"
                      << Utilities::int_to_string(spatial_cycle, 2)
                      << ".txt";

              std::filesystem::path filename = path / sstream.str();
              convergence_data.save(filename.string());

            }
        }
      break;
    }
    default:
      Assert(false, ExcMessage("Convergence test type is undefined."));
      break;
  }
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

      ConvectionDiffusionProblemParameters        parameters("ThermalTGV.prm");
      ConvergenceTest::ConvergenceTestParameters  convergence_parameters("ThermalTGVConvergence.prm");

      ThermalTGV<2> simulation(parameters, convergence_parameters);

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
