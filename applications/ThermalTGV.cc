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

  virtual void setup_velocity_field() override;

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

  this->scalar_field->close_boundary_conditions();
  this->scalar_field->apply_boundary_conditions();
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


} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

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
