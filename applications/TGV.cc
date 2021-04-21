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

  TGV(RunTimeParameters::HydrodynamicProblemParameters  &parameters,
      const ConvergenceTest::ConvergenceTestParameters  &convergence_parameters);

  void run();

private:

  const ConvergenceTest::ConvergenceTestType  test_type;

  const unsigned int n_spatial_cycles;
  const unsigned int n_temporal_cycles;
  const double       step_size_reduction_factor;

  const types::boundary_id  left_bndry_id;
  const types::boundary_id  right_bndry_id;
  const types::boundary_id  bottom_bndry_id;
  const types::boundary_id  top_bndry_id;

  std::map<typename VectorTools::NormType, double> velocity_error_map;
  std::map<typename VectorTools::NormType, double> pressure_error_map;

  unsigned int n_additional_refinements;

  virtual void make_grid() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

};

template <int dim>
TGV<dim>::TGV
(RunTimeParameters::HydrodynamicProblemParameters &parameters,
 const ConvergenceTest::ConvergenceTestParameters     &convergence_parameters)
:
HydrodynamicProblem<dim>(parameters),
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

  this->velocity->apply_boundary_conditions(/* print_summary */ false);
  this->pressure->apply_boundary_conditions(/* print_summary */ false);
}

template <int dim>
void TGV<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  using namespace EquationData::TGV;
  VelocityExactSolution<dim> velocity_function(this->parameters.Re);
  PressureExactSolution<dim> pressure_function(this->parameters.Re);

  const double step_size = this->time_stepping.get_next_step_size();
  Assert(step_size > 0.0, ExcLowerRangeType<double>(step_size, 0.0));

  this->initialize_from_function(velocity_function,
                                 pressure_function,
                                 step_size);
}


template <int dim>
void TGV<dim>::save_postprocessing_results()
{
  const double current_time = this->time_stepping.get_current_time();

  const Triangulation<dim> &tria{this->triangulation};
  Vector<double>  cellwise_error(tria.n_active_cells());


  auto compute_error
  = [&tria, &cellwise_error, this]
     (const Quadrature<dim>          &quadrature,
      const Entities::EntityBase<dim>&entity,
      const Function<dim>            &exact_solution,
      const VectorTools::NormType     norm_type)
  ->
  double
  {
    VectorTools::integrate_difference(*this->mapping,
                                      *entity.dof_handler,
                                      entity.solution,
                                      exact_solution,
                                      cellwise_error,
                                      quadrature,
                                      norm_type);
    return (VectorTools::compute_global_error(tria,
                                              cellwise_error,
                                              norm_type));
  };

  typename VectorTools::NormType norm_type;

  using namespace EquationData::TGV;
  {
    const VelocityExactSolution<dim> velocity_function(this->parameters.Re,
                                                       current_time);

    const unsigned int fe_degree{this->velocity->fe_degree};
    const QGauss<dim> quadrature_formula(fe_degree + 2);

    double error = compute_error(quadrature_formula,
                                 *this->velocity,
                                 velocity_function,
                                 VectorTools::NormType::L2_norm);
    velocity_error_map[VectorTools::NormType::L2_norm] = error;

    error = compute_error(quadrature_formula,
                          *this->velocity,
                          velocity_function,
                          VectorTools::NormType::H1_norm);
    velocity_error_map[VectorTools::NormType::H1_norm] = error;

    const QTrapez<1>     trapezoidal_rule;
    const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                   fe_degree);
    error = compute_error(linfty_quadrature_formula,
                          *this->velocity,
                          velocity_function,
                          VectorTools::NormType::Linfty_norm);
    velocity_error_map[VectorTools::NormType::Linfty_norm] = error;
  }
  {
    const PressureExactSolution<dim> pressure_function(this->parameters.Re,
                                                       current_time);

    const unsigned int fe_degree{this->pressure->fe_degree};
    const QGauss<dim> quadrature_formula(fe_degree + 2);

    double error = compute_error(quadrature_formula,
                                 *this->pressure,
                                 pressure_function,
                                 VectorTools::NormType::L2_norm);
    pressure_error_map[VectorTools::NormType::L2_norm] = error;


    error = compute_error(quadrature_formula,
                          *this->pressure,
                          pressure_function,
                          VectorTools::NormType::H1_norm);
    pressure_error_map[VectorTools::NormType::H1_norm] = error;

    const QTrapez<1>     trapezoidal_rule;
    const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                   fe_degree);
    error = compute_error(linfty_quadrature_formula,
                          *this->pressure,
                          pressure_function,
                          VectorTools::NormType::Linfty_norm);
    pressure_error_map[norm_type] = error;
  }
}

template <int dim>
void TGV<dim>::run()
{
  using TestType = ConvergenceTest::ConvergenceTestType;

  switch (test_type)
  {
    case TestType::spatial:
    {
      ConvergenceTest::ConvergenceTestData velocity_convergence_table(TestType::spatial);
      ConvergenceTest::ConvergenceTestData pressure_convergence_table(TestType::spatial);

      for (unsigned int cycle=0; cycle < n_spatial_cycles; ++cycle, ++n_additional_refinements)
      {
        {
          unsigned int n_refinements = n_additional_refinements;
          n_refinements += this->parameters.spatial_discretization_parameters.n_initial_global_refinements;

          *this->pcout << "On spatial cycle "
                       << Utilities::to_string(cycle)
                       << " with a refinement level of "
                       << Utilities::int_to_string(n_refinements)
                       << std::endl;
        }

        HydrodynamicProblem<dim>::run();

        velocity_convergence_table.update_table(*this->velocity->dof_handler,
                                                velocity_error_map);
        pressure_convergence_table.update_table(*this->pressure->dof_handler,
                                                pressure_error_map);
        velocity_error_map.clear();
        pressure_error_map.clear();

        this->clear();
      }

      // print tabular output
      *this-> pcout << "Velocity convergence table" << std::endl;
      velocity_convergence_table.print_data(*this->pcout);
      *this->pcout << std::endl;

      *this-> pcout << "Pressure convergence table" << std::endl;
      pressure_convergence_table.print_data(*this->pcout);
      *this->pcout << std::endl;

      // write tabular output to a file
      velocity_convergence_table.save("velocity_spatial_convergence.txt");
      pressure_convergence_table.save("pressure_spatial_convergence.txt");

      break;
    }
    case TestType::temporal:
    {
      ConvergenceTest::ConvergenceTestData velocity_convergence_table(TestType::temporal);
      ConvergenceTest::ConvergenceTestData pressure_convergence_table(TestType::temporal);

      for (unsigned int cycle=0; cycle < n_temporal_cycles; ++cycle)
      {
        const double time_step
        = this->prm.time_discretization_parameters.initial_time_step *
          std::pow(step_size_reduction_factor, cycle);

        const double final_time
        = this->prm.time_discretization_parameters.final_time;

        this->prm.time_discretization_parameters.n_maximum_steps
        = static_cast<unsigned int>(std::ceil(final_time / time_step));;


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

        velocity_convergence_table.update_table(*this->velocity->dof_handler,
                                                time_step,
                                                velocity_error_map);
        pressure_convergence_table.update_table(*this->pressure->dof_handler,
                                                time_step,
                                                pressure_error_map);
        velocity_error_map.clear();
        pressure_error_map.clear();

        this->clear();
      }

      // print tabular output
      *this-> pcout << "Velocity convergence table" << std::endl;
      velocity_convergence_table.print_data(*this->pcout);
      *this->pcout << std::endl;

      *this-> pcout << "Pressure convergence table" << std::endl;
      pressure_convergence_table.print_data(*this->pcout);
      *this->pcout << std::endl;

      // write tabular output to a file
      velocity_convergence_table.save("velocity_temporal_convergence.txt");
      pressure_convergence_table.save("pressure_temporal_convergence.txt");

      break;
    }
    case TestType::spatio_temporal:
    {
      for (unsigned int spatial_cycle=0; spatial_cycle < n_spatial_cycles;
           ++spatial_cycle, ++n_additional_refinements)
      {
        ConvergenceTest::ConvergenceTestData velocity_convergence_table(TestType::spatio_temporal);
        ConvergenceTest::ConvergenceTestData pressure_convergence_table(TestType::spatio_temporal);

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

            *this->pcout << "On temporal cycle "
                         << Utilities::to_string(temporal_cycle)
                         << " with a refinement level of "
                         << Utilities::int_to_string(n_refinements)
                         << std::endl;
          }

          HydrodynamicProblem<dim>::run();

          velocity_convergence_table.update_table(*this->velocity->dof_handler,
                                                  time_step,
                                                  velocity_error_map);
          pressure_convergence_table.update_table(*this->pressure->dof_handler,
                                                  time_step,
                                                  pressure_error_map);
          velocity_error_map.clear();
          pressure_error_map.clear();

          this->clear();

          // print tabular output
          *this-> pcout << "Velocity convergence table" << std::endl;
          velocity_convergence_table.print_data(*this->pcout);
          *this->pcout << std::endl;

          *this-> pcout << "Pressure convergence table" << std::endl;
          pressure_convergence_table.print_data(*this->pcout);
          *this->pcout << std::endl;

          // write tabular output to a file
          std::stringstream sstream;
          sstream << "velocity_convergence_on_level"
                  << Utilities::int_to_string(spatial_cycle, 2)
                  << ".txt";
          velocity_convergence_table.save(sstream.str());

          sstream.clear();
          sstream << "pressure_convergence_on_level"
                  << Utilities::int_to_string(spatial_cycle, 2)
                  << ".txt";
          pressure_convergence_table.save("pressure_convergence.txt");

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

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      RunTimeParameters::HydrodynamicProblemParameters problem_parameters("TGV.prm");
      ConvergenceTest::ConvergenceTestParameters convergence_parameters("TGVConvergence.prm");

      TGV<2> tgv(problem_parameters, convergence_parameters);

      tgv.run();

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
