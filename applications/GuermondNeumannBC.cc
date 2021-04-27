/*!
 *@file Guermond
 *@brief The .cc file replicating the numerical test of section
  10.3 of the Guermond paper.
 */
#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/hydrodynamic_problem.h>
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <memory>
#include <sstream>

namespace RMHD
{
  using namespace dealii;

/*!
 * @class GuermondNeumann
 *
 * @todo Add documentation
 *
 */
template <int dim>
class Guermond : public HydrodynamicProblem<dim>
{

public:
  Guermond(RunTimeParameters::HydrodynamicProblemParameters &parameters,
           const ConvergenceTest::ConvergenceTestParameters &convergence_parameters);

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

  const bool  flag_set_exact_pressure_value;

  unsigned int n_additional_refinements;

  EquationData::GuermondNeumannBC::BodyForce<dim> body_force;

  std::map<typename VectorTools::NormType, double> velocity_error_map;
  std::map<typename VectorTools::NormType, double> pressure_error_map;

  virtual void make_grid() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

};

template <int dim>
Guermond<dim>::Guermond
(RunTimeParameters::HydrodynamicProblemParameters &parameters,
 const ConvergenceTest::ConvergenceTestParameters &convergence_parameters)
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
flag_set_exact_pressure_value(true),
n_additional_refinements(0),
body_force(this->parameters.Re, this->time_stepping.get_start_time())
{
  this->navier_stokes.set_body_force(body_force);
}

template <>
void Guermond<2>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  unsigned int n_refinements
  = this->parameters.spatial_discretization_parameters.n_initial_global_refinements
  + n_additional_refinements;

  this->triangulation.refine_global(n_refinements);
}

template <int dim>
void Guermond<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  this->velocity->boundary_conditions.clear();
  this->pressure->boundary_conditions.clear();

  const double current_time = this->time_stepping.get_current_time();
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Boundary conditions are not setup at the start time."));

  using namespace EquationData::GuermondNeumannBC;
  const std::shared_ptr<Function<dim>> velocity_exact_solution
  = std::make_shared<VelocityExactSolution<dim>>(current_time);

  this->velocity->boundary_conditions.set_neumann_bcs(left_bndry_id);

  this->velocity->boundary_conditions.set_dirichlet_bcs(right_bndry_id,
                                                        velocity_exact_solution,
                                                        true);
  this->velocity->boundary_conditions.set_dirichlet_bcs(bottom_bndry_id,
                                                        velocity_exact_solution,
                                                        true);
  this->velocity->boundary_conditions.set_dirichlet_bcs(top_bndry_id,
                                                        velocity_exact_solution,
                                                        true);

  this->velocity->boundary_conditions.close();
  this->pressure->boundary_conditions.close();

  this->velocity->apply_boundary_conditions();
  this->pressure->apply_boundary_conditions();
}

template <int dim>
void Guermond<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  const double current_time = this->time_stepping.get_current_time();
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Boundary conditions are not setup at the start time."));

  using namespace EquationData::GuermondNeumannBC;
  VelocityExactSolution<dim> velocity_function(current_time);
  PressureExactSolution<dim> pressure_function(current_time);

  const double step_size = this->time_stepping.get_next_step_size();
  Assert(step_size > 0.0, ExcLowerRangeType<double>(step_size, 0.0));

  this->initialize_from_function(velocity_function,
                                 pressure_function,
                                 step_size);
}

template <int dim>
void Guermond<dim>::save_postprocessing_results()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

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

  using namespace EquationData::GuermondNeumannBC;

  if (flag_set_exact_pressure_value)
  {
    PressureExactSolution<dim> pressure_function(current_time);

    QGauss<dim> quadrature(this->pressure->fe.degree + 1);

    LinearAlgebra::MPI::Vector  distributed_analytical_pressure;
    distributed_analytical_pressure.reinit(this->pressure->distributed_vector);

    VectorTools::project(*this->mapping,
                         *this->pressure->dof_handler,
                         this->pressure->hanging_nodes,
                         quadrature,
                         pressure_function,
                         distributed_analytical_pressure);

    LinearAlgebra::MPI::Vector  analytical_pressure;
    analytical_pressure.reinit(this->pressure->solution);
    analytical_pressure = distributed_analytical_pressure;

    const LinearAlgebra::MPI::Vector::value_type analytical_mean_value
    = VectorTools::compute_mean_value(*this->mapping,
                                      *this->pressure->dof_handler,
                                      quadrature,
                                      analytical_pressure,
                                      0);

    const LinearAlgebra::MPI::Vector::value_type numerical_mean_value
    = VectorTools::compute_mean_value(*this->mapping,
                                      *this->pressure->dof_handler,
                                      quadrature,
                                      this->pressure->solution,
                                      0);

    LinearAlgebra::MPI::Vector  distributed_numerical_pressure;
    distributed_numerical_pressure.reinit(this->pressure->distributed_vector);
    distributed_numerical_pressure = this->pressure->solution;
    distributed_numerical_pressure.add(analytical_mean_value -
                                       numerical_mean_value);

    this->pressure->solution = distributed_numerical_pressure;
  }

  typename VectorTools::NormType norm_type;
  {
    const VelocityExactSolution<dim> velocity_function(current_time);

    const unsigned int fe_degree{this->velocity->fe_degree};
    const QGauss<dim> quadrature_formula(fe_degree + 2);

    norm_type = VectorTools::NormType::L2_norm;
    double error = compute_error(quadrature_formula,
                                 *this->velocity,
                                 velocity_function,
                                 norm_type);
    velocity_error_map[norm_type] = error;

    norm_type = VectorTools::NormType::H1_norm;
    error = compute_error(quadrature_formula,
                          *this->velocity,
                          velocity_function,
                          norm_type);
    velocity_error_map[norm_type] = error;

    const QTrapez<1>     trapezoidal_rule;
    const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                   fe_degree);

    norm_type = VectorTools::NormType::Linfty_norm;
    error = compute_error(linfty_quadrature_formula,
                          *this->velocity,
                          velocity_function,
                          norm_type);
    velocity_error_map[norm_type] = error;
  }
  {
    const PressureExactSolution<dim> pressure_function(current_time);

    const unsigned int fe_degree{this->pressure->fe_degree};
    const QGauss<dim> quadrature_formula(fe_degree + 2);

    norm_type = VectorTools::NormType::L2_norm;
    double error = compute_error(quadrature_formula,
                                 *this->pressure,
                                 pressure_function,
                                 norm_type);
    pressure_error_map[norm_type] = error;

    norm_type = VectorTools::NormType::H1_norm;
    error = compute_error(quadrature_formula,
                          *this->pressure,
                          pressure_function,
                          norm_type);
    pressure_error_map[norm_type] = error;

    const QTrapez<1>     trapezoidal_rule;
    const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                   fe_degree);
    norm_type = VectorTools::NormType::Linfty_norm;
    error = compute_error(linfty_quadrature_formula,
                          *this->pressure,
                          pressure_function,
                          norm_type);
    pressure_error_map[norm_type] = error;
  }
}


template <int dim>
void Guermond<dim>::run()
{
  if (!std::filesystem::exists(this->prm.graphical_output_directory) &&
      Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    try
    {
      std::filesystem::create_directories(this->prm.graphical_output_directory);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the creation of the output directory: "
                << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
    catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                  << std::endl;
      std::cerr << "Unknown exception in the creation of the output directory!"
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
  }

  std::filesystem::path path{this->prm.graphical_output_directory};

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
      {
        std::filesystem::path filename = path / "velocity_spatial_convergence.txt";
        velocity_convergence_table.save(filename.string());

        filename = path / "pressure_spatial_convergence.txt";
        pressure_convergence_table.save(filename.string());
      }

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

        velocity_convergence_table.update_table(time_step,
                                                velocity_error_map);
        pressure_convergence_table.update_table(time_step,
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
      {
        std::filesystem::path filename = path / "velocity_temporal_convergence.txt";
        velocity_convergence_table.save(filename.string());

        filename = path / "pressure_temporal_convergence.txt";
        pressure_convergence_table.save(filename.string());
      }

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
          {
            std::stringstream sstream;
            sstream << "velocity_convergence_on_level"
                    << Utilities::int_to_string(spatial_cycle, 2)
                    << ".txt";

            std::filesystem::path filename = path / sstream.str();
            velocity_convergence_table.save(filename.string());

            sstream.clear();
            sstream << "pressure_convergence_on_level"
                    << Utilities::int_to_string(spatial_cycle, 2)
                    << ".txt";
            filename = path / sstream.str();
            pressure_convergence_table.save(filename.string());
          }
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

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

      RunTimeParameters::HydrodynamicProblemParameters problem_parameters("GuermondNeumannBC.prm");
      ConvergenceTest::ConvergenceTestParameters       convergence_parameters("GuermondNeumannBCConvergence.prm");

      Guermond<2> guermond(problem_parameters, convergence_parameters);
      guermond.run();

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
