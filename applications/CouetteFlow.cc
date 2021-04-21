/*!
 *
 * @file Couette flow.
 *
 * @brief The source file solving the Couette flow problem.
 *
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

namespace RMHD
{
  using namespace dealii;

/*!
 * @class Couette
 *
 * @brief This class solves the Couette flow problem.
 *
 * @details The Couette flow was chosen as a test problem to verify the
 * correct implementation of the Neumann boundary conditions in the
 * @ref NavierStokesProjection solver. The problem considers stationary
 * laminar flow of fluid between two infinitely long horizontal plates,
 * where the lower plate is fixed and the upper one is moved.
 *
 * The problem is governed by the equation
 * \f[
 * \ddsqr{u_\mathrm{x}}{y} = 0, \quad \forall \bs{x} \in \Omega\,,
 * \f]
 *
 * which can be easily derived from the momentum equation by
 * considering an isotropic and homogeneous fluid, an unidirectional and
 * stationary flow in the \f$ x\f$-direction, neglecting body forces and
 * assuming a vanishing horizontal pressure gradient. An unique solution is
 * obtained by imposing a no-slip boundary condition at the lower
 * plate and a traction vector \f$ \bs{t} = t_0 \bs{e}_\textrm{x} \f$ at the
 * upper plate.
 *
 * This yields
 * \f[
 * \bs{u} = t_0 \Reynolds \dfrac{y}{H} \bs{e}_\textrm{x}\,,
 * \f]
 *
 * where \f$ \Reynolds \f$ is the Reynolds number and \f$ H \f$ the height of
 * the channel. The stationary solution is obtained at around
 * \f$ t ~ H^2 \Reynolds \f$.
 *
 * @note Periodic boundary conditions are implemented in order to
 * simulate an infinitely long channel.
 *
 */
template <int dim>
class CouetteFlow : public HydrodynamicProblem<dim>
{
public:

  CouetteFlow(RunTimeParameters::HydrodynamicProblemParameters &parameters,
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

  std::map<typename VectorTools::NormType, double> velocity_error_map;
  std::map<typename VectorTools::NormType, double> pressure_error_map;

  unsigned int n_additional_refinements;

  const double traction_magnitude;

  virtual void make_grid() override;

  virtual void save_postprocessing_results() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;

};

template <int dim>
CouetteFlow<dim>::CouetteFlow
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
n_additional_refinements(0),
traction_magnitude(1.0)
//test_type(convergence_parameters.test_type),
//time_stepping(parameters.time_discretization_parameters),
//navier_stokes(parameters.navier_stokes_parameters,
//              time_stepping,
//              velocity,
//              pressure,
//              this->mapping,
//              this->pcout,
//              this->computing_timer),
//t_0(1.0),
//exact_solution(
//  std::make_shared<EquationData::Couette::VelocityExactSolution<dim>>(
//    t_0,
//    parameters.Re,
//    1.0)),
//traction_vector(
//  std::make_shared<EquationData::Couette::TractionVector<dim>>(t_0)),
//convergence_table(velocity, *exact_solution)
{
  // The Couette flow is a 2-dimensional problem.
  AssertDimension(dim, 2);
}

template <>
void CouetteFlow<2>::make_grid()
{
  constexpr int dim = 2;

  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  // The solution, being independent of the x-component of the position
  // vector, allows a coarser mesh in the x-direction.
  std::vector<unsigned int> repetitions;
  repetitions.emplace_back(1);
  repetitions.emplace_back(3);

  GridGenerator::subdivided_hyper_rectangle(this->triangulation,
                                            repetitions,
                                            Point<dim>(0., 0.),
                                            Point<dim>(1., 1.),
                                            true);

  // The infinite domain is implemented with periodic boundary conditions,
  // this periodicity has to be first implemented in the triangulation.
  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  unsigned int n_refinements
  = this->parameters.spatial_discretization_parameters.n_initial_global_refinements
  + n_additional_refinements;

  this->triangulation.refine_global(n_refinements);

}

template <int dim>
void CouetteFlow<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  // The domain represents an infinite channel. In order to obtain the
  // analytical solution, periodic boundary conditions need to be
  // implemented.
  this->velocity->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);
  this->pressure->boundary_conditions.set_periodic_bcs(left_bndry_id, right_bndry_id, 0);

  // No-slip boundary conditions on the lower plate
  this->velocity->boundary_conditions.set_dirichlet_bcs(bottom_bndry_id);
  // Traction boundary condition on the upper plate
  this->velocity->boundary_conditions.set_neumann_bcs
  (top_bndry_id,
   std::make_shared<EquationData::Couette::TractionVector<dim>>(traction_magnitude));

  this->velocity->apply_boundary_conditions(/* print_summary */ false);
  this->pressure->apply_boundary_conditions(/* print_summary */ false);

}

template <int dim>
void CouetteFlow<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  // The initial conditions are zero and the velocity's Dirichlet
  // boundary conditions are homogeneous. This allows to just
  // set the solution vectors to zero without of projecting a function
  // or distributing constraints.
  this->velocity->set_solution_vectors_to_zero();
  this->pressure->set_solution_vectors_to_zero();

}

template <int dim>
void CouetteFlow<dim>::save_postprocessing_results()
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

  using namespace EquationData::Couette;
  {
    const VelocityExactSolution<dim> velocity_function(this->parameters.Re,
                                                         current_time);

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

}

template <int dim>
void CouetteFlow<dim>::run()
{
  using TestType = ConvergenceTest::ConvergenceTestType;

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

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      RunTimeParameters::HydrodynamicProblemParameters problem_parameters("CouetteFlow.prm");
      ConvergenceTest::ConvergenceTestParameters convergence_parameters("CouetteFlowConvergence.prm");

      CouetteFlow<2> couette_flow(problem_parameters, convergence_parameters);

      couette_flow.run();

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
