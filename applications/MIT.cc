#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/heat_equation.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <memory>
#include <string>
#include <iomanip>
namespace RMHD
{

using namespace dealii;

template <int dim>
class Step35 : public Problem<dim>
{
public:
  Step35(const RunTimeParameters::ParameterSet &parameters);

  void run();
private:
  const RunTimeParameters::ParameterSet       &params;

  std::vector<types::boundary_id>             boundary_ids;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  std::shared_ptr<Entities::ScalarEntity<dim>>  temperature;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  std::shared_ptr<Mapping<dim>>               mapping;

  NavierStokesProjection<dim>                 navier_stokes;
  
  HeatEquation<dim>                           heat_equation;

  BenchmarkData::MIT<dim>                     mit_benchmark;

  void make_grid(const unsigned int n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();
  void postprocessing();
  void output();
  void update_solution_vectors();
};

template <int dim>
Step35<dim>::Step35(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(parameters),
params(parameters),
velocity(std::make_shared<Entities::VectorEntity<dim>>(parameters.p_fe_degree + 1, this->triangulation)),
pressure(std::make_shared<Entities::ScalarEntity<dim>>(parameters.p_fe_degree, this->triangulation)),
temperature(std::make_shared<Entities::ScalarEntity<dim>>(parameters.temperature_fe_degree, this->triangulation)),
time_stepping(parameters.time_stepping_parameters),
mapping(new MappingQ<dim>(1)),
navier_stokes(parameters,
              time_stepping,
              velocity,
              pressure,
              temperature,
              mapping,
              this->pcout,
              this->computing_timer),
heat_equation(parameters,
              time_stepping,
              temperature,
              velocity,
              mapping,
              this->pcout,
              this->computing_timer),
mit_benchmark(velocity,
              pressure,
              temperature,
              time_stepping,
              mapping,
              this->pcout,
              this->computing_timer)
{
  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity->reinit();
  pressure->reinit();
  temperature->reinit();
  initialize();

  this->container.add_entity(velocity);
  this->container.add_entity(pressure, false);
  this->container.add_entity(navier_stokes.phi, false);
  this->container.add_entity(temperature, false);
}

template <int dim>
void Step35<dim>::make_grid(const unsigned int n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  std::vector<unsigned int> repetitions;
  repetitions.emplace_back(1);
  repetitions.emplace_back(5);

  GridGenerator::subdivided_hyper_rectangle(this->triangulation,
                                 repetitions,
                                 Point<2>(0.0, 0.0),
                                 Point<2>(1.0, 8.0),
                                 true);

  this->triangulation.refine_global(n_global_refinements);

  boundary_ids = this->triangulation.get_boundary_ids();

  *(this->pcout) << "Number of refines                        = "
              << n_global_refinements << std::endl;
  *(this->pcout) << "Number of active cells                   = "
              << this->triangulation.n_global_active_cells() << std::endl;
}

template <int dim>
void Step35<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();
  temperature->setup_dofs();
  
  *(this->pcout) << "Number of velocity degrees of freedom    = "
                 << (velocity->dof_handler)->n_dofs()
                 << std::endl
                 << "Number of pressure degrees of freedom    = "
                 << pressure->dof_handler->n_dofs()
                 << std::endl
                 << "Number of temperature degrees of freedom = "
                 << temperature->dof_handler->n_dofs()
                 << std::endl;
}

template <int dim>
void Step35<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->boundary_conditions.set_dirichlet_bcs(0);
  velocity->boundary_conditions.set_dirichlet_bcs(1);
  velocity->boundary_conditions.set_dirichlet_bcs(2);
  velocity->boundary_conditions.set_dirichlet_bcs(3);
  
  temperature->boundary_conditions.set_dirichlet_bcs(0,
    std::make_shared<Functions::ConstantFunction<dim>>(0.5));
  temperature->boundary_conditions.set_dirichlet_bcs(1,
    std::make_shared<Functions::ConstantFunction<dim>>(-0.5));

  velocity->apply_boundary_conditions();

  pressure->apply_boundary_conditions();

  temperature->apply_boundary_conditions();
}

template <int dim>
void Step35<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  Functions::ZeroFunction<dim> zero_vector_function(dim);
  Functions::ZeroFunction<dim> zero_scalar_function(1);

  this->set_initial_conditions(velocity, 
                               zero_vector_function, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               zero_scalar_function, 
                               time_stepping);
  this->set_initial_conditions(temperature,
                               zero_scalar_function, 
                               time_stepping);
  velocity->solution = velocity->old_solution;
  pressure->solution = pressure->old_solution;
  temperature->solution = temperature->old_solution;
  output();
}

template <int dim>
void Step35<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  mit_benchmark.compute_benchmark_data();

  std::cout.precision(1);
  /*! @attention For some reason, a run time error happens when I try
      to only use one pcout */
  *this->pcout << time_stepping << ", ";
  *this->pcout << mit_benchmark
               << ", Norms: ("
               << std::noshowpos << std::scientific
               << navier_stokes.get_diffusion_step_rhs_norm()
               << ", "
               << navier_stokes.get_projection_step_rhs_norm()
               << ", "
               << heat_equation.get_rhs_norm()
               << ") ["
               << std::setw(5) 
               << std::fixed
               << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
               << "%] \r";
}

template <int dim>
void Step35<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(*velocity->dof_handler,
                           velocity->solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(*pressure->dof_handler, 
                           pressure->solution, 
                           "Pressure");
  data_out.add_data_vector(*temperature->dof_handler, 
                           temperature->solution, 
                           "Temperature");
  data_out.build_patches(velocity->fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record("./",
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

template <int dim>
void Step35<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
  temperature->update_solution_vectors();
}

template <int dim>
void Step35<dim>::run()
{
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();
    heat_equation.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage
    postprocessing();

    if (time_stepping.get_step_number() % 
        this->prm.adaptive_meshing_interval == 0)
      this->adaptive_mesh_refinement();

    if ((time_stepping.get_step_number() % 
          this->prm.graphical_output_interval == 0) ||
        (time_stepping.get_current_time() == 
                   time_stepping.get_end_time()))
      output();
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

      RunTimeParameters::ParameterSet parameter_set("MIT.prm");

      Step35<2> simulation(parameter_set);
      simulation.run();

      std::cout.precision(1);
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
  std::cout << "----------------------------------------------------"
            << std::endl;
  return 0;
}
