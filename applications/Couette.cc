/*!
 *@file Couette
 *@brief The source file solving the Couette flow problem.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/convergence_struct.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

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
 * @class Couette
 * @brief This class solves the Couette flow problem.
 * @details The Couette flow was chosen as a test problem to verify the
 * correct implementation of the Neumann boundary conditions in the
 * @ref NavierStokesProjection solver. The problem considers stationary
 * laminar flow of fluid between two infinitely long horiozontal plates,
 * where the lower plate is fixed and the upper one is being moved. 
 * The problem is governed by the equation
 * \f[
 * \ddsqr{u_\mathrm{x}}{y} = 0, \quad \forall \bs{x} \in \Omega
 * \f]
 * which can be easily obtained from the momentum equation by
 * considering an isotropic and homogeneous fluid, an unidirectional and
 * stationary flow in the \f$ x\f$-direction, neglecting body forces and
 * a vanishing horizontal pressure gradient. An unique solution is 
 * obtained by considering a no-slip boundary condition at the lower 
 * plate and a traction vector
 * \f$ \bs{t} = t_0 \bs{e}_\textrm{x} \f$ applied to the upper plate.
 * This leads to
 * \f[
 * \bs{u} = t_0 \Reynolds \dfrac{y}{H} \bs{e}_\textrm{x},
 * \f]
 * where \f$ \Reynolds \f$ and \f$ H \f$ are the Reynolds
 * number and the height of the channel. The stationary solution is
 * obtained at around \f$ t ~ H^2 \Reynolds \f$.
 * @note Periodic boundary conditions are implemented in order to 
 * simulate an infinitely long channel.
 */  
template <int dim>
class Couette : public Problem<dim>
{
public:

  Couette(const RunTimeParameters::ParameterSet &parameters);

  void run(const bool flag_convergence_test);

private:

  std::vector<types::boundary_id>             boundary_ids;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  LinearAlgebra::MPI::Vector                  error;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  std::shared_ptr<Mapping<dim>>               mapping;

  NavierStokesProjection<dim>                 navier_stokes;

  const double                                t_0;

  std::shared_ptr<EquationData::Couette::VelocityExactSolution<dim>>
                                              exact_solution;

  std::shared_ptr<EquationData::Couette::TractionVector<dim>>
                                              traction_vector;

  ConvergenceAnalysisData<dim>                convergence_table;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_entities();

  void solve(const unsigned int &level);

  void compute_error_vector(LinearAlgebra::MPI::Vector  &error_vector,
                            Entities::EntityBase<dim>   &entity,
                            Function<dim>               &exact_solution);
};

template <int dim>
Couette<dim>::Couette(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(parameters),
velocity(std::make_shared<Entities::VectorEntity<dim>>(parameters.p_fe_degree + 1,
                                                       this->triangulation,
                                                       "velocity")),
pressure(std::make_shared<Entities::ScalarEntity<dim>>(parameters.p_fe_degree,
                                                       this->triangulation,
                                                       "pressure")),
time_stepping(parameters.time_stepping_parameters),
mapping(std::make_shared<MappingQ<dim>>(1)),
navier_stokes(parameters,
              time_stepping,
              velocity,
              pressure,
              mapping,
              this->pcout,
              this->computing_timer),
t_0(1.0),
exact_solution(
  std::make_shared<EquationData::Couette::VelocityExactSolution<dim>>(
              t_0,
              parameters.Re,
              1.0)),
traction_vector(
  std::make_shared<EquationData::Couette::TractionVector<dim>>(t_0)),
convergence_table(velocity, *exact_solution)
{
  // The Couette flow is a 2-dimensional problem.
  AssertDimension(dim, 2);
}

template <int dim>
void Couette<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  // The solution, being independent of the x-component of the position
  // vector, allows a coarser mesh in the x-direction.
  std::vector<unsigned int> repetitions;
  repetitions.emplace_back(1);
  repetitions.emplace_back(3);

  GridGenerator::subdivided_hyper_rectangle(this->triangulation,
                                 repetitions,
                                 Point<2>(0., 0.),
                                 Point<2>(1., 1.),
                                 true);
  
  // The infinite domain is implemented with periodic boundary conditions,
  // this periodicity has to be first implemented in the triangulation.
  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(n_global_refinements);
  boundary_ids = this->triangulation.get_boundary_ids();
}

template <int dim>
void Couette<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();
  *(this->pcout)  << "  Number of active cells                = " 
                  << this->triangulation.n_global_active_cells() << std::endl;
  *(this->pcout)  << "  Number of velocity degrees of freedom = " 
                  << velocity->dof_handler->n_dofs()
                  << std::endl
                  << "  Number of pressure degrees of freedom = " 
                  << pressure->dof_handler->n_dofs()
                  << std::endl
                  << "  Number of total degrees of freedom    = " 
                  << (pressure->dof_handler->n_dofs() + 
                     velocity->dof_handler->n_dofs())
                  << std::endl;
}

template <int dim>
void Couette<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->boundary_conditions.clear();
  pressure->boundary_conditions.clear();

  // The domain represents an infinite channel. In order to obtain the
  // analytical solution, periodic boundary conditions need to be 
  // implemented. 
  velocity->boundary_conditions.set_periodic_bcs(0, 1, 0);
  pressure->boundary_conditions.set_periodic_bcs(0, 1, 0);
  // No-slip boundary conditions on the lower plate
  velocity->boundary_conditions.set_dirichlet_bcs(2);
  // The upper plate is displaced by a traction vector
  velocity->boundary_conditions.set_neumann_bcs(3, traction_vector);

  
  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void Couette<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  // The initial conditions are zero and the velocity's Dirichlet
  // boundary conditions are homogenous. This allows to just
  // set the solution vectors to zero instead of projecting a function
  // or distributing constraints.
  velocity->set_solution_vectors_to_zero();
  pressure->set_solution_vectors_to_zero();
}

template <int dim>
void Couette<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  std::cout.precision(1);
  *(this->pcout)  << time_stepping
                  << " Norms = ("
                  << std::noshowpos << std::scientific
                  << navier_stokes.get_diffusion_step_rhs_norm()
                  << ", "
                  << navier_stokes.get_projection_step_rhs_norm()
                  << ") CFL = "
                  << navier_stokes.get_cfl_number()
                  << " ["
                  << std::setw(5) 
                  << std::fixed
                  << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                  << "%] \r";
}

template <int dim>
void Couette<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  this->compute_error(error,
                      velocity,
                      *exact_solution);

  std::vector<std::string> names(dim, "velocity");
  std::vector<std::string> error_name(dim, "velocity_error");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(*(velocity->dof_handler),
                           velocity->solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(*(velocity->dof_handler),
                           error,
                           error_name, 
                           component_interpretation);
  data_out.add_data_vector(*(pressure->dof_handler), 
                           pressure->solution, 
                           "pressure");
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
void Couette<dim>::update_entities()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void Couette<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  velocity->reinit();
  pressure->reinit();
  error.reinit(velocity->solution);
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  output();

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

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();

    // Post-processing
    postprocessing();

    // Graphical output
    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_interval == 0) ||
        (time_stepping.get_current_time() == 
          time_stepping.get_end_time()))
      output();
  }
  
  convergence_table.update_table(level, 
                                 time_stepping.get_previous_step_size(), 
                                 this->prm.flag_spatial_convergence_test);
  
  *(this->pcout) << std::endl;
  *(this->pcout) << std::endl;
}

template <int dim>
void Couette<dim>::run(const bool flag_convergence_test)
{
  // Set ups the initial triangulation
  make_grid(this->prm.initial_refinement_level);

  // The following if allows to perform either spatial or temporal
  // convergence tests, depending on the settings described in the
  // parameter file.
  if (flag_convergence_test)
    for (unsigned int level = this->prm.initial_refinement_level; 
          level <= this->prm.final_refinement_level; ++level)
    {
      std::cout.precision(1);
      *(this->pcout)  << "Solving until t = " 
                      << std::fixed << time_stepping.get_end_time()
                      << " with a refinement level of " << level 
                      << std::endl;
      time_stepping.restart();
      solve(level);
      this->triangulation.refine_global();
      navier_stokes.reset_phi();
    }
  else
  {
    for (unsigned int cycle = 0; 
         cycle < this->prm.temporal_convergence_cycles; ++cycle)
    {
      double time_step = this->prm.time_stepping_parameters.initial_time_step *
                         pow(this->prm.time_step_scaling_factor, cycle);
      std::cout.precision(1);
      *(this->pcout)  << "Solving until t = " 
                      << std::fixed << time_stepping.get_end_time()
                      << " with a refinement level of " 
                      << this->prm.initial_refinement_level << std::endl;
      time_stepping.restart();
      time_stepping.set_desired_next_step_size(time_step);
      solve(this->prm.initial_refinement_level);
      navier_stokes.reset_phi();
    }
  }
  
  *(this->pcout) << convergence_table;

  std::ostringstream tablefilename;
  tablefilename << ((this->prm.flag_spatial_convergence_test) ?
                    "CouetteSpatialTest" : "CouetteTemporalTest_Level")
                << this->prm.initial_refinement_level
                << "_Re"
                << this->prm.Re;

  convergence_table.write_text(tablefilename.str() + "_Velocity");
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

      RunTimeParameters::ParameterSet parameter_set("Couette.prm");

      Couette<2> simulation(parameter_set);
      
      simulation.run(parameter_set.flag_spatial_convergence_test);
  
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
