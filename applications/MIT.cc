#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/convection_diffusion_solver.h>
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
#include <fstream>
#include <memory>
#include <string>
#include <iomanip>
namespace RMHD
{

using namespace dealii;

/*!
 * @class MITBenchmark
 * @brief Class solving the problem formulated in the MIT benchmark.
 * @details The benchmark considers the case of a buoyancy-driven flow
 * for which the Boussinesq approximation is assumed to hold true,
 * <em> i. e.</em>, the fluid's behaviour is described by the following
 * dimensionless equations
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 * \pd{\bs{u}}{t} + \bs{u} \cdot ( \nabla \otimes \bs{u}) &=
 * - \nabla p + \sqrt{\dfrac{\Prandtl}{\Rayleigh}} \nabla^2 \bs{u} +
 * \vartheta \ey,
 * &\forall (\bs{x}, t) \in \Omega \times \left[0, T \right]\\
 * \nabla \cdot \bs{u} &= 0,
 * &\forall (\bs{x}, t) \in \Omega \times \left[0, T \right]\\
 * \pd{\vartheta}{t} + \bs{u} \cdot \nabla \vartheta &=
 * \dfrac{1}{\sqrt{\Rayleigh \Prandtl}} \nabla^2 \vartheta
 * &\forall (\bs{x}, t) \in \Omega \times \left[0, T \right]
 * \end{aligned}
 * \end{equation*}
 * \f]
 * where \f$ \bs{u} \f$,  \f$ \,p \f$, \f$ \,\vartheta \f$, \f$\, \Prandtl \f$,
 * \f$ \,\Rayleigh \f$, \f$ \,\bs{x} \f$, \f$\, t \f$, \f$\, \Omega \f$ and
 * \f$ T \f$ are the velocity, pressure, temperature,
 * Prandtl number, Rayleigh number, position vector, time, domain and
 * final time respectively. The problem's domain is a long cavity
 * \f$ \Omega = [0,1] \times [0,8] \f$, whose boundary is divided into the
 * left wall \f$ \Gamma_1 \f$, the right wall \f$ \Gamma_2 \f$,
 * the bottom wall \f$ \Gamma_3 \f$ and the top wall \f$ \Gamma_4 \f$.
 * The boundary conditions are
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 * \bs{u} &= \bs{0}, &\forall (\bs{x}, t) &\in \partial\Omega \times \left[0, T \right], \\
 * \vartheta &= \frac{1}{2}, &\forall (\bs{x}, t) &\in \Gamma_1 \times \left[0, T \right], \\
 * \vartheta &= -\frac{1}{2}, &\forall (\bs{x}, t) &\in \Gamma_2 \times \left[0, T \right], \\
 * \nabla \vartheta \cdot \bs{n} &= 0, &\forall (\bs{x}, t) &\in \Gamma_3 \cup \Gamma_4 \times \left[0, T \right]
 * \end{aligned}
 * \end{equation*}
 * \f]
 * the initial conditions are given by
 * \f[
 * \bs{u}_0 = \bs{0}, \quad p_0 = 0, \quad \textrm{and} \quad
 * \vartheta_0 = 0.
 * \f]
 * and the parameters are \f$ \Prandtl = 0.71 \f$ and
 * \f$ \Rayleigh = 3.4\times 10^5. \f$
 * @note The temperature's Dirichlet boundary conditions are implemented
 * with a factor \f$ [1-\exp(-10 t)] \f$ to smooth the dynamic response
 * of the system.
 * @todo Add a picture
 */
template <int dim>
class MITBenchmark : public Problem<dim>
{
public:
  MITBenchmark(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  std::shared_ptr<Entities::ScalarEntity<dim>>  temperature;

  std::shared_ptr<EquationData::MIT::TemperatureBoundaryCondition<dim>>
                                                temperature_boundary_conditions;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  HeatEquation<dim>                             heat_equation;

  BenchmarkData::MIT<dim>                       mit_benchmark;

  EquationData::MIT::GravityUnitVector<dim>     gravity_vector;

  double                                        cfl_number;

  std::ofstream                                 cfl_output_file;


  const types::boundary_id  left_bndry_id = 1;
  const types::boundary_id  right_bndry_id = 2;
  const types::boundary_id  top_bndry_id = 3;
  const types::boundary_id  bottom_bndry_id = 4;

  void make_grid();

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_solution_vectors();
};

template <int dim>
MITBenchmark<dim>::MITBenchmark(const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
velocity(std::make_shared<Entities::VectorEntity<dim>>(
              parameters.fe_degree_velocity,
              this->triangulation,
              "Velocity")),
pressure(std::make_shared<Entities::ScalarEntity<dim>>(
              parameters.fe_degree_pressure,
              this->triangulation,
              "Pressure")),
temperature(std::make_shared<Entities::ScalarEntity<dim>>(
              parameters.fe_degree_temperature,
              this->triangulation,
              "Temperature")),
temperature_boundary_conditions(
              std::make_shared<EquationData::MIT::TemperatureBoundaryCondition<dim>>(
              parameters.time_discretization_parameters.start_time)),
time_stepping(parameters.time_discretization_parameters),
navier_stokes(parameters.navier_stokes_parameters,
              time_stepping,
              velocity,
              pressure,
              temperature,
              this->mapping,
              this->pcout,
              this->computing_timer),
heat_equation(parameters.heat_equation_parameters,
              time_stepping,
              temperature,
              velocity,
              this->mapping,
              this->pcout,
              this->computing_timer),
mit_benchmark(velocity,
              pressure,
              temperature,
              time_stepping,
              1,
              2,
              this->mapping,
              this->pcout,
              this->computing_timer),
gravity_vector(parameters.time_discretization_parameters.start_time),
cfl_output_file("MIT_cfl_number.csv")
{
  *this->pcout << parameters << std::endl << std::endl;

  AssertDimension(dim, 2);
  navier_stokes.set_gravity_vector(gravity_vector);
  make_grid();
  setup_dofs();
  setup_constraints();
  velocity->reinit();
  pressure->reinit();
  temperature->reinit();
  initialize();

  // Stores all the fields to the SolutionTransfor container
  this->container.add_entity(velocity);
  this->container.add_entity(pressure, false);
  this->container.add_entity(navier_stokes.phi, false);
  this->container.add_entity(temperature, false);

  cfl_output_file << "Time" << "," << "CFL" << std::endl;
}

template <>
void MITBenchmark<2>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  constexpr int dim = 2;
  constexpr double tol = 1e-12;

  Triangulation<dim>  tria;

  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {16u, 16u*8u},
                                            Point<dim>(0., 0.),
                                            Point<dim>(1., 8.));

  for (auto &face: tria.active_face_iterators())
    if (face->at_boundary())
    {
      const Point<dim> center = face->center();

      if (std::abs(center[1]) > tol && std::abs(center[1] - 8.0) > tol)
      {
        if (std::abs(center[0]) < tol)
          face->set_boundary_id(left_bndry_id);
        else if (std::abs(center[0] - 1.0) < tol)
          face->set_boundary_id(right_bndry_id);
        else
          Assert(false, ExcInternalError());
      }
      else if (std::abs(center[1]) < tol)
        face->set_boundary_id(bottom_bndry_id);
      else if (std::abs(center[1] - 8.0) < tol)
        face->set_boundary_id(top_bndry_id);
      else
        Assert(false, ExcInternalError());
    }

  // Transforms the mesh
  auto transformation = [](const Point<dim> &point)
      {
        // Compute coordinate of Chebyshev node
        const std::array<double, dim> chebyshev_coords
          {0.5 - 0.5 * cos(point[0] * numbers::PI),
           4.0 - 4.0 * cos(point[1] / 8.0 * numbers::PI)};
        // Take the mean of the Chebyshev and the original coordinate
        return Point<dim>(0.5 * (point[0] + chebyshev_coords[0]),
                          0.5 * (point[1] + chebyshev_coords[1]));
      };
  GridTools::transform(transformation, tria);

  this->triangulation.copy_triangulation(tria);

  // Performs global refinements
  this->triangulation.refine_global(prm.spatial_discretization_parameters.n_initial_global_refinements);

  // Performs a one level local refinement of the cells located at the
  // side walls
  for (unsigned int i=0;
        i<prm.spatial_discretization_parameters.n_initial_boundary_refinements;
        ++i)
  {
    for (auto &cell: this->triangulation.active_cell_iterators())
      if (cell->at_boundary() && cell->is_locally_owned())
        cell->set_refine_flag();
    this->triangulation.execute_coarsening_and_refinement();
  }

  *(this->pcout) << "Triangulation:"
                 << std::endl
                 << " Number of initial active cells           = "
                 << this->triangulation.n_global_active_cells()
                 << std::endl
                 << " Maximum aspect ratio                     = "
                 << GridTools::compute_maximum_aspect_ratio(*this->mapping,
                                                            this->triangulation,
                                                            QGauss<dim>(2))
                 << std::endl << std::endl;
}

template <int dim>
void MITBenchmark<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  // Sets up the locally owned and relevant degrees of freedom of each
  // field.
  velocity->setup_dofs();
  pressure->setup_dofs();
  temperature->setup_dofs();

  *(this->pcout) << "Spatial discretization:"
                 << std::endl
                 << " Number of velocity degrees of freedom    = "
                 << (velocity->dof_handler)->n_dofs()
                 << std::endl
                 << " Number of pressure degrees of freedom    = "
                 << pressure->dof_handler->n_dofs()
                 << std::endl
                 << " Number of temperature degrees of freedom = "
                 << temperature->dof_handler->n_dofs()
                 << std::endl
                 << " Number of total degrees of freedom       = "
                 << (velocity->dof_handler->n_dofs() +
                     pressure->dof_handler->n_dofs() +
                     temperature->dof_handler->n_dofs())
                 << std::endl << std::endl;
}

template <int dim>
void MITBenchmark<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  // Homogeneous Dirichlet boundary conditions over the whole boundary
  // for the velocity field.
  velocity->boundary_conditions.set_dirichlet_bcs(left_bndry_id);
  velocity->boundary_conditions.set_dirichlet_bcs(right_bndry_id);
  velocity->boundary_conditions.set_dirichlet_bcs(top_bndry_id);
  velocity->boundary_conditions.set_dirichlet_bcs(bottom_bndry_id);

  // The pressure itself has no boundary conditions, leading to a pure
  // Neumann problem. A datum ensures the well-posedness of the problem
  // and The Navier-Stokes solver will enforce a zero mean value
  // constraint.
  pressure->boundary_conditions.set_datum_at_boundary();

  // Inhomogeneous time dependent Dirichlet boundary conditions over
  // the side walls and homogeneous Neumann boundary conditions over
  // the bottom and top walls for the temperature field.
  temperature->boundary_conditions.set_dirichlet_bcs(
    left_bndry_id, temperature_boundary_conditions, true);
  temperature->boundary_conditions.set_dirichlet_bcs(
    right_bndry_id, temperature_boundary_conditions, true);
  temperature->boundary_conditions.set_neumann_bcs(top_bndry_id);
  temperature->boundary_conditions.set_neumann_bcs(bottom_bndry_id);

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();
  temperature->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
  temperature->apply_boundary_conditions();
}

template <int dim>
void MITBenchmark<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  // Due to the homogeneous boundary conditions of the velocity, one may
  // directly set the solution vectors to zero instead of projecting.
  velocity->set_solution_vectors_to_zero();
  pressure->set_solution_vectors_to_zero();

  // The temperature's boundary conditions and its zero scalar field as
  // initial condition allows one to avoid a projection
  // by just distributing the constraints to the zero'ed out vector.
  temperature->set_solution_vectors_to_zero();

  {
    LinearAlgebra::MPI::Vector distributed_old_temperature(temperature->distributed_vector);

    distributed_old_temperature = temperature->old_solution;

    temperature->constraints.distribute(distributed_old_temperature);

    temperature->old_solution   = distributed_old_temperature;
  }

  // Outputs the initial conditions
  temperature->solution = temperature->old_solution;
  output();
}

template <int dim>
void MITBenchmark<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  // Computes all the benchmark's data. See documentation of the
  // class for further information.
  mit_benchmark.compute_benchmark_data();

  /*! @attention For some reason, a run time error happens when I try
      to only use one pcout */
  *this->pcout << "    ";
  *this->pcout << mit_benchmark << std::endl;
  *this->pcout << "    CFL = "
               << cfl_number
               << ", Norms: ("
               << std::noshowpos << std::scientific
               << navier_stokes.get_diffusion_step_rhs_norm()
               << ", "
               << navier_stokes.get_projection_step_rhs_norm()
               << ", "
               << heat_equation.get_rhs_norm()
               << ")"
               << std::endl;

  if (time_stepping.get_step_number() % 10 == 0)
    cfl_output_file << time_stepping.get_current_time() << ","
                    << cfl_number << std::endl;
}

template <int dim>
void MITBenchmark<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  // Explicit declaration of the velocity as a vector
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  // Loading the DataOut instance with the solution vectors
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

  // To properly showcase the velocity field (whose k-th order finite
  // elements are one order higher than those of the pressure field),
  // the k-th order elements are interpolated to four (k-1)-th order
  // elements. In other words, the triangulation visualized in the
  // *.pvtu file is one globl refinement finer than the actual
  // triangulation.
  data_out.build_patches(velocity->fe_degree);

  // Writes the DataOut instance to the file.
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

template <int dim>
void MITBenchmark<dim>::update_solution_vectors()
{
  // Sets the solution vectors at t^{k-j} to those at t^{k-j+1}
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
  temperature->update_solution_vectors();
}

template <int dim>
void MITBenchmark<dim>::run()
{
  const unsigned int n_steps = this->prm.time_discretization_parameters.n_maximum_steps;

  *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
               << std::endl;
  while (time_stepping.get_current_time() < time_stepping.get_end_time() &&
         time_stepping.get_step_number() < n_steps)
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(time_stepping, cfl_number));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and constraints to t^{k}
    temperature->boundary_conditions.set_time(time_stepping.get_next_time());
    temperature->update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    heat_equation.solve();
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();
    *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                 << std::endl;

    // Performs post-processing
    postprocessing();

    // Performs coarsening and refining of the triangulation
    if (time_stepping.get_step_number() %
        this->prm.spatial_discretization_parameters.adaptive_mesh_refinement_frequency == 0)
      this->adaptive_mesh_refinement();

    // Graphical output of the solution vectors
    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()))
      output();
  }

  // Prints the benchmark's data to the .txt file.
  mit_benchmark.print_data_to_file("MIT_benchmark");
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string parameter_filename;
      if (argc >= 2)
        parameter_filename = argv[1];
      else
        parameter_filename = "MIT.prm";

      RunTimeParameters::ProblemParameters parameter_set(parameter_filename);

      MITBenchmark<2> simulation(parameter_set);

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
  return 0;
}
