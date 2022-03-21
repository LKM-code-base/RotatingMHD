#include <rotatingMHD/angular_velocity.h>
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <iomanip>

namespace ChristensenBenchmark
{

using namespace dealii;
using namespace RMHD;

namespace EquationData
{

/*!
 * @class Omega
 *
 * @brief The angular velocity of the rotating frame of reference.
 *
 * @details Given by
 * \f[
 * \ \bs{\omega} = \ez
 * \f]
 * where \f$ \bs{\omega} \f$ is the angular velocity and
 * \f$ \ez \f$ the unit vector in the \f$ z \f$-direction.
 */
template <int dim>
class Omega: public AngularVelocity<dim>
{
public:
  /*
   * @brief Default constructor.
   */
  Omega(const double time = 0);

  /*
   * @brief Overloads the value method such that
   * \f[
   * \ \bs{\omega} = \ez\,.
   * \f]
   */
  virtual typename AngularVelocity<dim>::value_type value() const override;

};



template <int dim>
Omega<dim>::Omega(const double time)
:
AngularVelocity<dim>(time)
{}



template <int dim>
typename AngularVelocity<dim>::value_type Omega<dim>::value() const
{
  typename AngularVelocity<dim>::value_type value;
  value = 0;

  if constexpr(dim == 2)
    value[0] = 1.0;
  else if constexpr(dim == 3)
    value[2] = 1.0;

  return (value);
}

/*!
 * @class TemperatureInitialCondition
 *
 * @brief The initial temperature field of the Christensen benchmark.
 *
 * @details Given by
 * \f[
 * \vartheta = \frac{r_o r_i}{r} - r_i + \frac{210 A}{\sqrt{17920 \pi}}
 * (1-3x^2+3x^4-x^6) \sin^4 \theta \cos 4 \phi
 * \f]
 * where \f$ \vartheta \f$ is the temperature field,
 * \f$ r \f$ the radius,
 * \f$ r_i \f$ the inner radius of the shell,
 * \f$ r_o \f$ the outer radius,
 * \f$ A \f$ the amplitude of the harmonic perturbation,
 * \f$ x \f$ a quantitiy defined as \f$ x = 2r - r_i - r_o\f$,
 * \f$ \theta \f$ the colatitude (polar angle) and
 * \f$ \phi \f$ the longitude (azimuthal angle).
 */
template <int dim>
class TemperatureInitialCondition : public Function<dim>
{
public:
  TemperatureInitialCondition(const double inner_radius,
                              const double outer_radius,
                              const double A = 0.1,
                              const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  /*!
   * @brief Inner radius of the shell.
   */
  const double inner_radius;

  /*!
   * @brief Outer radius of the shell.
   */
  const double outer_radius;

  /*!
   * @brief Amplitude of the harmonic perturbation.
   */
  const double A;
};



template <int dim>
TemperatureInitialCondition<dim>::TemperatureInitialCondition
(const double inner_radius,
 const double outer_radius,
 const double A,
 const double time)
:
Function<dim>(1, time),
inner_radius(inner_radius),
outer_radius(outer_radius),
A(A)
{
  Assert(outer_radius > inner_radius, ExcMessage("The outer radius has to be greater then the inner radius"))
}



template<int dim>
double TemperatureInitialCondition<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  double temperature;

  const std::array<double, dim> spherical_coordinates = GeometricUtilities::Coordinates::to_spherical(point);

  // Radius
  const double r        = spherical_coordinates[0];
  // Azimuthal angle
  const double phi      = spherical_coordinates[1];
  // Polar angle
  double theta;
  if constexpr(dim == 2)
    theta = numbers::PI_2;
  else if constexpr(dim == 3)
    theta = spherical_coordinates[2];

  const double x_0    = 2. * r - inner_radius - outer_radius;

  temperature = outer_radius * inner_radius / r
                - inner_radius
                + 210. * A / std::sqrt(17920. * M_PI) *
                  (1. - 3. * x_0 * x_0 + 3. * std::pow(x_0, 4) - std::pow(x_0,6)) *
                  pow(std::sin(theta), 4) *
                  std::cos(4. * phi);

  return temperature;
}



/*!
 * @class TemperatureBoundaryCondition
 *
 * @brief The boundary conditions of the temperature field of the
 * Christensen benchmark.
 *
 * @details At the inner boundary the temperature is set to \f$ 1.0 \f$
 * and at the outer boundary to \f$ 0.0 \f$.
 *
 */
template <int dim>
class TemperatureBoundaryCondition : public Function<dim>
{
public:
  TemperatureBoundaryCondition(const double inner_radius,
                               const double outer_radius,
                               const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  /*!
   * @brief Inner radius of the shell.
   */
  const double inner_radius;

  /*!
   * @brief Outer radius of the shell.
   */
  const double outer_radius;
};



template <int dim>
TemperatureBoundaryCondition<dim>::TemperatureBoundaryCondition
(const double inner_radius,
 const double outer_radius,
 const double time)
:
Function<dim>(1, time),
inner_radius(inner_radius),
outer_radius(outer_radius)
{
  Assert(outer_radius > inner_radius, ExcMessage("The outer radius has to be greater then the inner radius"))
}



template<int dim>
double TemperatureBoundaryCondition<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double r = point.norm();

  double value = (r > 0.5*(inner_radius + outer_radius)) ? 0.0 : 1.0;

  return (value);
}



/*!
 * @class GravityVector
 *
 * @brief The gravity field
 *
 * @details Given by the linear function
 * \f[
 * \ \bs{g} = \frac{1}{r_o} r\bs{e}_\textrm{r}
 * \f]
 * where \f$ \bs{g} \f$ is the gravity field,
 * \f$ r_o \f$ the outer radius of the shell and
 * \f$ \bs{r} \f$ the radius vector.
 *
 */
template <int dim>
class GravityVector: public TensorFunction<1, dim>
{
public:
  GravityVector(const double outer_radius,
                const double time = 0);

  virtual Tensor<1, dim> value(const Point<dim> &point) const override;

private:

  /*!
   * @brief Outer radius of the shell.
   */
  const double outer_radius;
};



template <int dim>
GravityVector<dim>::GravityVector
(const double outer_radius,
 const double time)
:
TensorFunction<1, dim>(time),
outer_radius(outer_radius)
{}



template <int dim>
Tensor<1, dim> GravityVector<dim>::value(const Point<dim> &point) const
{
  Tensor<1, dim> value(point);

  value /= -outer_radius;

  return value;
}


}  // namespace EquationData



/*!
 * @class Christensen
 *
 * @brief Class solving the problem formulated in the Christensen benchmark.
 *
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
 *
 * @note The temperature's Dirichlet boundary conditions are implemented
 * with a factor \f$ [1-\exp(-10 t)] \f$ to smooth the dynamic response
 * of the system.
 *
 * @todo Add a picture
 */
template <int dim>
class Christensen : public Problem<dim>
{
public:
  Christensen(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  std::ofstream                                 log_file;

  const double                                  inner_radius;

  const double                                  outer_radius;

  const double                                  A;

  const unsigned int                            inner_boundary_id;

  const unsigned int                            outer_boundary_id;

  std::shared_ptr<Entities::FE_VectorField<dim>>  velocity;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  pressure;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  temperature;

  std::shared_ptr<Entities::FE_VectorField<dim>>  magnetic_field;

  std::shared_ptr<EquationData::TemperatureInitialCondition<dim>>
                                                temperature_initial_conditions;

  std::shared_ptr<EquationData::TemperatureBoundaryCondition<dim>>
                                                temperature_boundary_conditions;

  EquationData::GravityVector<dim>              gravity_vector;

  EquationData::Omega<dim>                      angular_velocity;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  ConvectionDiffusionSolver<dim>                             heat_equation;

  BenchmarkData::ChristensenBenchmark<dim>      benchmark_requests;

  double                                        cfl_number;

  void make_grid(const unsigned int n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_solution_vectors();
};

template <int dim>
Christensen<dim>::Christensen(const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
log_file("Christensen_Log.csv"),
inner_radius(7./13.),
outer_radius(20./13.),
A(0.1),
inner_boundary_id(0),
outer_boundary_id(1),
velocity(std::make_shared<Entities::FE_VectorField<dim>>(
           parameters.fe_degree_velocity,
           this->triangulation,
           "Velocity")),
pressure(std::make_shared<Entities::FE_ScalarField<dim>>(
           parameters.fe_degree_pressure,
           this->triangulation,
           "Pressure")),
temperature(std::make_shared<Entities::FE_ScalarField<dim>>(
              parameters.fe_degree_temperature,
              this->triangulation,
              "Temperature")),
magnetic_field(std::make_shared<Entities::FE_VectorField<dim>>(
              1/*parameters.fe_degree_magnetic_field*/,
              this->triangulation,
              "Magnetic field")),
temperature_initial_conditions(
  std::make_shared<EquationData::TemperatureInitialCondition<dim>>(
    inner_radius,
    outer_radius,
    A,
    parameters.time_discretization_parameters.start_time)),
temperature_boundary_conditions(
  std::make_shared<EquationData::TemperatureBoundaryCondition<dim>>(
    inner_radius,
    outer_radius,
    parameters.time_discretization_parameters.start_time)),
gravity_vector(outer_radius,
               parameters.time_discretization_parameters.start_time),
angular_velocity(parameters.time_discretization_parameters.start_time),
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
benchmark_requests(inner_radius, outer_radius)
{
  Assert(outer_radius > inner_radius,
         ExcMessage("The outer radius has to be greater then the inner radius"));

  *this->pcout << parameters << std::endl;

  navier_stokes.set_gravity_vector(gravity_vector);
  navier_stokes.set_angular_velocity_vector(angular_velocity);
  make_grid(parameters.spatial_discretization_parameters.n_initial_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity->setup_vectors();
  pressure->setup_vectors();
  temperature->setup_vectors();
  initialize();

  // Stores all the fields to the SolutionTransfer container
  this->container.add_entity(*velocity);
  this->container.add_entity(*pressure, false);
  this->container.add_entity(*navier_stokes.phi, false);
  this->container.add_entity(*temperature, false);

  log_file << "Step" << ","
           << "Time" << ","
           << "dt" << ","
           << "CFL" << ","
           << "D_norm" << ","
           << "P_norm" << ","
           << "H_norm" << ","
           << std::endl;
}

template <int dim>
void Christensen<dim>::make_grid(const unsigned int n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  // Generates the shell with the inner and outer boundary indicators
  // 0 and 1 respectively
  GridGenerator::hyper_shell(this->triangulation,
                             Point<dim>(),
                             inner_radius,
                             outer_radius,
                             0,
                             true);

  // Performs global refinements
  this->triangulation.refine_global(n_global_refinements);

  for (unsigned int i = 0;
       i < this->prm.spatial_discretization_parameters.n_initial_boundary_refinements;
       ++i)
  {
    for (const auto &cell : this->triangulation.active_cell_iterators())
      if (cell->is_locally_owned() && cell->at_boundary())
        cell->set_refine_flag();

    this->triangulation.execute_coarsening_and_refinement();
  }

  *(this->pcout) << "Triangulation:"
                 << std::endl
                 << " Number of initial active cells           = "
                 << this->triangulation.n_global_active_cells()
                 << std::endl << std::endl;
}

template <int dim>
void Christensen<dim>::setup_dofs()
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
                 << velocity->n_dofs()
                 << std::endl
                 << " Number of pressure degrees of freedom    = "
                 << pressure->n_dofs()
                 << std::endl
                 << " Number of temperature degrees of freedom = "
                 << temperature->n_dofs()
                 << std::endl
                 << " Number of total degrees of freedom       = "
                 << (velocity->n_dofs() + pressure->n_dofs() + temperature->n_dofs())
                 << std::endl << std::endl;
}

template <int dim>
void Christensen<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->clear_boundary_conditions();
  pressure->clear_boundary_conditions();
  temperature->clear_boundary_conditions();

  velocity->setup_boundary_conditions();
  pressure->setup_boundary_conditions();
  temperature->setup_boundary_conditions();

  // Homogeneous Dirichlet boundary conditions over the whole boundary
  // for the velocity field.
  velocity->set_dirichlet_boundary_condition(inner_boundary_id);
  velocity->set_dirichlet_boundary_condition(outer_boundary_id);

  // The pressure itself has no boundary conditions. A datum needs to be
  // set to make the system matrix regular
  pressure->set_datum_boundary_condition();

  // Inhomogeneous Dirichlet boundary conditions over the whole boundary
  // for the velocity field.
  temperature->set_dirichlet_boundary_condition(inner_boundary_id,
                                                temperature_boundary_conditions);
  temperature->set_dirichlet_boundary_condition(outer_boundary_id,
                                                temperature_boundary_conditions);

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();
  temperature->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
  temperature->apply_boundary_conditions();
}

template <int dim>
void Christensen<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  // Due to the homogeneous boundary conditions of the velocity, one may
  // directly set the solution vectors to zero instead of projecting.
  velocity->set_solution_vectors_to_zero();
  pressure->set_solution_vectors_to_zero();

  // The initial conditions describe a linear function with a
  // trigonometric perturbation. See the funciton for further reference
  this->set_initial_conditions(temperature,
                               *temperature_initial_conditions,
                               time_stepping);
}

template <int dim>
void Christensen<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  // Computes all the benchmark's data. See documentation of the
  // class for further information.
  benchmark_requests.update(time_stepping.get_current_time(),
                               time_stepping.get_step_number(),
                               *velocity,
                               *temperature,
                               *this->mapping);

  // Outputs CFL number and norms of the right-hand sides
  *this->pcout << "CFL = " << std::scientific << std::setprecision(2)
               << cfl_number
               << ", Norms = ("
               << std::noshowpos << std::setprecision(3)
               << navier_stokes.get_diffusion_step_rhs_norm()
               << ", "
               << navier_stokes.get_projection_step_rhs_norm()
               << ", "
               << heat_equation.get_rhs_norm()
               << ")\n";
  // Outputs the benchmark's data to the terminal
  *this->pcout << benchmark_requests << std::endl << std::endl;

  log_file << time_stepping.get_step_number() << ","
           << time_stepping.get_current_time() << ","
           << time_stepping.get_next_step_size() << ","
           << cfl_number << ","
           << navier_stokes.get_diffusion_step_rhs_norm() << ","
           << navier_stokes.get_projection_step_rhs_norm() << ","
           << heat_equation.get_rhs_norm()
           << std::endl;

  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    if (!std::filesystem::exists(this->prm.graphical_output_directory))
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

    const std::filesystem::path path{this->prm.graphical_output_directory};

    std::filesystem::path filename = path / "benchmark_data.txt";

    try
    {
      std::ofstream fstream(filename.string());
      benchmark_requests.write_text(fstream);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the creation of the output file: "
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
      std::cerr << "Unknown exception in the creation of the output file!"
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
  }
}

template <int dim>
void Christensen<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  // Explicit declaration of the velocity as a vector
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  // Loading the DataOut instance with the solution vectors
  DataOut<dim>        data_out;
  data_out.add_data_vector(velocity->get_dof_handler(),
                           velocity->solution,
                           names,
                           component_interpretation);
  data_out.add_data_vector(pressure->get_dof_handler(),
                           pressure->solution,
                           "Pressure");
  data_out.add_data_vector(temperature->get_dof_handler(),
                           temperature->solution,
                           "Temperature");

  // To properly showcase the velocity field (whose k-th order finite
  // elements are one order higher than those of the pressure field),
  // the k-th order elements are interpolated to four (k-1)-th order
  // elements. In other words, the triangulation visualized in the
  // *.pvtu file is one globl refinement finer than the actual
  // triangulation.

  data_out.build_patches(*this->mapping,
                         velocity->fe_degree(),
                         DataOut<dim>::curved_inner_cells);


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
void Christensen<dim>::update_solution_vectors()
{
  // Sets the solution vectors at t^{k-j} to those at t^{k-j+1}
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
  temperature->update_solution_vectors();
}

template <int dim>
void Christensen<dim>::run()
{
  // Outputs the initial conditions
  velocity->solution    = velocity->old_solution;
  pressure->solution    = pressure->old_solution;
  temperature->solution = temperature->old_solution;
  output();

  const unsigned int n_steps = this->prm.time_discretization_parameters.n_maximum_steps;

  while (time_stepping.get_current_time() < time_stepping.get_end_time() &&
         (n_steps > 0? time_stepping.get_step_number() < n_steps: true))
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(time_stepping, cfl_number));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    heat_equation.solve();
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();
    *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                 << std::endl;

    // Performs post-processing
    if ((time_stepping.get_step_number() %
          this->prm.terminal_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()))
      postprocessing();
    /*
    // Performs coarsening and refining of the triangulation
    if (time_stepping.get_step_number() %
        this->prm.spatial_discretization_parameters.adaptive_mesh_refinement_frequency == 0)
      this->adaptive_mesh_refinement();*/

    // Graphical output of the solution vectors
    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()))
      output();
  }










}

} // namespace ChristensenBenchmark

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace ChristensenBenchmark;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 2);

      RunTimeParameters::ProblemParameters parameter_set("Christensen.prm");

      switch (parameter_set.dim)
      {
        case 2:
        {
          Christensen<2> simulation(parameter_set);
          simulation.run();
          break;
        }
        case 3:
        {
          Christensen<3> simulation(parameter_set);
          simulation.run();
          break;
        }
        default:
          AssertThrow(false, ExcNotImplemented());
          break;
      }
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
