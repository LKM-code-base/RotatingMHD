/*!
 *@file DFG
 *@brief The source file for solving the DFG benchmark.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/navier_stokes_parameters.h>
#include <iostream>
#include <string>

namespace RMHD
{

using namespace dealii;

using namespace EquationData::DFG;

using namespace BenchmarkData;

using namespace RunTimeParameters;

/*!
 * @class DFG
 *
 * @brief This class solves the DFG benchmark 2D-2 of the flow around cylinder
 * which is placed in a channel. This is the time-periodic case with a
 * Reynolds number of \f$ \Reynolds=100 \f$.
 *
 * @details The DFG benchmark considers the flow of fluid inside a channel with
 * a cylinder as an obstacle (See Fig. ).
 *
 * @todo A figure is missing here.
 *
 *
 * **Defintion of the benchmark**
 *
 * The field equations of the problem are given by the incompressible
 * Navier-Stokes equations
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \pd{\bs{v}}{t} + \bs{v} \cdot
 *    (\nabla \otimes \bs{v})&=  - \dfrac{1}{\rho_0}\nabla p + \nu
 *    \Delta \bs{v} + \bs{b}, &
 *    \forall (\bs{x}, t) \in \Omega \times \left[0, T \right]\,, \\
 *    \nabla \cdot \bs{v}&= 0, &
 *    \forall (\bs{x}, t) \in \Omega \times  \left[0, T \right]
 * \end{aligned}
 * \end{equation*}
 * \f]
 *
 * with the following boundary conditions
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \bs{v} &= \bs{v}_\text{in}\,,  &
 *    \forall(\bs{x}, t) &\in \Gamma_0 \times [0, T]\,, \\
 *    \bs{v} &= \bs{0}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_2 \times [0,T]\,, \\
 *    \bs{v} &= \bs{0}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_3 \times [0, T]\,, \\
 *    \bs{n} \cdot [-p\bs{1} + \mu \nabla \otimes \bs{v}] &=0\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_1 \times [0, T]\,.
 * \end{aligned}
 * \end{equation*}
 * \f]
 *
 * The velocity profile at the inlet of the channel is given by a quadratic
 * function
 *
 * \f[
 * \begin{equation*}
 *    \bs{v}_\text{in} = v_0\left(\frac{2}{H}\right)^2 y(H-y) \ex \quad
 *    \text{with} \quad v_0 =\frac{3}{2}\frac{\mathrm{m}}{\mathrm{s}}\,.
 * \end{equation*}
 * \f]
 *
 * Furthermore, the density is given by \f$\rho_0 = 1\,\mathrm{kg}\,\mathrm{m}^{-3}\f$
 * and the kinematic viscosity by \f$\nu = 0{.}001\,\mathrm{m}^2\,\mathrm{s}^{-1}\f$.
 * In order to compute the Reynolds number of the problem, a reference velocity
 * and a reference length need to be defined. The average of the velocity
 * at the inlet is chosen as the reference velocity \f$ v_\mathrm{ref} \f$,
 * *i. e.*,
 *
 * \f[
 * \begin{equation*}
 *    v_\mathrm{ref} = \frac{1}{H}\int\limits_0^H \bs{v}_\text{in}(y)\cdot\ex
 *    \dint{y}=\frac{2}{3}v_0=1\frac{\mathrm{m}}{\mathrm{s}}
 * \end{equation*}
 * \f]
 *
 *
 * Moreover, the diameter of the cylinder is chosen as the reference length,
 * *i. e.*, \f$ \ell_\mathrm{ref}=D=0{.}1\,\mathrm{m}\f$. These two choice yield
 * the following definition and value of the Reynolds number of the problem
 * \f[
 * \begin{equation*}
 *    \Reynolds = \frac{v_\mathrm{ref} D}{\nu} = 100\,.
 * \end{equation*}
 * \f]
 *
 * **Benchmark requests**
 *
 * The DFG benchmark requires to specify the pressure difference between two
 * points located at the front and the rear of the cylinder, respectively. These
 * points are located at the following positions
 * \f$ \bs{x}_\text{front} = 0{.}15\,\mathrm{m}\, \ex + 0.20\,\mathrm{m}\, \ey \f$
 * and
 * \f$ \bs{x}_\text{rear} = 0{.}25\,\mathrm{m}\, \ex + 0.20\,\mathrm{m}\, \ey \f$,
 * respectively. Thus, the pressure difference \f$ \Delta p\f$ is computed
 * according to
 *
 * \f[
 * \begin{equation*}
 *    \Delta p = p|_{\bs{x}_\text{front}} - p|_{\bs{x}_\text{rear}}\,.
 * \end{equation*}
 * \f]
 *
 * Additionally, the DFG benchmark request the value of the lift and drag
 * coefficients, which are defined as
 *
 * \f[
 * \begin{equation*}
 *    c_\text{drag} = \dfrac{2}{\rho v_\mathrm{ref}^2 D} F_\text{drag}\,,\qquad
 *    c_\text{lift} = \dfrac{2}{\rho v_\mathrm{ref}^2 D} F_\text{lift}\,.
 * \end{equation*}
 * \f]
 *
 * Here, \f$ F_\text{drag} \f$ and \f$ F_\text{lift} \f$ refer to the horizontal
 * and vertical components of the resulting force acting on the cylinder. The
 * resulting force is computed according to
 *
 * \f[
 * \begin{equation*}
 *	 \bs{F} = F_\text{drag}\ex +  F_\text{lift}\ey
 *	 = \int\limits_{\Gamma_3} \bs{\sigma}\cdot \bs{n} \dint{A}
 *   = \int\limits_{\Gamma_3} \Big(-p \bs{1} + \mu \big(\nabla \otimes \bs{v}
 *   + \bs{v} \otimes \nabla \big) \Big) \cdot \bs{n} \dint{A}\,.
 * \end{equation*}
 * \f]
 *
 * Another quantity requested is the Strouhal number related to the periodic
 * oscillation of the flow. This dimesionless number is defined as
 *
 * \f[
 * \begin{equation*}
 *   \Strouhal = \dfrac{f D}{v_\mathrm{ref}}\,,
 * \end{equation*}
 * \f]
 *
 * where \f$ f \f$ denotes the frequency of the oscillation. This frequency is
 * the reciprocal of the time period \f$ T \f$ of the oscillation, *i. e.*,
 * \f$ f = 1/ T \f$. The period of the oscillation is computed by considering
 * two consecutive maximum values of the lift and drag coefficients. Taking the
 * time difference between these yields the time period of the oscillation.
 * Additionally the minimum, the average and the amplitude of the lift and drag
 * coefficients are also computed.
 *
 * **Dimensionless formulation of the benchmark**
 *
 * The @ref NavierStokesProjection class is based on the dimensionless
 * form of the Navier-Stokes equations. Therefore, the benchmark has to
 * be reformulated into its dimensionless form as follows
 *
 * \f[
 * \begin{equation*}
 * 	\begin{aligned}
 * 		\pd{\tilde{\bs{v}}}{\tilde{t}} +
 *    \tilde{\bs{v}} \cdot (\tilde{\nabla} \otimes \tilde{\bs{v}})
 *    &= -\tilde{\nabla}\tilde{p} +
 *    \frac{1}{\Reynolds} \nabla^2 \tilde{\bs{v}} + \tilde{\bs{b}}\, &
 *    \forall (\tilde{\bs{x}}, \tilde{t})
 *    \in \tilde{\Omega} \times [0, \tilde{T} ] \\
 * 		\tilde{\nabla} \cdot \tilde{\bs{v}}&= 0, &
 * 		\forall (\tilde{\bs{x}}, \tilde{t}) \in\tilde{\Omega} \times [0, \tilde{T} ]
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 *
 * with the following boundary conditions
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \tilde{\bs{v}} &= \tilde{\bs{v}}_\text{in}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_0 \times [0, T] \\
 *    \tilde{\bs{v}} &= \bs{0}\,, &
 * 		\forall(\bs{x}, t) &\in \Gamma_2 \times [0,T]\\
 * 	  \tilde{\bs{v}} &= \bs{0}, &
 * 		\forall(\bs{x}, t) &\in \Gamma_3 \times [0, T] \\
 * 	  \bs{n}\cdot [-\tilde{p}\bs{1} + \tfrac{1}{\Reynolds}\tilde{\nabla}
 *    \otimes \tilde{\bs{v}}] &=0, &
 *    \forall(\bs{x}, t) &\in \Gamma_1 \times [0, T] \\
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 *
 * The dimensionless velocity profile at the inlet of channel is then given by
 *
 * \f[
 * \begin{equation*}
 *    \tilde{\bs{v}}_\text{in} = \tilde{v}_0 \left(\frac{2}{\tilde{H}}\right)^2
 *    \tilde{y}(\tilde{H}-\tilde{y}) \ex \,, \quad
 *    \text{with}\quad
 * 	  \tilde{v}_0=\frac{3}{2}\,.
 * \end{equation*}
 * \f]
 *
 * The benchmark requests are computed in dimensionless form. In order to compare
 * with those of the DFG benchmark, they need to be scaled back. The reference
 * pressure is given by the dynamic pressure, *i. e.*,
 * \f$ p_\mathrm{ref}=\rho_0 v_\mathrm{ref}^2
 * = 1\,\mathrm{kg}\,\mathrm{m}^{-1}\,\mathrm{s}^{-2} \f$. Thus, the numerical
 * values of the dimensionless and dimensioned pressure are equal. Of course,
 * the same also applies for the pressure differences requested by the DFG
 * benchmark. They are interchangeable.
 *
 * The reference value of the force is \f$ F_\mathrm{ref}=\rho_0
 * v_\mathrm{ref}^2 D^2\f$. Thus the dimensionless resulting force is given by
 *
 * \f[
 * \begin{equation*}
 *    \tilde{\bs{F}} = \frac{1}{\rho_0 v_\mathrm{ref}^2 D^2}
 * 	  \int_{\tilde{\Gamma}_3} \Big(-p\bs{1}
 *    + \nu \big( \nabla \otimes \bs{v} + \bs{v} \otimes \nabla\big)
 *    \Big) \cdot \bs{n} \dint{A}
 *    = \int_{\tilde{\Gamma}_3} \Big(-\tilde{p}\bs{1}
 *    + \tfrac{1}{\Reynolds} \big( \tilde{\nabla} \otimes \tilde{\bs{v}} +
 *    \tilde{\bs{v}} \otimes \tilde{\nabla} \big) \Big) \cdot \bs{n} \dint{\tilde{A}}\,.
 * \end{equation*}
 * \f]
 *
 * With this formula for the dimensionless force, it is easy to see that the
 * coefficients are given by
 *
 * \f[
 * \begin{equation*}
 *    c_\text{drag} = 2 \tilde{F}_\text{drag}\,,\qquad
 *    c_\text{lift} = 2 \tilde{F}_\text{lift}\,.
 * \end{equation*}
 * \f]
 *
 * The frequency is scaled back with the reference time \f$ t_\mathrm{ref} \f$.
 * Hence,
 *
 * \f[
 * \begin{equation*}
 * 	  f = \frac{1}{t_\mathrm{ref}} \tilde{f}
 * \end{equation*}
 * \f]
 *
 * Since the reference time is given by \f$ t_\mathrm{ref} = D / v_\mathrm{ref}\f$,
 * the Strouhal number may also expressed in terms of the dimesionless frequency
 * as follows
 *
 * \f[
 * \begin{equation*}
 *    \Strouhal = \tilde{f}\,.s
 * \end{equation*}
 * \f]
 *
 */  
template <int dim>
class DFG : public Problem<dim>
{
public:
  DFG(const NavierStokesProblemParameters &prm);

  void run();

private:

  const NavierStokesProblemParameters          &params;

  std::vector<types::boundary_id>               boundary_ids;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  DFGBechmarkRequest<dim>                       benchmark_request;

  VelocityInitialCondition<dim>                 velocity_initial_condition;

  PressureInitialCondition<dim>                 pressure_initial_condition;

  void make_grid();

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_solution_vectors();
};

template <int dim>
DFG<dim>::DFG(const NavierStokesProblemParameters &prm)
:
Problem<dim>(prm),
params(prm),
velocity(std::make_shared<Entities::VectorEntity<dim>>
         (params.fe_degree + 1,
          this->triangulation)),
pressure(std::make_shared<Entities::ScalarEntity<dim>>
         (params.fe_degree,
          this->triangulation)),
time_stepping(params),
navier_stokes(params.navier_stokes_discretization,
              velocity,
              pressure,
              time_stepping,
              this->pcout,
              this->computing_timer),
benchmark_request(),
velocity_initial_condition(dim),
pressure_initial_condition()
{
  make_grid();

  setup_dofs();

  setup_constraints();

  velocity->reinit();
  pressure->reinit();

  initialize();
}

template <int dim>
void DFG<dim>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::channel_with_cylinder(this->triangulation);


  boundary_ids = this->triangulation.get_boundary_ids();

//  const PolarManifold<dim> inner_boundary;
//  this->triangulation.set_all_manifold_ids_on_boundary(2, 1);
//  this->triangulation.set_manifold(1, inner_boundary);

  *(this->pcout)  << "Number of active cells                = "
                  << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void DFG<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();
  
  *(this->pcout)  << "Number of velocity degrees of freedom = "
                  << (velocity->dof_handler)->n_dofs()
                  << std::endl
                  << "Number of pressure degrees of freedom = "
                  << (pressure->dof_handler)->n_dofs()
                  << std::endl;
}

template <int dim>
void DFG<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->boundary_conditions.set_dirichlet_bcs
  (0,
   std::make_shared<VelocityInflowBoundaryCondition<dim>>(params.start_time));

  velocity->boundary_conditions.set_dirichlet_bcs(2);
  velocity->boundary_conditions.set_dirichlet_bcs(3);

  pressure->boundary_conditions.set_dirichlet_bcs(1);

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void DFG<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity, 
                               velocity_initial_condition, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_condition, 
                               time_stepping);
  
  navier_stokes.initialize();
}

template <int dim>
void DFG<dim>::postprocessing(const bool flag_point_evaluation)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  if (flag_point_evaluation)
  {
    benchmark_request.compute_pressure_difference(pressure);
    benchmark_request.compute_drag_and_lift_coefficients(velocity,
                                                         pressure);
    benchmark_request.print_step_data(time_stepping);
    benchmark_request.update_table(time_stepping);
  }
}

template <int dim>
void DFG<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(*(velocity->dof_handler),
                           velocity->solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(*(pressure->dof_handler), 
                           pressure->solution, 
                           "Pressure");
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
void DFG<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void DFG<dim>::run()
{
  // Advances the time to t^{k-1}, either t^0 or t^1
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  *(this->pcout) << "Solving until t = 350..." << std::endl;
  while (time_stepping.get_current_time() <= 350.0)
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}    time_stepping.set_desired_next_step_size(
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_current_time()
    postprocessing((time_stepping.get_step_number() % params.terminal_output_frequency == 0) ||
                   (time_stepping.get_current_time() == time_stepping.get_end_time()));
  }

  *(this->pcout) << "Restarting..." << std::endl;
  time_stepping.restart();
  velocity->old_old_solution = velocity->solution;
  navier_stokes.reset_phi();
  navier_stokes.initialize();
  velocity->solution = velocity->old_solution;
  pressure->solution = pressure->old_solution;
  output();

  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  *(this->pcout)  << "Solving until t = " << time_stepping.get_end_time()
                  << "..." << std::endl;

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the time step, i.e sets the value of t^{k}    time_stepping.set_desired_next_step_size(
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(
        time_stepping, 
        navier_stokes.get_cfl_number()));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_current_time()
    postprocessing((time_stepping.get_step_number() % params.terminal_output_frequency == 0) ||
                   (time_stepping.get_current_time() == time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() % params.graphical_output_frequency == 0) ||
        (time_stepping.get_next_time() == time_stepping.get_end_time()))
      output();
  }

  benchmark_request.write_table_to_file("dfg_benchmark.tex");

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

      NavierStokesProblemParameters parameter_set("DFG.prm");

      deallog.depth_console(parameter_set.verbose ? 2 : 0);

      if (parameter_set.dim == 2)
      {
        DFG<2> simulation(parameter_set);
        simulation.run();
      }
      else if (parameter_set.dim == 3)
      {
        DFG<3> simulation(parameter_set);
        simulation.run();
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
  std::cout << "----------------------------------------------------"
            << std::endl;
  return 0;
}
