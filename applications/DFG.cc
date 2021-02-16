/*!
 *@file DFG
 *@brief The source file for solving the DFG benchmark.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/hydrodynamic_problem.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <string>

namespace RMHD
{

using namespace dealii;

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
class DFG : public HydrodynamicProblem<dim>
{
public:
  DFG(const RunTimeParameters::HydrodynamicProblemParameters &parameters);

  void run();

private:
  BenchmarkData::DFGBechmarkRequest<dim>        benchmark_request;

  virtual void make_grid() override;

  virtual void postprocess_solution() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;
};

template <int dim>
DFG<dim>::DFG(const RunTimeParameters::HydrodynamicProblemParameters &parameters)
:
HydrodynamicProblem<dim>(parameters),
benchmark_request()
{}

template <int dim>
void DFG<dim>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  /*
   *
   * SG: Why are we reading the grid from the filesystem? There is a method to
   * do this! GridGenerator::channel_with_cylinder
   *
   */
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);

  {
    std::string   filename = "dfg.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  const PolarManifold<dim> inner_boundary;
  this->triangulation.set_all_manifold_ids_on_boundary(2, 1);
  this->triangulation.set_manifold(1, inner_boundary);

  *this->pcout << "Number of active cells                = "
               << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void DFG<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  const double current_time = this->time_stepping.get_current_time();

  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Boundary conditions are not setup at the start time."));

  this->velocity->boundary_conditions.set_dirichlet_bcs
  (0,
   std::make_shared<EquationData::DFG::VelocityInflowBoundaryCondition<dim>>(current_time));

  this->velocity->boundary_conditions.set_dirichlet_bcs(2);
  this->velocity->boundary_conditions.set_dirichlet_bcs(3);

  this->pressure->boundary_conditions.set_dirichlet_bcs(1);

  this->velocity->apply_boundary_conditions();
  this->pressure->apply_boundary_conditions();
}

template <int dim>
void DFG<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  using namespace EquationData::DFG;

  const double current_time = this->time_stepping.get_current_time();

  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Initial conditions are not setup at the start time."));

  const VelocityInitialCondition<dim>  velocity_initial_condition(dim);
  this->project_function(velocity_initial_condition,
                         this->velocity,
                         this->velocity->old_solution);
  this->velocity->solution = this->velocity->old_solution;

  const PressureInitialCondition<dim> pressure_initial_condition;
  this->project_function(pressure_initial_condition,
                         this->pressure,
                         this->pressure->old_solution);
  this->pressure->solution = this->pressure->old_solution;
}

template <int dim>
void DFG<dim>::postprocess_solution()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  benchmark_request.compute_pressure_difference(this->pressure);
  benchmark_request.compute_drag_and_lift_coefficients(this->velocity,
                                                       this->pressure);
  benchmark_request.print_step_data(this->time_stepping);
  benchmark_request.update_table(this->time_stepping);
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      RunTimeParameters::HydrodynamicProblemParameters parameter_set("DFG.prm");

      DFG<2> dfg(parameter_set);

      HydrodynamicProblem<2>* hydro_problem = &dfg;
      hydro_problem->run();

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
