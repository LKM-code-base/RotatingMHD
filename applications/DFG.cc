/*!
 *@file DFG
 *@brief The source file for solving the DFG benchmark.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include <string>

namespace RMHD
{
  using namespace dealii;
/*!
 * @class DFG
 * @brief This class solves the DFG flow around cylinder benchmark 2D-2, 
 * time-periodic case Re=100
 * @details The DFG benchmark models the flow inside a pipe with a 
 * circular cylinder as obstacle (See Fig ). The field equations for the 
 * problem are given by
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 * \dfrac{ \partial \mathbf{v}}{ \partial t} + \mathbf{v} \cdot 
 * (\nabla \otimes \mathbf{v})&=  - \dfrac{1}{\rho}\nabla p + \nu 
 * \Delta \mathbf{v} + \mathbf{b}, &\forall (\mathbf{x}, t) \in \Omega 
 * \times \left[0, T \right] \\
 * \nabla \cdot \mathbf{v}&= 0, &\forall (\mathbf{x}, t) \in \Omega 
 * \times  \left[0, T \right]
 * \end{aligned}
 * \end{equation*}
 * \f]
 *   with boundary conditions
 * \f[
 *   \begin{equation*}
 * \begin{aligned}
 *   \mathbf{v} &= \mathbf{v}_\textrm{in},  &\forall(\mathbf{x}, t) 
 *    &\in \Gamma_0 \times [0, T] \\
 *   \mathbf{v} &= \mathbf{0}, &\forall(\mathbf{x}, t) 
 *    &\in \Gamma_2 \times [0,T]\\
 *   \mathbf{v} &= \mathbf{0}, &\forall(\mathbf{x}, t) 
 *    &\in \Gamma_3 \times [0, T] \\
 *    [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] \cdot \mathbf{n} 
 *    &=0, &\forall(\mathbf{x}, t) &\in \Gamma_1 \times [0, T] \\
 *   \end{aligned}
 *   \end{equation*}
 * \f]
 *   where the inflow velocity is given by
 * \f[
 *   \begin{equation*}
 *     \mathbf{v}_\textrm{in} = 4v_\textrm{max}\dfrac{y(H-y)}
 *      {H^2}\mathbf{e}_\textrm{x},
 *     \qquad
 *     v_\textrm{max} =1.5
 *   \end{equation*}
 * \f]
 * Furthermore is \f$\rho = 1.0\f$ and \f$\nu = 0.001\f$. Taking the mean
 * velocity \f$\bar{v} = \tfrac{2}{3} \mathbf{v}_\textrm{in}|_{(0,0.5H)}
 * \cdot \mathbf{e}_\textrm{x} =  1.0\f$ and the 
 * cylinder diameter \f$ D=0.1\f$ as reference velocity and length, the 
 * Reynolds number is
 * \f[
 * \begin{equation*}
 *	\textrm{Re} = \dfrac{\bar{v}D}{\nu} = 100.
 * \end{equation*}
 * \f]
 * The benchmarks to reach are the pressure difference
 * \f[
 * \begin{equation*}
 *	\Delta p = p|_{(0.15,0.2)} - p|_{(0.25,0.2)}
 * \end{equation*}
 * \f]
 * the lift and drag coefficients,
 * \f[
 * \begin{equation*}
 *	c_\textrm{drag} = \dfrac{2}{\rho\bar{v}^2D} f_\textrm{drag}
 *	\qquad
 *	c_\textrm{lift} = \dfrac{2}{\rho\bar{v}^2D} f_\textrm{lift}
 * \end{equation*}
 * \f]
 * where the drag and lift forces are given by
 * \f[
 * \begin{equation*}
 *	 \boldsymbol{\mathfrak{f}} = \mathfrak{f}_\textrm{drag}  
 *   \mathbf{e}_\textrm{x} +  \mathfrak{f}_\textrm{lift} 
 *   \mathbf{e}_\textrm{y} =
 *	 \int_{\Gamma_3} [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] 
 *   \cdot \mathbf{n} \mathrm{d} \ell
 * \end{equation*}
 * \f]
 * Defining a cycle by the time between two consecutive 
 * \f$\max(c_\textrm{lift})\f$ values and denoting it by 
 * \f$[t_\textrm{start}, t_\textrm{end}]\f$, where the frequency \f$ f =
 * \tfrac{1}{(t_\textrm{end} - t_\textrm{start})}\f$, Strouhal number is 
 * given by
 * \f[
 * \begin{equation*}
 *	\textrm{St} = \dfrac{fD}{\bar{v}}
 * \end{equation*}
 * \f]
 * Additionally the minimum, average and amplitude of the 
 * coefficients are to be computed for whole cycle.
 * The NavierStokesProjection classs is based on the dimensionless 
 * form of the Navier-Stokes equations. Therefore the benchmark has to 
 * be reformulated into its dimensionless form:
 * \f[
 * \begin{equation*}
 * 	\begin{aligned}
 * 		\frac{\partial \mathbf{\tilde{v}}}{\partial \tilde{t}} + 
 *    \mathbf{\tilde{v}} \cdot
 *    (\tilde{\nabla} \otimes \mathbf{\tilde{v}})&=  -\tilde{\nabla} 
 *    \tilde{p}+ \dfrac{1}{\mathrm{Re}}\tilde{\Delta} \mathbf{\tilde{v}} 
 *    + \mathbf{\tilde{b}}, &\forall (\mathbf{\tilde{x}}, \tilde{t}) 
 *    \in \tilde{\Omega} \times [0, \tilde{T} ] \\
 * 		\tilde{\nabla} \cdot \mathbf{\tilde{v}}&= 0, &\forall 
 *  (\mathbf{\tilde{x}}, \tilde{t}) \in\tilde{ \Omega} \times  
 *  [0, \tilde{T} ]
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 * with boundary conditions
 * \f[
 * \begin{equation*}
 * 	\begin{aligned}
 * 		\mathbf{\tilde{v}} &= \mathbf{\tilde{v}}_\textrm{in},  
 *    &\forall(\mathbf{x}, t) &\in \Gamma_0 \times [0, T] \\
 * 		\mathbf{\tilde{v}} &= \mathbf{0}, &\forall(\mathbf{x}, t) 
 *    &\in \Gamma_2 \times [0,T]\\
 * 		\mathbf{\tilde{v}} &= \mathbf{0}, &\forall(\mathbf{x}, t) 
 *    &\in \Gamma_3 \times [0, T] \\
 * 		[-\tilde{p}\mathbf{I} + (\rho\textrm{Re})^{-1}\tilde{ \nabla} 
 *    \otimes \mathbf{\tilde{v}}] \cdot \mathbf{n} &=0, 
 *    &\forall(\mathbf{x}, t) &\in \Gamma_1 \times [0, T] \\
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 * The inflow velocity is given by
 * \f[
 * \begin{equation*}
 * 	\mathbf{\tilde{v}}_\textrm{in} = 4\dfrac{v_\textrm{max}}{\bar{v}}
 *  \dfrac{\tilde{y}(\tilde{H}-\tilde{y})}{\tilde{H}^2}
 *  \mathbf{e}_\textrm{x},
 * 	\qquad
 * 	v_\textrm{max} = 1.5
 * \end{equation*}
 * \f]
 * The computed variables need to be scaled back in order for them to 
 * be compared with those from the benchmark. The pressure difference
 * is scaled by
 * \f[
 * \begin{equation*}
 * 	\Delta p = \hat{p} \Delta \tilde{p},
 * \end{equation*}
 * \f]
 * but since \f$\hat{p} =1\f$, they are interchangeable. 
 * The dimensionless form of the force is given by
 * \f[
 * \begin{equation*}
 * 	\boldsymbol{\tilde{\mathfrak{f}}} = \dfrac{\hat{p}D}
 *  {\hat{\mathfrak{f}}} \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} 
 *  + (\rho\textrm{Re})^{-1} \tilde{\nabla} \otimes \mathbf{\tilde{v}}] 
 *  \cdot \mathbf{n} \mathrm{d} \tilde{\ell}
 * \end{equation*}
 * \f]
 * from which it is easy to see that the coefficients are equivalent to
 * \f[
 * \begin{equation*}
 * 	c_\textrm{drag} = 2 \hat{\mathfrak{f}}_\textrm{drag}
 * 	\qquad
 * 	c_\textrm{lift} = 2 \hat{\mathfrak{f}}_\textrm{lift}.
 * \end{equation*}
 * \f]
 * The frequency is scaled back with
 * \f[
 * \begin{equation*}
 * 	f = \hat{t}^{-1} \tilde{f}
 * \end{equation*}
 * \f]
 * from which it is easy to see that 
 * \f[
 * \begin{equation*}
 * 	\textrm{St} = \tilde{f}
 * \end{equation*}
 * \f]
 * @attention Due to the formulation of the NavierStokesProjection class
 * the force around the cylinder is computed by @ref BenchmarkData::DFG 
 * as
 * \f[
 * \begin{equation*}
 * 	\boldsymbol{\tilde{\mathfrak{f}}} = 
 *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} 
 *  + (\rho\textrm{Re})^{-1} (\tilde{\nabla} \otimes \mathbf{\tilde{v}} 
 *  + \mathbf{\tilde{v}} \otimes \tilde{\nabla})]
 *  \cdot \mathbf{n} \mathrm{d} \tilde{\ell}
 * \end{equation*}
 * \f]
 */  
template <int dim>
class DFG : public Problem<dim>
{
public:
  DFG(const RunTimeParameters::ParameterSet &parameters);
  void run(const bool         flag_verbose_output           = false,
           const unsigned int terminal_output_periodicity   = 10,
           const unsigned int graphical_output_periodicity  = 10);
private:

  std::vector<types::boundary_id>             boundary_ids;

  Entities::VectorEntity<dim>                 velocity;

  Entities::ScalarEntity<dim>                 pressure;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;

  BenchmarkData::DFG<dim>                     dfg_benchmark;

  EquationData::DFG::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;

  EquationData::DFG::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;

  EquationData::DFG::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;

  void make_grid();

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_solution_vectors();
};

template <int dim>
DFG<dim>::DFG(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(),
velocity(parameters.p_fe_degree + 1, this->triangulation),
pressure(parameters.p_fe_degree, this->triangulation),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters, velocity, pressure, time_stepping),
dfg_benchmark(),
inflow_boundary_condition(parameters.time_stepping_parameters.start_time),
velocity_initial_conditions(parameters.time_stepping_parameters.start_time),
pressure_initial_conditions(parameters.time_stepping_parameters.start_time)
{
  make_grid();
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup();
  initialize();
}

template <int dim>
void DFG<dim>::
make_grid()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);

  {
    std::string   filename = "dfg.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  boundary_ids = this->triangulation.get_boundary_ids();

  const PolarManifold<dim> inner_boundary;
  this->triangulation.set_all_manifold_ids_on_boundary(2, 1);
  this->triangulation.set_manifold(1, inner_boundary);

  this->pcout << "Number of active cells                = "
              << this->triangulation.n_active_cells() << std::endl;
}

template <int dim>
void DFG<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  
  this->pcout << "Number of velocity degrees of freedom = "
              << velocity.dof_handler.n_dofs()
              << std::endl
              << "Number of pressure degrees of freedom = "
              << pressure.dof_handler.n_dofs()
              << std::endl;
}

template <int dim>
void DFG<dim>::setup_constraints()
{
  velocity.constraints.clear();
  velocity.constraints.reinit(velocity.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(velocity.dof_handler,
                                          velocity.constraints);
  for (const auto &boundary_id : boundary_ids)
    switch (boundary_id)
    {
      case 0:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    inflow_boundary_condition,
                                    velocity.constraints);
        break;
      case 1:
        break;
      case 2:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    Functions::ZeroFunction<dim>(dim),
                                    velocity.constraints);
        break;
      case 3:
        VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
                                    Functions::ZeroFunction<dim>(dim),
                                    velocity.constraints);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  velocity.constraints.close();

  pressure.constraints.clear();
  pressure.constraints.reinit(pressure.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(pressure.dof_handler,
                                          pressure.constraints);
  VectorTools::interpolate_boundary_values(
                                      pressure.dof_handler,
                                      1,
                                      Functions::ZeroFunction<dim>(),
                                      pressure.constraints);
  pressure.constraints.close();
}

template <int dim>
void DFG<dim>::initialize()
{
  this->set_initial_conditions(velocity, 
                               velocity_initial_conditions, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_conditions, 
                               time_stepping);
  
  navier_stokes.initialize();
}

template <int dim>
void DFG<dim>::postprocessing(const bool flag_point_evaluation)
{
  if (flag_point_evaluation)
  {
    dfg_benchmark.compute_pressure_difference(pressure);
    dfg_benchmark.compute_drag_and_lift_forces_and_coefficients(
                                                            velocity,
                                                            pressure);
    dfg_benchmark.print_step_data(time_stepping);
    dfg_benchmark.update_table(time_stepping);
  }
}

template <int dim>
void DFG<dim>::output()
{
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(velocity.dof_handler,
                           velocity.solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure.solution, 
                           "Pressure");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

template <int dim>
void DFG<dim>::update_solution_vectors()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
  navier_stokes.update_internal_entities();
}

template <int dim>
void DFG<dim>::run(const bool          /* flag_verbose_output */,
                   const unsigned int  terminal_output_periodicity,
                   const unsigned int  graphical_output_periodicity)
{
  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  pcout << "Solving until t = 35..." << std::endl;
  while (time_stepping.get_current_time() <= 35.0)
  {
    // snapshot stage
    time_stepping.set_desired_next_step_size(
                          navier_stokes.compute_next_time_step());

    // update stage
    time_stepping.update_coefficients();

    navier_stokes.solve(time_stepping.get_step_number());

    // snapshot stage
    postprocessing((time_stepping.get_step_number() %
                    terminal_output_periodicity == 0) ||
                    (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    update_solution_vectors();

    time_stepping.advance_time();
  }

  pcout << "Restarting..." << std::endl;
  time_stepping.restart();
  velocity.old_old_solution = velocity.solution;
  navier_stokes.reinit_internal_entities();
  navier_stokes.initialize();
  velocity.solution = velocity.old_solution;
  pressure.solution = pressure.old_solution;
  output();

  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();

  pcout << "Solving until t = " << time_stepping.get_end_time()
        << "..." << std::endl;

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    time_stepping.set_desired_next_step_size(
                              navier_stokes.compute_next_time_step());

    time_stepping.update_coefficients();

    navier_stokes.solve(time_stepping.get_step_number());

    // snapshot stage
    postprocessing((time_stepping.get_step_number() %
                    terminal_output_periodicity == 0) ||
                    (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          graphical_output_periodicity == 0) ||
        (time_stepping.get_next_time() == 
          time_stepping.get_end_time()))
      output();

    update_solution_vectors();

    time_stepping.advance_time();
  }

  dfg_benchmark.write_table_to_file("dfg_benchmark.tex");

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

      RunTimeParameters::ParameterSet parameter_set("DFG.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      DFG<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_verbose_output, 
                     parameter_set.terminal_output_interval,
                     parameter_set.graphical_output_interval);
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
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}
