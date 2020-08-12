/*
 * data_storage.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */


#include <rotatingMHD/run_time_parameters.h>

#include <fstream>

namespace Step35
{

  using namespace dealii;

namespace RunTimeParameters
{
// In the constructor of this class we declare all the parameters. The
// details of how this works have been discussed elsewhere, for example in
// step-19 and step-29.
ParameterSet::ParameterSet()
  : projection_method(ProjectionMethod::rotational),
    dt(5e-4),
    t_0(0.0),
    T(1.0),
    vsimex_input_gamma(1.0),
    vsimex_input_c(0.0),
    Re(1.0),
    n_global_refinements(0),
    p_fe_degree(1),
    solver_max_iterations(1000),
    solver_krylov_size(30),
    solver_off_diagonals(60),
    solver_update_preconditioner(15),
    solver_tolerance(1e-12),
    solver_diag_strength(0.01),
    flag_verbose_output(true),
    flag_adaptive_time_step(true),
    flag_DFG_benchmark(false),
    graphical_output_interval(15)
{
  prm.declare_entry("projection_method",
                    "rotational",
                    Patterns::Selection("rotational|standard"),
                    " Projection method to implement. ");
  prm.enter_subsection("Physical parameters");
  {
    prm.declare_entry("Re",
                      "1.",
                      Patterns::Double(0.),
                      " Reynolds number. ");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time discretization parameters");
  {
    prm.declare_entry("dt",
                      "5e-4",
                      Patterns::Double(0.),
                      " Time step size. ");
    prm.declare_entry("t_0",
                      "0.",
                      Patterns::Double(0.),
                      " The initial time of the simulation. ");
    prm.declare_entry("T",
                      "1.",
                      Patterns::Double(0.),
                      " The final time of the simulation. ");
    prm.declare_entry("vsimex_input_gamma",
                      "1.",
                      Patterns::Double(0.),
                      " Input parameter gamma of the VSIMEX method. ");
    prm.declare_entry("vsimex_input_c",
                      "0.",
                      Patterns::Double(0.),
                      " Input parameter c of the VSIMEX method. ");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space discretization parameters");
  {
    prm.declare_entry("n_global_refinements",
                      "0",
                      Patterns::Integer(0, 15),
                      " Number of global refinements done on the"
                        "input mesh. ");
    prm.declare_entry("p_fe_degree",
                      "1",
                      Patterns::Integer(1, 5),
                      " The polynomial degree of the pressure finite" 
                        "element. ");
  }
  prm.leave_subsection();

  prm.enter_subsection("Parameters of the diffusion-step solver");
  {
    prm.declare_entry("solver_max_iterations",
                      "1000",
                      Patterns::Integer(1, 1000),
                      " Maximal number of iterations done by the" 
                        "GMRES solver. ");
    prm.declare_entry("solver_tolerance",
                      "1e-12",
                      Patterns::Double(0.),
                      " Tolerance of the GMRES solver. ");
    prm.declare_entry("solver_krylov_size",
                      "30",
                      Patterns::Integer(1),
                      " The size of the Krylov subspace to be used. ");
    prm.declare_entry("solver_off_diagonals",
                      "60",
                      Patterns::Integer(0),
                      " The number of off-diagonal elements ILU must" 
                        "compute. ");
    prm.declare_entry("solver_diag_strength",
                      "0.01",
                      Patterns::Double(0.),
                      " Diagonal strengthening coefficient. ");
    prm.declare_entry("solver_update_preconditioner",
                      "15",
                      Patterns::Integer(1),
                      " This number indicates how often we need to" 
                        "update the preconditioner");
  }
  prm.leave_subsection();

  prm.declare_entry("flag_verbose_output",
                    "true",
                    Patterns::Bool(),
                    " This indicates whether the output of the" 
                      "solution process should be verbose. ");
  prm.declare_entry("flag_adaptive_time_step",
                    "true",
                    Patterns::Bool(),
                    " This indicates whether the output " 
                      "is fixed or adaptive. ");
  prm.declare_entry("flag_DFG_benchmark",
                    "false",
                    Patterns::Bool(),
                    " This indicates if the problem solves the DFG "
                    "benchmark or step-35");
  prm.declare_entry("graphical_output_interval",
                    "1",
                    Patterns::Integer(1),
                    " This indicates between how many time steps" 
                      "we output the solution for visualization. ");
  prm.declare_entry("terminal_output_interval",
                    "1",
                    Patterns::Integer(1),
                    " This indicates between how many time steps" 
                      "we print the point evaluation to the terminal.");
}

void ParameterSet::
read_data_from_file(const std::string &filename)
{
  std::ifstream file(filename);
  AssertThrow(file, ExcFileNotOpen(filename));

  prm.parse_input(file);

  if (prm.get("projection_method") == std::string("rotational"))
    projection_method = ProjectionMethod::rotational;
  else
    projection_method = ProjectionMethod::standard;

  prm.enter_subsection("Physical parameters");
  {
    Re  = prm.get_double("Re");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time discretization parameters");
  {
    dt  = prm.get_double("dt");
    t_0 = prm.get_double("t_0");
    T   = prm.get_double("T");
    vsimex_input_gamma = prm.get_double("vsimex_input_gamma");
    vsimex_input_c     = prm.get_double("vsimex_input_c");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space discretization parameters");
  {
    n_global_refinements  = prm.get_integer("n_global_refinements");
    p_fe_degree           = prm.get_integer("p_fe_degree");
  }
  prm.leave_subsection();

  prm.enter_subsection("Parameters of the diffusion-step solver");
  {
    solver_max_iterations = prm.get_integer("solver_max_iterations");
    solver_tolerance      = prm.get_double("solver_tolerance");
    solver_krylov_size    = prm.get_integer("solver_krylov_size");
    solver_off_diagonals  = prm.get_integer("solver_off_diagonals");
    solver_diag_strength  = prm.get_double("solver_diag_strength");
    solver_update_preconditioner  = prm.get_integer(
      "solver_update_preconditioner");
  }
  prm.leave_subsection();

  flag_verbose_output       = prm.get_bool("flag_verbose_output");
  flag_adaptive_time_step   = prm.get_bool("flag_adaptive_time_step");
  flag_DFG_benchmark        = prm.get_bool("flag_DFG_benchmark");
  graphical_output_interval = prm.get_integer("graphical_output_interval");
  terminal_output_interval  = prm.get_integer("terminal_output_interval");
}

} // namespace RunTimeParameters
  
} // namespace Step35
