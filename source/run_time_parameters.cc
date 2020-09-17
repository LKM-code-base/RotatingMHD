/*
 * run_time_parameters.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/run_time_parameters.h>

#include <fstream>

namespace RMHD
{

using namespace dealii;

namespace RunTimeParameters
{
// In the constructor of this class we declare all the parameters. The
// details of how this works have been discussed elsewhere, for example in
// step-19 and step-29.
ParameterSet::ParameterSet()
:
time_stepping_parameters(),
projection_method(ProjectionMethod::rotational),
Re(1.0),
n_global_refinements(0),
p_fe_degree(1),
n_maximum_iterations(1000),
solver_krylov_size(30),
solver_off_diagonals(60),
solver_update_preconditioner(15),
relative_tolerance(1e-6),
solver_diag_strength(0.01),
verbose(true),
flag_DFG_benchmark(false),
graphical_output_interval(15)
{}

ParameterSet::ParameterSet(const std::string &parameter_filename)
:
ParameterSet()
{
  ParameterHandler prm;
  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;
    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());
    prm.print_parameters(parameter_out,
                         ParameterHandler::OutputStyle::Text);

    AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}

void ParameterSet::declare_parameters(ParameterHandler &prm)
{

  TimeDiscretization::TimeSteppingParameters::declare_parameters(prm);

  prm.declare_entry("projection_method",
                    "rotational",
                    Patterns::Selection("rotational|standard"),
                    " Projection method to implement. ");

  prm.enter_subsection("Physical parameters");
  {
    prm.declare_entry("Reynolds_number",
                      "1.",
                      Patterns::Double(0.),
                      "The kinetic Reynolds number.");
  }
  prm.leave_subsection();


  prm.enter_subsection("Spatial discretization parameters");
  {
    prm.declare_entry("n_global_refinements",
                      "0",
                      Patterns::Integer(0),
                      "Number of global refinements done on the input mesh.");
    prm.declare_entry("p_fe_degree",
                      "1",
                      Patterns::Integer(1, 5),
                      " The polynomial degree of the pressure finite" 
                        "element. ");
  }
  prm.leave_subsection();

  prm.enter_subsection("Parameters of the diffusion-step solver");
  {
    prm.declare_entry("n_maximum_iterations",
                      "1000",
                      Patterns::Integer(1, 1000),
                      "Maximum number of iterations done of diffusion-step "
                      "solver.");

    prm.declare_entry("relative_tolerance",
                      "1e-12",
                      Patterns::Double(0.),
                      "Relative tolerance of the diffusion-step solver.");

    prm.declare_entry("gmres_restart_parameter",
                      "30",
                      Patterns::Integer(1),
                      "Parameter which controls the size of the Krylov subspace "
                      "in the GMRES algorithm.");

    prm.declare_entry("solver_off_diagonals",
                      "60",
                      Patterns::Integer(0),
                      "The number of off-diagonal elements ILU must compute.");

    prm.declare_entry("solver_diag_strength",
                      "0.01",
                      Patterns::Double(0.),
                      "Diagonal strengthening coefficient.");

    prm.declare_entry("update_frequency_preconditioner",
                      "15",
                      Patterns::Integer(1),
                      "The number of time steps after which the preconditioner "
                      "is updated.");
  }
  prm.leave_subsection();

  prm.declare_entry("verbosity_flag",
                    "true",
                    Patterns::Bool(),
                    "Verbosity flag.");
  /*
   * Is this flag still necessary?
   */
  prm.declare_entry("flag_DFG_benchmark",
                    "false",
                    Patterns::Bool(),
                    "This indicates whether the problem solves the DFG "
                    "benchmark or step-35");

  prm.declare_entry("graphical_output_frequency",
                    "1",
                    Patterns::Integer(1),
                    "Graphical output frequency. ");

  prm.declare_entry("diagnostics_output_frequency",
                    "1",
                    Patterns::Integer(1),
                    "Output frequency of diagnostic data on the terminal.");

}


void ParameterSet::parse_parameters(ParameterHandler &prm)
{
  time_stepping_parameters.parse_parameters(prm);

  if (prm.get("projection_method") == std::string("rotational"))
    projection_method = ProjectionMethod::rotational;
  else
    projection_method = ProjectionMethod::standard;

  prm.enter_subsection("Physical parameters");
  {
    /*
     * How about a some sanity checks?
     */
    Re  = prm.get_double("Reynolds_number");
  }
  prm.leave_subsection();

  prm.enter_subsection("Spatial discretization parameters");
  {
    /*
     * How about a some sanity checks?
     */
    n_global_refinements  = prm.get_integer("n_global_refinements");
    p_fe_degree           = prm.get_integer("p_fe_degree");
  }
  prm.leave_subsection();

  prm.enter_subsection("Parameters of the diffusion-step solver");
  {
    /*
     * How about a some sanity checks?
     */
    n_maximum_iterations  = prm.get_integer("n_maximum_iterations");

    relative_tolerance    = prm.get_double("relative_tolerance");

    solver_krylov_size    = prm.get_integer("gmres_restart_parameter");

    solver_off_diagonals  = prm.get_integer("solver_off_diagonals");

    solver_diag_strength  = prm.get_double("solver_diag_strength");

    solver_update_preconditioner  = prm.get_integer("update_frequency_preconditioner");
  }
  prm.leave_subsection();

  verbose       = prm.get_bool("verbosity_flag");
  /*
   * Is this still necessary?
   */
  flag_DFG_benchmark        = prm.get_bool("flag_DFG_benchmark");

  graphical_output_interval = prm.get_integer("graphical_output_frequency");
  terminal_output_interval  = prm.get_integer("diagnostics_output_frequency");
}

} // namespace RunTimeParameters
  
} // namespace RMHD
