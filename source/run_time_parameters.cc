/*
 * data_storage.cc
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
  : projection_method(ProjectionMethod::rotational),
    vsimex_scheme(VSIMEXScheme::BDF2),
    dt(5e-4),
    timestep_lower_bound(5e-5),
    timestep_upper_bound(5e-3),
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
    prm.declare_entry("timestep_lower_bound",
                      "5e-5",
                      Patterns::Double(0.),
                      " Time step lower bound. ");
    prm.declare_entry("timestep_upper_bound",
                      "5e-3",
                      Patterns::Double(0.),
                      " Time step upper bound. ");
    prm.declare_entry("t_0",
                      "0.",
                      Patterns::Double(0.),
                      " The initial time of the simulation. ");
    prm.declare_entry("T",
                      "1.",
                      Patterns::Double(0.),
                      " The final time of the simulation. ");
    prm.declare_entry("vsimex_scheme",
                      "BDF2",
                      Patterns::Selection("FE|CNFE|BEFE|BDF2|CNAB|mCNAB|CNLF"),
                      " VSIMEX time discretizatino scheme. ");
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
    timestep_lower_bound  = prm.get_double("timestep_lower_bound");
    timestep_upper_bound  = prm.get_double("timestep_upper_bound");
    t_0 = prm.get_double("t_0");
    T   = prm.get_double("T");
    if (prm.get("vsimex_scheme") == std::string("FE"))
      vsimex_scheme = VSIMEXScheme::FE;
    else if (prm.get("vsimex_scheme") == std::string("CNFE"))
      vsimex_scheme = VSIMEXScheme::CNFE;
    else if (prm.get("vsimex_scheme") == std::string("BEFE"))
      vsimex_scheme = VSIMEXScheme::BEFE;
    else if (prm.get("vsimex_scheme") == std::string("BDF2"))
      vsimex_scheme = VSIMEXScheme::BDF2;
    else if (prm.get("vsimex_scheme") == std::string("CNAB"))
      vsimex_scheme = VSIMEXScheme::CNAB;
    else if (prm.get("vsimex_scheme") == std::string("mCNAB"))
      vsimex_scheme = VSIMEXScheme::mCNAB;
    else if (prm.get("vsimex_scheme") == std::string("CNLF"))
      vsimex_scheme = VSIMEXScheme::CNLF;
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
  graphical_output_interval = prm.get_integer("graphical_output_interval");
  terminal_output_interval  = prm.get_integer("terminal_output_interval");
}

} // namespace RunTimeParameters
  
} // namespace RMHD
