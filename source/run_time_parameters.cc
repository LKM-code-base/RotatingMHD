/*
 * run_time_parameters.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/base/conditional_ostream.h>

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
convection_term_form(ConvectiveTermWeakForm::skewsymmetric),
Re(1.0),
Pe(1.0),
n_global_refinements(0),
p_fe_degree(1),
temperature_fe_degree(1),
n_maximum_iterations(1000),
solver_krylov_size(30),
solver_off_diagonals(60),
solver_update_preconditioner(15),
relative_tolerance(1e-6),
solver_diag_strength(0.01),
verbose(true),
flag_semi_implicit_convection(true),
graphical_output_interval(15),
terminal_output_interval(1),
adaptive_meshing_interval(20),
refinement_and_coarsening_max_level(10),
refinement_and_coarsening_min_level(1),
flag_spatial_convergence_test(true),
initial_refinement_level(3),
final_refinement_level(8),
time_step_scaling_factor(1./10.)
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

  prm.declare_entry("convection_term_form",
                    "skewsymmetric",
                    Patterns::Selection("standard|skewsymmetric|divergence|rotational"),
                    " Form the of the convection term to implement. ");

  prm.enter_subsection("Physical parameters");
  {
    prm.declare_entry("Reynolds_number",
                      "1.",
                      Patterns::Double(0.),
                      "The kinetic Reynolds number.");
    prm.declare_entry("Peclet_number",
                      "1.",
                      Patterns::Double(0.),
                      "The Peclet number.");
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
    prm.declare_entry("temperature_fe_degree",
                      "1",
                      Patterns::Integer(1, 5),
                      " The polynomial degree of the temperature finite" 
                        "element. ");
    prm.declare_entry("refinement_and_coarsening_max_level",
                      "0",
                      Patterns::Integer(0),
                      "Maximum refinement and coarsening level"
                      " allowed.");
    prm.declare_entry("refinement_and_coarsening_min_level",
                      "0",
                      Patterns::Integer(0),
                      "Minimum refinement and coarsening level"
                      " allowed.");
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

  prm.enter_subsection("Convergence test parameters");
  {
    prm.declare_entry("flag_spatial_convergence_test",
                      "true",
                      Patterns::Bool(),
                      "Chooses between an spatial or temporal "
                      "convergence test");
    prm.declare_entry("initial_refinement_level",
                      "3",
                      Patterns::Integer(1),
                      "The initial refinement level of the test");
    prm.declare_entry("final_refinement_level",
                      "8",
                      Patterns::Integer(1),
                      "The final refinement level of the test");
    prm.declare_entry("temporal_convergence_cycles",
                      "6",
                      Patterns::Integer(1),
                      "The amount of cycles for the temporal convergence test");
    prm.declare_entry("time_step_scaling_factor",
                      "0.1",
                      Patterns::Double(0.),
                      "The scaling factor of the temporal convergence test");
  }
  prm.leave_subsection();

  prm.declare_entry("verbosity_flag",
                    "true",
                    Patterns::Bool(),
                    "Verbosity flag.");

  prm.declare_entry("semi_implicit_convection_flag",
                    "true",
                    Patterns::Bool(),
                    "Flag determing the treatment of the convection"
                    " term inside the VSIMEX method.");

  prm.declare_entry("graphical_output_frequency",
                    "1",
                    Patterns::Integer(1),
                    "Graphical output frequency. ");

  prm.declare_entry("diagnostics_output_frequency",
                    "1",
                    Patterns::Integer(1),
                    "Output frequency of diagnostic data on the terminal.");

  prm.declare_entry("adaptive_meshing_frequency",
                    "1",
                    Patterns::Integer(1),
                    "Frequency at which adaptive refinement and"
                    " coarsening is perofmed");
}


void ParameterSet::parse_parameters(ParameterHandler &prm)
{
  time_stepping_parameters.parse_parameters(prm);

  if (prm.get("projection_method") == std::string("rotational"))
    projection_method = ProjectionMethod::rotational;
  else if (prm.get("projection_method") == std::string("standard"))
    projection_method = ProjectionMethod::standard;
  else
    AssertThrow(false,
                ExcMessage("Unexpected projection method."));

  if (prm.get("convection_term_form") == std::string("standard"))
    convection_term_form = ConvectiveTermWeakForm::standard;
  else if (prm.get("convection_term_form") == std::string("skewsymmetric"))
    convection_term_form = ConvectiveTermWeakForm::skewsymmetric;
  else if (prm.get("convection_term_form") == std::string("divergence"))
    convection_term_form = ConvectiveTermWeakForm::divergence;
  else if (prm.get("convection_term_form") == std::string("rotational"))
    convection_term_form = ConvectiveTermWeakForm::rotational;
  else
    AssertThrow(false,
                ExcMessage("Unexpected identifier for the weak form "
                           "of the convective term."));
  

  prm.enter_subsection("Physical parameters");
  {
    Re  = prm.get_double("Reynolds_number");

    Assert(Re > 0, ExcLowerRange(Re, 0));

    Pe  = prm.get_double("Peclet_number");

    Assert(Pe > 0, ExcLowerRange(Pe, 0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Spatial discretization parameters");
  {
    n_global_refinements  = prm.get_integer("n_global_refinements");

    p_fe_degree           = prm.get_integer("p_fe_degree");

    temperature_fe_degree = prm.get_integer("temperature_fe_degree");

    refinement_and_coarsening_max_level = prm.get_integer("refinement_and_coarsening_max_level");

    refinement_and_coarsening_min_level = prm.get_integer("refinement_and_coarsening_min_level");

    Assert(n_global_refinements > 0, ExcLowerRange(n_global_refinements, 0));
    Assert(p_fe_degree > 0, ExcLowerRange(p_fe_degree, 0));
    Assert(temperature_fe_degree > 0, ExcLowerRange(temperature_fe_degree, 0));
    Assert(refinement_and_coarsening_max_level > 0, 
           ExcLowerRange(refinement_and_coarsening_max_level, 0));
    Assert(refinement_and_coarsening_min_level > 0, 
           ExcLowerRange(refinement_and_coarsening_min_level, 0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Parameters of the diffusion-step solver");
  {
    n_maximum_iterations  = prm.get_integer("n_maximum_iterations");

    relative_tolerance    = prm.get_double("relative_tolerance");

    solver_krylov_size    = prm.get_integer("gmres_restart_parameter");

    solver_off_diagonals  = prm.get_integer("solver_off_diagonals");

    solver_diag_strength  = prm.get_double("solver_diag_strength");

    solver_update_preconditioner  = prm.get_integer("update_frequency_preconditioner");
  
    Assert(n_maximum_iterations > 0, ExcLowerRange(n_maximum_iterations, 0));
    Assert(relative_tolerance > 0, ExcLowerRange(relative_tolerance, 0));
    Assert(solver_krylov_size > 0, ExcLowerRange(solver_krylov_size, 0));
    Assert(solver_off_diagonals > 0, ExcLowerRange(solver_off_diagonals, 0));
    Assert(solver_diag_strength > 0, ExcLowerRange(solver_diag_strength, 0));
    Assert(solver_update_preconditioner > 0, ExcLowerRange(solver_update_preconditioner, 0));
  }
  prm.leave_subsection();

  prm.enter_subsection("Convergence test parameters");
  {
    flag_spatial_convergence_test = prm.get_bool("flag_spatial_convergence_test");

    initial_refinement_level = prm.get_integer("initial_refinement_level");

    final_refinement_level = prm.get_integer("final_refinement_level");

    temporal_convergence_cycles = prm.get_integer("temporal_convergence_cycles");

    time_step_scaling_factor = prm.get_double("time_step_scaling_factor");
  }
  prm.leave_subsection();

  verbose                       = prm.get_bool("verbosity_flag");
  flag_semi_implicit_convection = prm.get_bool("semi_implicit_convection_flag");

  graphical_output_interval = prm.get_integer("graphical_output_frequency");
  terminal_output_interval  = prm.get_integer("diagnostics_output_frequency");
  adaptive_meshing_interval = prm.get_integer("adaptive_meshing_frequency");

  Assert(graphical_output_interval > 0, 
         ExcLowerRange(graphical_output_interval, 0));
  Assert(terminal_output_interval > 0, 
         ExcLowerRange(terminal_output_interval, 0));
  Assert(adaptive_meshing_interval > 0, 
         ExcLowerRange(adaptive_meshing_interval, 0));
}

ConvergenceAnalysisParameters::ConvergenceAnalysisParameters()
:
spatial_convergence_test(false),
n_global_initial_refinements(0),
n_spatial_convergence_cycles(2),
temporal_convergence_test(false),
n_temporal_convergence_cycles(2),
timestep_reduction_factor(0.1)
{}

ConvergenceAnalysisParameters::ConvergenceAnalysisParameters
(const std::string &parameter_filename)
:
ConvergenceAnalysisParameters()
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

RefinementParameters::RefinementParameters()
:
adaptive_mesh_refinement(false),
adaptive_mesh_refinement_frequency(100),
n_maximum_levels(5),
n_minimum_levels(1),
n_adaptive_initial_refinements(0),
n_global_initial_refinements(0),
n_boundary_initial_refinements(0)
{}

RefinementParameters::RefinementParameters
(const std::string &parameter_filename)
:
RefinementParameters()
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

void RefinementParameters::declare_parameters(ParameterHandler &prm)
{

  prm.enter_subsection("Refinement control parameters");
  {
    prm.declare_entry("Adaptive mesh refinement",
                      "false",
                      Patterns::Bool(),
                      "Enable or disable adaptive mesh refinement");

    prm.declare_entry("Adaptive mesh refinement frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Maximum number of levels",
                      "5",
                      Patterns::Integer(1));

    prm.declare_entry("Minimum number of levels",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of global initial refinements",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of adaptive initial refinements",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of initial boundary refinements",
                      "0",
                      Patterns::Integer(0));
  }
  prm.leave_subsection();
}

void RefinementParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Refinement control parameters");
  {
    adaptive_mesh_refinement = prm.get_bool("Adaptive mesh refinement");

    if (adaptive_mesh_refinement)
    {
      adaptive_mesh_refinement_frequency = prm.get_integer("Adaptive mesh refinement frequency");

      n_maximum_levels = prm.get_integer("Maximum number of levels");

      n_minimum_levels = prm.get_integer("Minimum number of levels");

      Assert(n_minimum_levels > 0,
             ExcMessage("Minimum number of levels must be larger than zero."));
      Assert(n_minimum_levels <= n_maximum_levels ,
             ExcMessage("Maximum number of levels must be larger equal than the "
                        "minimum number of levels."));

      n_global_initial_refinements = prm.get_integer("Number of global initial refinements");

      n_adaptive_initial_refinements = prm.get_integer("Number of adaptive initial refinements");

      n_boundary_initial_refinements = prm.get_integer("Number of initial boundary refinements");

      const unsigned int n_initial_refinements
      = n_global_initial_refinements + n_adaptive_initial_refinements
      + n_boundary_initial_refinements;

      Assert(n_initial_refinements <= n_maximum_levels ,
             ExcMessage("Number of initial refinements must be less equal than "
                        "the maximum number of levels."));
    }
  }
  prm.leave_subsection();
}

template<typename Stream>
Stream& operator<<(Stream &stream, const RefinementParameters &prm)
{
  const size_t column_width[2] =
      {
          std::string("----------------------------------").size(),
          std::string("-------------------").size()
      };
  const char header[] = "+-----------------------------------+--------------------+";

  auto add_line = [&]
                  (const char first_column[],
                   const auto second_column)->void
    {
      stream << "| "
             << std::setw(column_width[0]) << first_column
             << "| "
             << std::setw(column_width[1]) << second_column
             << "|"
             << std::endl;
    };

  stream << std::left << header << std::endl;

  add_line("Refinement control parameters","");

  stream << header << std::endl;

  add_line("Adaptive mesh refinement", (prm.adaptive_mesh_refinement ?
                                        "True": "False"));
  add_line("Adapt. mesh refinement frequency", prm.adaptive_mesh_refinement_frequency);
  add_line("Maximum number of levels", prm.n_maximum_levels);
  add_line("Minimum number of levels", prm.n_minimum_levels);
  add_line("Number of adapt. initial refinements", prm.n_adaptive_initial_refinements);
  add_line("Number of adapt. global refinements", prm.n_global_initial_refinements);
  add_line("Number of initial boundary refinements", prm.n_boundary_initial_refinements);

  stream << header << std::endl;

  return (stream);
}

OutputControlParameters::OutputControlParameters()
:
graphical_output_frequency(100),
terminal_output_frequency(100),
graphical_output_directory("./")
{}

OutputControlParameters::OutputControlParameters
(const std::string &parameter_filename)
:
OutputControlParameters()
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

void OutputControlParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Output control parameters");
  {
    prm.declare_entry("Graphical output frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Terminal output frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Graphical output directory",
                      "./",
                      Patterns::DirectoryName());
  }
  prm.leave_subsection();
}

void OutputControlParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Output control parameters");
  {
    graphical_output_frequency = prm.get_integer("Graphical output frequency");
    Assert(graphical_output_frequency > 0,
           ExcMessage("The graphical output frequency must larger than zero."));


    terminal_output_frequency = prm.get_integer("Terminal output frequency");
    Assert(terminal_output_frequency > 0,
           ExcMessage("The terminal output frequency must larger than zero."));

    graphical_output_directory = prm.get("Graphical output directory");
  }
  prm.leave_subsection();
}

template<typename Stream>
Stream& operator<<(Stream &stream, const OutputControlParameters &prm)
{
  const size_t column_width[2] =
      {
          std::string("----------------------------------").size(),
          std::string("-------------------").size()
      };
  const char header[] = "+-----------------------------------+--------------------+";

  auto add_line = [&]
                  (const char first_column[],
                   const auto second_column)->void
    {
      stream << "| "
             << std::setw(column_width[0]) << first_column
             << "| "
             << std::setw(column_width[1]) << second_column
             << "|"
             << std::endl;
    };

  stream << std::left << header << std::endl;

  add_line("Output control parameters","");

  stream << header << std::endl;

  add_line("Graphical output frequency", prm.graphical_output_frequency);
  add_line("Terminal output frequency", prm.terminal_output_frequency);
  add_line("Graphical output directory", prm.graphical_output_directory);

  stream << header << std::endl;

  return (stream);
}

ProblemParameters::ProblemParameters()
:
OutputControlParameters(),
RefinementParameters(),
TimeSteppingParameters(),
verbose(false)
{}

ProblemParameters::ProblemParameters
(const std::string &parameter_filename)
:
ProblemParameters()
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

void ProblemParameters::declare_parameters(ParameterHandler &prm)
{
  OutputControlParameters::declare_parameters(prm);

  RefinementParameters::declare_parameters(prm);

  TimeSteppingParameters::declare_parameters(prm);

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());
}

void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  OutputControlParameters::parse_parameters(prm);

  RefinementParameters::parse_parameters(prm);

  TimeSteppingParameters::parse_parameters(prm);

  verbose = prm.get_bool("Verbose");
}

template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm)
{
  stream << static_cast<const OutputControlParameters &>(prm);

  stream << static_cast<const RefinementParameters &>(prm);

  stream << static_cast<const TimeSteppingParameters &>(prm);
}


LinearSolverParameters::LinearSolverParameters()
:
relative_tolerance(1e-6),
absolute_tolerance(1e-9),
n_maximum_iterations(50)
{}

LinearSolverParameters::LinearSolverParameters
(const std::string &parameter_filename)
:
LinearSolverParameters()
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

void LinearSolverParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Maximum number of iterations",
                    "50",
                    Patterns::Integer(1));

  prm.declare_entry("Relative tolerance",
                    "1e-6",
                    Patterns::Double());

  prm.declare_entry("Absolute tolerance",
                    "1e-9",
                    Patterns::Double());
}

void LinearSolverParameters::parse_parameters(const ParameterHandler &prm)
{
  n_maximum_iterations = prm.get_integer("Maximum number of iterations");
  Assert(n_maximum_iterations > 0, ExcLowerRange(n_maximum_iterations, 0));

  relative_tolerance = prm.get_double("Relative tolerance");
  Assert(relative_tolerance > 0, ExcLowerRange(relative_tolerance, 0));

  absolute_tolerance = prm.get_double("Absolute tolerance");
  Assert(relative_tolerance > absolute_tolerance, ExcLowerRange(relative_tolerance , absolute_tolerance));
}

template<typename Stream>
Stream& operator<<(Stream &stream, const LinearSolverParameters &prm)
{
  const size_t column_width[2] =
      {
          std::string("----------------------------------").size(),
          std::string("-------------------").size()
      };
  const char header[] = "+-----------------------------------+--------------------+";

  auto add_line = [&]
                  (const char first_column[],
                   const auto second_column)->void
    {
      stream << "| "
             << std::setw(column_width[0]) << first_column
             << "| "
             << std::setw(column_width[1]) << second_column
             << "|"
             << std::endl;
    };

  stream << std::left << header << std::endl;

  add_line("Linear solver parameters","");

  stream << header << std::endl;

  add_line("Maximum number of iterations", prm.n_maximum_iterations);
  add_line("Relative tolerance", prm.relative_tolerance);
  add_line("Absolute tolerance", prm.absolute_tolerance);

  stream << header << std::endl;

  return (stream);
}

} // namespace RunTimeParameters
  
} // namespace RMHD

// explicit instantiations
template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::RefinementParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::RefinementParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::OutputControlParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::OutputControlParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::LinearSolverParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::LinearSolverParameters &);
