/*
 * general_parameters.cc
 *
 *  Created on: Nov 23, 2020
 *      Author: sg
 */

#include <rotatingMHD/general_parameters.h>

#include <deal.II/base/conditional_ostream.h>

#include <fstream>

namespace RMHD
{

namespace RunTimeParameters
{

RefinementParameters::RefinementParameters()
:
adaptive_mesh_refinement(false),
adaptive_mesh_refinement_frequency(100),
n_maximum_levels(5),
n_minimum_levels(1),
n_initial_adaptive_refinements(0),
n_initial_global_refinements(0),
n_initial_boundary_refinements(0)
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

    prm.declare_entry("Number of initial global refinements",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of initial adaptive refinements",
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

      n_initial_global_refinements = prm.get_integer("Number of global initial refinements");

      n_initial_adaptive_refinements = prm.get_integer("Number of adaptive initial refinements");

      n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

      const unsigned int n_initial_refinements
      = n_initial_global_refinements + n_initial_adaptive_refinements
      + n_initial_boundary_refinements;

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
    std::string("----------------------------------------").size(),
    std::string("-------------------").size()
  };
  const char header[] = "+-----------------------------------------+"
                        "--------------------+";

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

  stream << "| "
         << std::setw(column_width[0] + column_width[1] + 2)
         << "Refinement control parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  add_line("Adaptive mesh refinement", (prm.adaptive_mesh_refinement ?
                                        "True": "False"));
  add_line("Adapt. mesh refinement frequency", prm.adaptive_mesh_refinement_frequency);
  add_line("Maximum number of levels", prm.n_maximum_levels);
  add_line("Minimum number of levels", prm.n_minimum_levels);
  add_line("Number of adapt. initial refinements", prm.n_initial_adaptive_refinements);
  add_line("Number of adapt. global refinements", prm.n_initial_global_refinements);
  add_line("Number of initial boundary refinements", prm.n_initial_boundary_refinements);

  stream << header;

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
    std::string("----------------------------------------").size(),
    std::string("-------------------").size()
  };
  const char header[] = "+-----------------------------------------+"
                        "--------------------+";
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

  stream << "| "
         << std::setw(column_width[0] + column_width[1] + 2)
         << "Output control parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  add_line("Graphical output frequency", prm.graphical_output_frequency);
  add_line("Terminal output frequency", prm.terminal_output_frequency);
  add_line("Graphical output directory", prm.graphical_output_directory);

  stream << header;

  return (stream);
}

ProblemParameters::ProblemParameters()
:
OutputControlParameters(),
RefinementParameters(),
TimeSteppingParameters(),
dim(2),
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

  prm.declare_entry("Spatial dimension",
                    "2",
                    Patterns::Integer(1));

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());
}

void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  OutputControlParameters::parse_parameters(prm);

  RefinementParameters::parse_parameters(prm);

  TimeSteppingParameters::parse_parameters(prm);

  dim = prm.get_integer("Spatial dimension");
  Assert(dim > 0, ExcLowerRange(dim, 0) );
  Assert(dim <= 3, ExcMessage("The spatial dimension are larger than three.") );

  verbose = prm.get_bool("Verbose");
}

template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm)
{
  stream << static_cast<const OutputControlParameters &>(prm);

  stream << "\r";

  stream << static_cast<const RefinementParameters &>(prm);

  stream << "\r";

  stream << static_cast<const TimeSteppingParameters &>(prm);

  return (stream);
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
    std::string("----------------------------------------").size(),
    std::string("-------------------").size()
  };
  const char header[] = "+-----------------------------------------+"
                        "--------------------+";
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

  stream << "| "
         << std::setw(column_width[0] + column_width[1] + 2)
         << "Linear solver parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  add_line("Maximum number of iterations", prm.n_maximum_iterations);
  add_line("Relative tolerance", prm.relative_tolerance);
  add_line("Absolute tolerance", prm.absolute_tolerance);

  stream << header;

  return (stream);
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

void ConvergenceAnalysisParameters::declare_parameters(ParameterHandler &/* prm */)
{
  AssertThrow(false, ExcNotImplemented());
}

void ConvergenceAnalysisParameters::parse_parameters(ParameterHandler &/* prm */)
{
  AssertThrow(false, ExcNotImplemented());
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
(std::ostream &, const RMHD::RunTimeParameters::ProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ProblemParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::LinearSolverParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::LinearSolverParameters &);
