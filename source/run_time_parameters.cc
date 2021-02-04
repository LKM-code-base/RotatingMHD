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


SpatialDiscretizationParameters::SpatialDiscretizationParameters()
:
adaptive_mesh_refinement(false),
adaptive_mesh_refinement_frequency(100),
cell_fraction_to_coarsen(0.30),
cell_fraction_to_refine(0.03),
n_maximum_levels(5),
n_minimum_levels(1),
n_initial_adaptive_refinements(0),
n_initial_global_refinements(0),
n_initial_boundary_refinements(0)
{}



SpatialDiscretizationParameters::SpatialDiscretizationParameters
(const std::string &parameter_filename)
:
SpatialDiscretizationParameters()
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



void SpatialDiscretizationParameters::declare_parameters(ParameterHandler &prm)
{

  prm.enter_subsection("Refinement control parameters");
  {
    prm.declare_entry("Adaptive mesh refinement",
                      "false",
                      Patterns::Bool());

    prm.declare_entry("Adaptive mesh refinement frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Fraction of cells set to coarsen",
                      "0.3",
                      Patterns::Double(0));

    prm.declare_entry("Fraction of cells set to refine",
                      "0.03",
                      Patterns::Double(0));

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



void SpatialDiscretizationParameters::parse_parameters(ParameterHandler &prm)
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

      cell_fraction_to_coarsen = prm.get_double("Fraction of cells set to coarsen");

      cell_fraction_to_refine = prm.get_double("Fraction of cells set to refine");

      const double total_cell_fraction_to_modify =
        cell_fraction_to_coarsen + cell_fraction_to_refine;

      Assert(cell_fraction_to_coarsen >= 0.0,
             ExcLowerRange(cell_fraction_to_coarsen, 0));

      Assert(cell_fraction_to_refine >= 0.0,
             ExcLowerRange(cell_fraction_to_refine, 0));

      Assert(1.0 > total_cell_fraction_to_modify,
             ExcMessage("The sum of the top and bottom fractions to "
                        "coarsen and refine may not exceed 1.0"));
    }

    n_initial_global_refinements = prm.get_integer("Number of initial global refinements");

    n_initial_adaptive_refinements = prm.get_integer("Number of initial adaptive refinements");

    n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

    const unsigned int n_initial_refinements
    = n_initial_global_refinements + n_initial_adaptive_refinements
    + n_initial_boundary_refinements;

    Assert(n_initial_refinements <= n_maximum_levels ,
            ExcMessage("Number of initial refinements must be less equal than "
                      "the maximum number of levels."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SpatialDiscretizationParameters &prm)
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
  if (prm.adaptive_mesh_refinement)
  {
    add_line("Adapt. mesh refinement frequency", prm.adaptive_mesh_refinement_frequency);
    add_line("Fraction of cells set to coarsen", prm.cell_fraction_to_coarsen);
    add_line("Fraction of cells set to refine", prm.cell_fraction_to_refine);
    add_line("Maximum number of levels", prm.n_maximum_levels);
    add_line("Minimum number of levels", prm.n_minimum_levels);
  }
  add_line("Number of initial adapt. refinements", prm.n_initial_adaptive_refinements);
  add_line("Number of initial global refinements", prm.n_initial_global_refinements);
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



ConvergenceTestParameters::ConvergenceTestParameters()
:
convergence_test_type(ConvergenceTestType::temporal),
n_global_initial_refinements(5),
n_spatial_convergence_cycles(2),
timestep_reduction_factor(0.5),
n_temporal_convergence_cycles(2)
{}



ConvergenceTestParameters::ConvergenceTestParameters
(const std::string &parameter_filename)
:
ConvergenceTestParameters()
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



void ConvergenceTestParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    prm.declare_entry("Convergence test type",
                      "temporal",
                      Patterns::Selection("spatial|temporal"));

    prm.declare_entry("Number of initial global refinements",
                      "5",
                      Patterns::Integer(1));

    prm.declare_entry("Number of spatial convergence cycles",
                      "2",
                      Patterns::Integer(1));

    prm.declare_entry("Time-step reduction factor",
                      "0.5",
                      Patterns::Double());

    prm.declare_entry("Number of temporal convergence cycles",
                      "2",
                      Patterns::Integer(1));
  }
  prm.leave_subsection();
}



void ConvergenceTestParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    if (prm.get("Convergence test type") == std::string("spatial"))
      convergence_test_type = ConvergenceTestType::spatial;
    else if (prm.get("Convergence test type") == std::string("temporal"))
      convergence_test_type = ConvergenceTestType::temporal;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the type of"
                             " of convergence test."));

    n_global_initial_refinements =
                prm.get_integer("Number of initial global refinements");

    Assert(n_global_initial_refinements > 0,
           ExcLowerRange(n_global_initial_refinements, 0));

    n_spatial_convergence_cycles =
                prm.get_integer("Number of spatial convergence cycles");

    Assert(n_spatial_convergence_cycles > 0,
           ExcLowerRange(n_spatial_convergence_cycles, 0));

    timestep_reduction_factor =
                prm.get_double("Time-step reduction factor");

    Assert(timestep_reduction_factor > 0.0,
           ExcLowerRange(timestep_reduction_factor, 0.0));

    Assert(timestep_reduction_factor < 1.0,
           ExcLowerRange(1.0, timestep_reduction_factor));

    n_temporal_convergence_cycles =
                prm.get_integer("Number of temporal convergence cycles");

    Assert(n_temporal_convergence_cycles > 0,
           ExcLowerRange(n_temporal_convergence_cycles, 0));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ConvergenceTestParameters &prm)
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
         << "Convergence test parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  switch (prm.convergence_test_type)
  {
    case ConvergenceTestType::spatial:
      add_line("Convergence test type", "spatial");
      break;
    case ConvergenceTestType::temporal:
      add_line("Convergence test type", "temporal");
      break;
    default:
      Assert(false, ExcMessage("Unexpected identifier for the type of"
                               " of convergence test."));
      break;
  }

  add_line("Number of spatial convergence cycles",
            prm.n_spatial_convergence_cycles);

  add_line("Number of initial global refinements",
            prm.n_global_initial_refinements);

  add_line("Number of temporal convergence cycles",
            prm.n_temporal_convergence_cycles);

  add_line("Time-step reduction factor",
            prm.timestep_reduction_factor);

  stream << header;

  return (stream);
}



LinearSolverParameters::LinearSolverParameters()
:
relative_tolerance(1e-6),
absolute_tolerance(1e-9),
n_maximum_iterations(50),
solver_name("default")
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



DimensionlessNumbers::DimensionlessNumbers()
:
Re(1.0),
Pr(1.0),
Pe(1.0),
Ra(1.0),
Ek(1.0),
Pm(1.0),
problem_type(ProblemType::boussinesq)
{}



DimensionlessNumbers::DimensionlessNumbers
(const std::string &parameter_filename)
:
DimensionlessNumbers()
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



void DimensionlessNumbers::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Dimensionless numbers");
  {
    prm.declare_entry("Reynolds number",
                      "1.0",
                      Patterns::Double(0));

    prm.declare_entry("Prandtl number",
                      "1.0",
                      Patterns::Double(0));

    prm.declare_entry("Peclet number",
                      "1.0",
                      Patterns::Double(0));

    prm.declare_entry("Rayleigh number",
                      "1.0",
                      Patterns::Double(0));

    prm.declare_entry("Ekman number",
                      "1.0",
                      Patterns::Double(0));

    prm.declare_entry("magnetic Prandtl number",
                      "1.0",
                      Patterns::Double(0));
  }
  prm.leave_subsection();

  prm.declare_entry("Problem type",
                    "hydrodynamic",
                    Patterns::Selection("hydrodynamic|"
                                         "heat_convection_diffusion |"
                                         "boussinesq |"
                                         "rotating_boussinesq |"
                                         "rotating_magnetohydrodynamic"));

}



void DimensionlessNumbers::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Dimensionless numbers");
  {
    Re = prm.get_double("Reynolds number");

    Pr = prm.get_double("Prandtl number");

    Pe = prm.get_double("Peclet number");

    Ra = prm.get_double("Rayleigh number");

    Ek = prm.get_double("Ekman number");

    Pm = prm.get_double("magnetic Prandtl number");
  }
  prm.leave_subsection();

  const std::string str_problem_type(prm.get("Problem type"));

  if (str_problem_type == std::string("hydrodynamic"))
    problem_type = ProblemType::hydrodynamic;
  else if (str_problem_type == std::string("heat_convection_diffusion"))
    problem_type = ProblemType::heat_convection_diffusion;
  else if (str_problem_type == std::string("boussinesq"))
    problem_type = ProblemType::boussinesq;
  else if (str_problem_type == std::string("rotating_boussinesq"))
    problem_type = ProblemType::rotating_boussinesq;
  else if (str_problem_type == std::string("rotating_magnetohydrodynamic"))
    problem_type = ProblemType::rotating_magnetohydrodynamic;
  else
    AssertThrow(false,
                ExcMessage("Unexpected identifier for the problem"
                           " type."));

  if (str_problem_type == std::string("hydrodynamic"))
  {
    Assert(Re > 0.0, ExcLowerRangeType<double>(Re, 0.0));
    AssertIsFinite(Re);
  }
  else if (str_problem_type == std::string("heat_convection_diffusion"))
  {
    Assert(Pe > 0.0, ExcLowerRangeType<double>(Pe, 0.0));
    AssertIsFinite(Pe);
  }
  else if (str_problem_type == std::string("boussinesq"))
  {
    Assert(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
    AssertIsFinite(Ra);

    Assert(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
    AssertIsFinite(Pr);
  }
  else if (str_problem_type == std::string("rotating_boussinesq"))
  {
    Assert(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
    AssertIsFinite(Ra);

    Assert(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
    AssertIsFinite(Pr);

    Assert(Ek > 0.0, ExcLowerRangeType<double>(Ek, 0.0));
    AssertIsFinite(Ek);
  }
  else if (str_problem_type == std::string("rotating_magnetohydrodynamic"))
  {
    Assert(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
    AssertIsFinite(Ra);

    Assert(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
    AssertIsFinite(Pr);

    Assert(Ek > 0.0, ExcLowerRangeType<double>(Ek, 0.0));
    AssertIsFinite(Ek);

    Assert(Pm > 0.0, ExcLowerRangeType<double>(Pm, 0.0));
    AssertIsFinite(Pm);
  }
  else
    AssertThrow(false,
                ExcMessage("Unexpected identifier for the problem"
                           " type."));
}



template<typename Stream>
Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm)
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
         << "Dimensionless numbers"
         << "|"
         << std::endl;

  stream << header << std::endl;

  switch (prm.problem_type)
  {
    case ProblemType::hydrodynamic:
      add_line("Reynolds number", prm.Re);
      break;
    case ProblemType::heat_convection_diffusion:
      add_line("Peclet number", prm.Pe);
      break;
    case ProblemType::boussinesq:
      add_line("Prandtl number", prm.Pr);
      add_line("Rayleigh number", prm.Ra);
      break;
    case ProblemType::rotating_boussinesq:
      add_line("Prandtl number", prm.Pr);
      add_line("Rayleigh number", prm.Ra);
      add_line("Ekman number", prm.Ek);
      break;
    case ProblemType::rotating_magnetohydrodynamic:
      add_line("Prandtl number", prm.Pr);
      add_line("Rayleigh number", prm.Ra);
      add_line("Ekman number", prm.Ek);
      add_line("magnetic Prandtl number", prm.Pm);
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "problem type"));
      break;
  }

  stream << header;

  return (stream);
}



NavierStokesParameters::NavierStokesParameters()
:
pressure_correction_scheme(PressureCorrectionScheme::rotational),
convective_term_weak_form(ConvectiveTermWeakForm::skewsymmetric),
convective_term_time_discretization(ConvectiveTermTimeDiscretization::semi_implicit),
C1(0.0),
C2(1.0),
C3(0.0),
C5(0.0),
diffusion_step_solver_parameters(),
projection_step_solver_parameters(),
correction_step_solver_parameters(),
poisson_prestep_solver_parameters(),
preconditioner_update_frequency(10),
verbose(false)
{}



NavierStokesParameters::NavierStokesParameters
(const std::string &parameter_filename)
:
NavierStokesParameters()
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



void NavierStokesParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes solver parameters");
  {
    prm.declare_entry("Incremental pressure-correction scheme",
                      "rotational",
                      Patterns::Selection("rotational|standard"));

    prm.declare_entry("Convective term weak form",
                      "skew-symmetric",
                      Patterns::Selection("standard|skew-symmetric|divergence|rotational"));

    prm.declare_entry("Convective term time discretization",
                      "semi-implicit",
                      Patterns::Selection("semi-implicit|explicit"));

    prm.declare_entry("Preconditioner update frequency",
                      "10",
                      Patterns::Integer(1));

    prm.declare_entry("Verbose",
                      "false",
                      Patterns::Bool());

    prm.enter_subsection("Linear solver parameters - Diffusion step");
    {
      LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Projection step");
    {
      LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Correction step");
    {
      LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Poisson pre-step");
    {
      LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void NavierStokesParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes solver parameters");
  {
    const std::string str_pressure_correction_scheme(prm.get("Incremental pressure-correction scheme"));

    if (str_pressure_correction_scheme == std::string("rotational"))
      pressure_correction_scheme = PressureCorrectionScheme::rotational;
    else if (str_pressure_correction_scheme == std::string("standard"))
      pressure_correction_scheme = PressureCorrectionScheme::standard;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identified for the incremental "
                             "pressure-correction scheme."));

    const std::string str_convective_term_weak_form(prm.get("Convective term weak form"));

    if (str_convective_term_weak_form == std::string("standard"))
      convective_term_weak_form = ConvectiveTermWeakForm::standard;
    else if (str_convective_term_weak_form == std::string("skew-symmetric"))
      convective_term_weak_form = ConvectiveTermWeakForm::skewsymmetric;
    else if (str_convective_term_weak_form == std::string("divergence"))
      convective_term_weak_form = ConvectiveTermWeakForm::divergence;
    else if (str_convective_term_weak_form == std::string("rotational"))
      convective_term_weak_form = ConvectiveTermWeakForm::rotational;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the weak form "
                             "of the convective term."));

    const std::string str_convective_term_time_discretization(prm.get("Convective term time discretization"));

    if (str_convective_term_time_discretization == std::string("semi-implicit"))
      convective_term_time_discretization = ConvectiveTermTimeDiscretization::semi_implicit;
    else if (str_convective_term_time_discretization == std::string("explicit"))
      convective_term_time_discretization = ConvectiveTermTimeDiscretization::fully_explicit;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the time discretization "
                             "of the convective term."));

    preconditioner_update_frequency = prm.get_integer("Preconditioner update frequency");
    Assert(preconditioner_update_frequency > 0,
           ExcLowerRange(preconditioner_update_frequency, 0));

    verbose = prm.get_bool("Verbose");

    prm.enter_subsection("Linear solver parameters - Diffusion step");
    {
      diffusion_step_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Projection step");
    {
      projection_step_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Correction step");
    {
      correction_step_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Linear solver parameters - Poisson pre-step");
    {
      poisson_prestep_solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const NavierStokesParameters &prm)
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
         << "Navier-Stokes discretization parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  switch (prm.pressure_correction_scheme)
  {
    case PressureCorrectionScheme::standard:
      add_line("Incremental pressure-correction scheme", "standard");
      break;
    case PressureCorrectionScheme::rotational:
      add_line("Incremental pressure-correction scheme", "rotational");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "incremental pressure-correction scheme"));
      break;
  }

  switch (prm.convective_term_weak_form) {
    case ConvectiveTermWeakForm::standard:
      add_line("Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      add_line("Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      add_line("Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      add_line("Convective term weak form", "skew-symmetric");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "weak form of the convective term."));
      break;
  }

  switch (prm.convective_term_time_discretization) {
    case ConvectiveTermTimeDiscretization::semi_implicit:
      add_line("Convective temporal form", "semi-implicit");
      break;
    case ConvectiveTermTimeDiscretization::fully_explicit:
      add_line("Convective temporal form", "explicit");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "time discretization of the convective term."));
      break;
  }

  add_line("Preconditioner update frequency", prm.preconditioner_update_frequency);

  stream << prm.diffusion_step_solver_parameters;

  stream << "\r";

  stream << prm.projection_step_solver_parameters;

  stream << "\r";

  stream << prm.correction_step_solver_parameters;

  stream << "\r";

  stream << prm.poisson_prestep_solver_parameters;

  stream << "\r";

  stream << header;

  return (stream);
}



HeatEquationParameters::HeatEquationParameters()
:
convective_term_weak_form(ConvectiveTermWeakForm::skewsymmetric),
convective_term_time_discretization(ConvectiveTermTimeDiscretization::semi_implicit),
C4(1.0),
solver_parameters(),
preconditioner_update_frequency(10),
verbose(false)
{}



HeatEquationParameters::HeatEquationParameters
(const std::string &parameter_filename)
:
HeatEquationParameters()
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



void HeatEquationParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Heat equation solver parameters");
  {
    prm.declare_entry("Convective term weak form",
                      "skew-symmetric",
                      Patterns::Selection("standard|skew-symmetric|divergence|rotational"));

    prm.declare_entry("Convective term time discretization",
                      "semi-implicit",
                      Patterns::Selection("semi-implicit|explicit"));

    prm.declare_entry("Preconditioner update frequency",
                      "10",
                      Patterns::Integer(1));

    prm.declare_entry("Verbose",
                      "false",
                      Patterns::Bool());

    prm.enter_subsection("Linear solver parameters");
    {
      LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void HeatEquationParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Heat equation solver parameters");
  {
    const std::string str_convective_term_weak_form(prm.get("Convective term weak form"));

    if (str_convective_term_weak_form == std::string("standard"))
      convective_term_weak_form = ConvectiveTermWeakForm::standard;
    else if (str_convective_term_weak_form == std::string("skew-symmetric"))
      convective_term_weak_form = ConvectiveTermWeakForm::skewsymmetric;
    else if (str_convective_term_weak_form == std::string("divergence"))
      convective_term_weak_form = ConvectiveTermWeakForm::divergence;
    else if (str_convective_term_weak_form == std::string("rotational"))
      convective_term_weak_form = ConvectiveTermWeakForm::rotational;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the weak form "
                             "of the convective term."));

    const std::string str_convective_term_time_discretization(prm.get("Convective term time discretization"));

    if (str_convective_term_time_discretization == std::string("semi-implicit"))
      convective_term_time_discretization = ConvectiveTermTimeDiscretization::semi_implicit;
    else if (str_convective_term_time_discretization == std::string("explicit"))
      convective_term_time_discretization = ConvectiveTermTimeDiscretization::fully_explicit;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the time discretization "
                             "of the convective term."));

    preconditioner_update_frequency = prm.get_integer("Preconditioner update frequency");
    Assert(preconditioner_update_frequency > 0,
           ExcLowerRange(preconditioner_update_frequency, 0));

    verbose = prm.get_bool("Verbose");

    prm.enter_subsection("Linear solver parameters");
    {
      solver_parameters.parse_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const HeatEquationParameters &prm)
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
         << "Heat equation solver parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  switch (prm.convective_term_weak_form) {
    case ConvectiveTermWeakForm::standard:
      add_line("Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      add_line("Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      add_line("Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      add_line("Convective term weak form", "skew-symmetric");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "weak form of the convective term."));
      break;
  }

  switch (prm.convective_term_time_discretization) {
    case ConvectiveTermTimeDiscretization::semi_implicit:
      add_line("Convective temporal form", "semi-implicit");
      break;
    case ConvectiveTermTimeDiscretization::fully_explicit:
      add_line("Convective temporal form", "explicit");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "time discretization of the convective term."));
      break;
  }

  add_line("Preconditioner update frequency", prm.preconditioner_update_frequency);

  stream << prm.solver_parameters;

  stream << "\r";

  stream << header;

  return (stream);
}



ProblemParameters::ProblemParameters()
:
OutputControlParameters(),
DimensionlessNumbers(),
problem_type(ProblemType::boussinesq),
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
fe_degree_pressure(1),
fe_degree_velocity(2),
fe_degree_temperature(2),
verbose(false),
convergence_test_parameters(),
spatial_discretization_parameters(),
time_discretization_parameters(),
navier_stokes_parameters(),
heat_equation_parameters(),
flag_convergence_test(false)
{}



ProblemParameters::ProblemParameters
(const std::string &parameter_filename,
 const bool        flag)
:
ProblemParameters()
{
  flag_convergence_test = flag;

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

  switch (problem_type)
  {
    case ProblemType::hydrodynamic:
      navier_stokes_parameters.C1 = 0.0;
      navier_stokes_parameters.C2 = 1.0/Re;
      navier_stokes_parameters.C3 = 0.0;
      heat_equation_parameters.C4 = 0.0;
      navier_stokes_parameters.C5 = 0.0;
      navier_stokes_parameters.C6 = 1.0;
      break;
    case ProblemType::heat_convection_diffusion:
      navier_stokes_parameters.C1 = 0.0;
      navier_stokes_parameters.C2 = 0.0;
      navier_stokes_parameters.C3 = 0.0;
      heat_equation_parameters.C4 = 1.0/Pe;
      navier_stokes_parameters.C5 = 0.0;
      navier_stokes_parameters.C6 = 1.0;
      break;
    case ProblemType::boussinesq:
      navier_stokes_parameters.C1 = 0.0;
      navier_stokes_parameters.C2 = std::sqrt(Pr/Ra);
      navier_stokes_parameters.C3 = 1.0;
      heat_equation_parameters.C4 = 1.0/std::sqrt(Ra*Pr);
      navier_stokes_parameters.C5 = 0.0;
      navier_stokes_parameters.C6 = 1.0;
      break;
    case ProblemType::rotating_boussinesq:
      navier_stokes_parameters.C1 = 2.0/Ek;
      navier_stokes_parameters.C2 = 1.0;
      navier_stokes_parameters.C3 = Ra/Pr;
      heat_equation_parameters.C4 = 1.0/Pr;
      navier_stokes_parameters.C5 = 0.0;
      navier_stokes_parameters.C6 = 1.0/Ek;
      break;
    case ProblemType::rotating_magnetohydrodynamic:
      navier_stokes_parameters.C1 = 2.0/Ek;
      navier_stokes_parameters.C2 = 1.0;
      navier_stokes_parameters.C3 = Ra/Pr;
      heat_equation_parameters.C4 = 1.0/Pr;
      navier_stokes_parameters.C5 = 1.0/Pm;
      navier_stokes_parameters.C6 = 1.0;
      break;
    default:
      AssertThrow(false,
            ExcMessage("Unexpected identifier for the problem"
                        " type."));
      break;
  }

  AssertIsFinite(navier_stokes_parameters.C1);
  AssertIsFinite(navier_stokes_parameters.C2);
  AssertIsFinite(navier_stokes_parameters.C3);
  AssertIsFinite(heat_equation_parameters.C4);
  AssertIsFinite(navier_stokes_parameters.C5);

}




void ProblemParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Problem type",
                    "hydrodynamic",
                    Patterns::Selection("hydrodynamic|"
                                         "heat_convection_diffusion |"
                                         "boussinesq |"
                                         "rotating_boussinesq |"
                                         "rotating_magnetohydrodynamic"));

  prm.declare_entry("Spatial dimension",
                    "2",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Polynomial degree",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Apply to interior cells",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("FE's polynomial degree - Pressure (Taylor-Hood)",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("FE's polynomial degree - Temperature",
                    "2",
                    Patterns::Integer(1));

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());

  OutputControlParameters::declare_parameters(prm);

  DimensionlessNumbers::declare_parameters(prm);

  ConvergenceTestParameters::declare_parameters(prm);

  SpatialDiscretizationParameters::declare_parameters(prm);

  TimeDiscretization::TimeDiscretizationParameters::declare_parameters(prm);

  NavierStokesParameters::declare_parameters(prm);

  HeatEquationParameters::declare_parameters(prm);
}




void ProblemParameters::parse_parameters(ParameterHandler &prm)
{
  const std::string str_problem_type(prm.get("Problem type"));

  if (str_problem_type == std::string("hydrodynamic"))
    problem_type = ProblemType::hydrodynamic;
  else if (str_problem_type == std::string("heat_convection_diffusion"))
    problem_type = ProblemType::heat_convection_diffusion;
  else if (str_problem_type == std::string("boussinesq"))
    problem_type = ProblemType::boussinesq;
  else if (str_problem_type == std::string("rotating_boussinesq"))
    problem_type = ProblemType::rotating_boussinesq;
  else if (str_problem_type == std::string("rotating_magnetohydrodynamic"))
    problem_type = ProblemType::rotating_magnetohydrodynamic;
  else
    AssertThrow(false,
                ExcMessage("Unexpected identifier for the problem"
                           " type."));

  dim = prm.get_integer("Spatial dimension");
  Assert(dim > 0, ExcLowerRange(dim, 0) );
  Assert(dim <= 3, ExcMessage("The spatial dimension are larger than three.") );

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  Assert(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  mapping_interior_cells = prm.get_bool("Mapping - Apply to interior cells");

  fe_degree_pressure = prm.get_integer("FE's polynomial degree - Pressure (Taylor-Hood)");

  fe_degree_temperature = prm.get_integer("FE's polynomial degree - Temperature");

  Assert(fe_degree_pressure > 0,
         ExcLowerRange(fe_degree_pressure, 0));

  Assert(fe_degree_temperature > 0,
         ExcLowerRange(fe_degree_temperature, 0));

  fe_degree_velocity = fe_degree_pressure + 1;

  verbose = prm.get_bool("Verbose");

  OutputControlParameters::parse_parameters(prm);

  DimensionlessNumbers::parse_parameters(prm);

  if (flag_convergence_test)
    convergence_test_parameters.parse_parameters(prm);
  else
    spatial_discretization_parameters.parse_parameters(prm);

  time_discretization_parameters.parse_parameters(prm);

  if (str_problem_type != std::string("heat_convection_diffusion"))
    navier_stokes_parameters.parse_parameters(prm);

  if (str_problem_type != std::string("hydrodynamic"))
    heat_equation_parameters.parse_parameters(prm);
}




template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm)
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
         << "Problem parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  switch (prm.problem_type)
  {
    case ProblemType::hydrodynamic:
      add_line("Problem type", "hydrodynamic");
      break;
    case ProblemType::heat_convection_diffusion:
      add_line("Problem type", "heat_convection_diffusion");
      break;
    case ProblemType::boussinesq:
      add_line("Problem type", "boussinesq");
      break;
    case ProblemType::rotating_boussinesq:
      add_line("Problem type", "rotating_boussinesq");
      break;
    case ProblemType::rotating_magnetohydrodynamic:
      add_line("Problem type", "rotating_magnetohydrodynamic");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "problem type"));
      break;
  }

  add_line("Spatial dimension", prm.dim);

  add_line("Mapping", ("MappingQ<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.mapping_degree) + ")"));

  add_line("Mapping - Apply to interior cells", (prm.mapping_interior_cells ? "true" : "false"));

  if (prm.problem_type != ProblemType::heat_convection_diffusion)
  {
    std::string fe_velocity = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_velocity) + ")^" + std::to_string(prm.dim);
    std::string fe_pressure = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_pressure) + ")";
    add_line("Finite Element - Velocity", fe_velocity);
    add_line("Finite Element - Pressure", fe_pressure);
  }

  if (prm.problem_type != ProblemType::hydrodynamic)
  {
    std::string fe_temperature = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_temperature) + ")";
    add_line("Finite Element - Temperature", fe_temperature);
  }

  add_line("Verbose", (prm.verbose? "true": "false"));


  stream << static_cast<const OutputControlParameters &>(prm);

  stream << "\r";

  stream << static_cast<const DimensionlessNumbers &>(prm);

  stream << "\r";

  if (prm.flag_convergence_test)
    stream << prm.convergence_test_parameters;
  else
    stream << prm.spatial_discretization_parameters;

  stream << "\r";

  stream << prm.time_discretization_parameters;

  stream << "\r";

  if (prm.problem_type != ProblemType::heat_convection_diffusion)
  {
  stream << prm.navier_stokes_parameters;

  stream << "\r";
  }

  if (prm.problem_type != ProblemType::hydrodynamic)
  {
  stream << prm.heat_equation_parameters;

  stream << "\r";
  }

  return (stream);
}



} // namespace RunTimeParameters

} // namespace RMHD

// explicit instantiations
template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::SpatialDiscretizationParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::SpatialDiscretizationParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::OutputControlParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::OutputControlParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::ConvergenceTestParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ConvergenceTestParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::LinearSolverParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::LinearSolverParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::DimensionlessNumbers &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::DimensionlessNumbers &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::NavierStokesParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::NavierStokesParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::ProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ProblemParameters &);
