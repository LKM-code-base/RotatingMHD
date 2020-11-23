/*
 * navier_stokes_parameters.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */
#include <rotatingMHD/navier_stokes_parameters.h>

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



NavierStokesDiscretizationParameters::NavierStokesDiscretizationParameters()
:
projection_method(ProjectionMethod::rotational),
convective_weak_form(ConvectiveTermWeakForm::skewsymmetric),
convective_temporal_form(ConvectiveTermTimeDiscretization::fully_explicit),
preconditioner_update_frequency(10),
verbose(false),
Re(0.0),
linear_solver_control()
{}

NavierStokesDiscretizationParameters::NavierStokesDiscretizationParameters
(const std::string &parameter_filename)
:
NavierStokesDiscretizationParameters()
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

void NavierStokesDiscretizationParameters::declare_parameters
(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes discretization parameters");
  {
    prm.declare_entry("Projection method",
                      "rotational",
                      Patterns::Selection("rotational|standard"),
                      "Type of the pressure update used in the pressure projection "
                      "algorithm.");

    prm.declare_entry("Convective term weak form",
                      "skew-symmetric",
                      Patterns::Selection("standard|skew-symmetric|divergence|rotational"),
                      "Weak form of the convective term.");

    prm.declare_entry("Convective term temporal form",
                      "semi-implicit",
                      Patterns::Selection("semi-implicit|explicit"),
                      "Temporal form the of the convective term.");


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

void NavierStokesDiscretizationParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Navier-Stokes discretization parameters");
  {

    const std::string str_project_method(prm.get("Projection method"));

    if (str_project_method == std::string("rotational"))
      projection_method = ProjectionMethod::rotational;
    else if (str_project_method == std::string("standard"))
      projection_method = ProjectionMethod::standard;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected input for the type of the "
                             "projection method."));

    const std::string str_convective_weak_form(prm.get("Convective term weak form"));

    if (str_convective_weak_form == std::string("standard"))
      convective_weak_form = ConvectiveTermWeakForm::standard;
    else if (str_convective_weak_form == std::string("skew-symmetric"))
      convective_weak_form = ConvectiveTermWeakForm::skewsymmetric;
    else if (str_convective_weak_form == std::string("divergence"))
      convective_weak_form = ConvectiveTermWeakForm::divergence;
    else if (str_convective_weak_form == std::string("rotational"))
      convective_weak_form = ConvectiveTermWeakForm::rotational;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the type of the weak form "
                             "of the convective term."));

    const std::string str_convective_temporal_form(prm.get("Convective term temporal form"));

    if (str_convective_temporal_form == std::string("semi-implicit"))
      convective_temporal_form = ConvectiveTermTimeDiscretization::semi_implicit;
    else if (str_convective_temporal_form == std::string("explicit"))
      convective_temporal_form = ConvectiveTermTimeDiscretization::fully_explicit;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the type of the temporal form "
                             "of the convective term."));

    preconditioner_update_frequency = prm.get_integer("Preconditioner update frequency");
    Assert(preconditioner_update_frequency > 0,
           ExcLowerRange(preconditioner_update_frequency, 0));

    prm.enter_subsection("Linear solver parameters");
    {
      linear_solver_control.parse_parameters(prm);
    }
    prm.leave_subsection();

    verbose = prm.get_bool("Verbose");

  }
  prm.leave_subsection();

}

template<typename Stream>
Stream& operator<<(Stream &stream,
                   const NavierStokesDiscretizationParameters &prm)
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

  switch (prm.projection_method)
  {
    case ProjectionMethod::standard:
      add_line("Projection method", "standard");
      break;
    case ProjectionMethod::rotational:
      add_line("Projection method", "rotational");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "projection method."));
      break;
  }

  switch (prm.convective_weak_form) {
    case ConvectiveTermWeakForm::standard:
      add_line("Convective weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      add_line("Convective weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      add_line("Convective weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      add_line("Convective weak form", "skew-symmetric");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "weak form of the convective term."));
      break;
  }

  switch (prm.convective_temporal_form) {
    case ConvectiveTermTimeDiscretization::semi_implicit:
      add_line("Convective temporal form", "semi-implicit");
      break;
    case ConvectiveTermTimeDiscretization::fully_explicit:
      add_line("Convective temporal form", "explicit");
      break;
    default:
      Assert(false, ExcMessage("Unexpected type identifier for the "
                               "temporal form of the convective term."));
      break;
  }

  add_line("Preconditioner update frequency", prm.preconditioner_update_frequency);

  add_line("Reynolds number", prm.Re);

  stream << prm.linear_solver_control;

  stream << "\r";

  stream << header;

  return (stream);
}

NavierStokesProblemParameters::NavierStokesProblemParameters()
:
ProblemParameters(),
fe_degree(1),
navier_stokes_discretization()
{}

NavierStokesProblemParameters::NavierStokesProblemParameters
(const std::string &parameter_filename)
:
NavierStokesProblemParameters()
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

void NavierStokesProblemParameters::declare_parameters
(ParameterHandler &prm)
{
  prm.declare_entry("Polynomial degree",
                    "1",
                    Patterns::Integer(1),
                    "Polynomial degree of discretization.");

  prm.declare_entry("Reynolds number",
                    "1.0",
                    Patterns::Double(1.0));

  ProblemParameters::declare_parameters(prm);

  NavierStokesDiscretizationParameters::declare_parameters(prm);
}

void NavierStokesProblemParameters::parse_parameters(ParameterHandler &prm)
{

  ProblemParameters::parse_parameters(prm);

  navier_stokes_discretization.parse_parameters(prm);

  fe_degree = prm.get_integer("Polynomial degree");
  Assert(fe_degree > 0, ExcLowerRange(fe_degree, 0));

  verbose = prm.get_bool("Verbose");

  const double tmp = prm.get_double("Reynolds number");
  Assert(tmp > 0.0, ExcLowerRangeType<double>(tmp, 0.0));
  AssertIsFinite(tmp);

  navier_stokes_discretization.Re = tmp;
}

template<typename Stream>
Stream& operator<<(Stream &stream,
                   const NavierStokesProblemParameters &prm)
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
         << "Navier-Stokes problem parameters"
         << "|"
         << std::endl;

  stream << header << std::endl;

  add_line("Polynomial degree", prm.fe_degree);

  stream << prm.navier_stokes_discretization;

  stream << "\r";

  stream << static_cast<const ProblemParameters &>(prm);

  stream << "\r";

  stream << header;

  return (stream);
}

} // namespace RunTimeParameters
  
} // namespace RMHD

// explicit instantiations
template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::NavierStokesDiscretizationParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::NavierStokesDiscretizationParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::NavierStokesProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::NavierStokesProblemParameters &);
