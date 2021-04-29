/*
 * run_time_parameters.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */
#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <fstream>

namespace RMHD
{

using namespace dealii;

namespace RunTimeParameters
{

namespace internal
{
  constexpr char header[] = "+------------------------------------------+"
                        "----------------------+";

  constexpr size_t column_width[2] ={ 40, 20 };

  constexpr size_t line_width = 63;

  template<typename Stream, typename A>
  void add_line(Stream  &stream,
                const A &line)
  {
    stream << "| "
           << std::setw(line_width)
           << line
           << " |"
           << std::endl;
  }

  template<typename Stream, typename A, typename B>
  void add_line(Stream  &stream,
                const A &first_column,
                const B &second_column)
  {
    stream << "| "
           << std::setw(column_width[0]) << first_column
           << " | "
           << std::setw(column_width[1]) << second_column
           << " |"
           << std::endl;
  }

  template<typename Stream>
  void add_header(Stream  &stream)
  {
    stream << std::left << header << std::endl;
  }

} // internal



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
    n_maximum_levels = prm.get_integer("Maximum number of levels");

    adaptive_mesh_refinement = prm.get_bool("Adaptive mesh refinement");

    n_initial_global_refinements = prm.get_integer("Number of initial global refinements");

    n_initial_adaptive_refinements = prm.get_integer("Number of initial adaptive refinements");

    n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

    if (adaptive_mesh_refinement)
    {
      n_minimum_levels = prm.get_integer("Minimum number of levels");
      AssertThrow(n_minimum_levels > 0,
                  ExcMessage("Minimum number of levels must be larger than zero."));
      AssertThrow(n_minimum_levels <= n_maximum_levels ,
                  ExcMessage("Maximum number of levels must be larger equal "
                             "than the minimum number of levels."));

      adaptive_mesh_refinement_frequency = prm.get_integer("Adaptive mesh refinement frequency");

      cell_fraction_to_coarsen = prm.get_double("Fraction of cells set to coarsen");

      cell_fraction_to_refine = prm.get_double("Fraction of cells set to refine");

      const double total_cell_fraction_to_modify =
        cell_fraction_to_coarsen + cell_fraction_to_refine;

      AssertThrow(cell_fraction_to_coarsen >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_coarsen, 0));

      AssertThrow(cell_fraction_to_refine >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_refine, 0));

      AssertThrow(1.0 > total_cell_fraction_to_modify,
                  ExcMessage("The sum of the top and bottom fractions to "
                             "coarsen and refine may not exceed 1.0"));
    }

    n_initial_adaptive_refinements = prm.get_integer("Number of initial adaptive refinements");

    n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

    const unsigned int n_initial_refinements
    = n_initial_global_refinements + n_initial_adaptive_refinements
    + n_initial_boundary_refinements;

    if (adaptive_mesh_refinement)
      AssertThrow(n_minimum_levels <= n_initial_refinements,
                  ExcMessage("Number of initial refinements must be larger "
                             "equal than the minimum number of levels."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SpatialDiscretizationParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Refinement control parameters");
  internal::add_header(stream);

  internal::add_line(stream,
                     "Adaptive mesh refinement",
                     (prm.adaptive_mesh_refinement ? "True": "False"));
  if (prm.adaptive_mesh_refinement)
  {
    internal::add_line(stream,
                       "Adapt. mesh refinement frequency",
                       prm.adaptive_mesh_refinement_frequency);
    internal::add_line(stream,
                       "Fraction of cells set to coarsen", prm.cell_fraction_to_coarsen);
    internal::add_line(stream,
                       "Fraction of cells set to refine",
                       prm.cell_fraction_to_refine);
    internal::add_line(stream,
                       "Maximum number of levels", prm.n_maximum_levels);
    internal::add_line(stream,
                       "Minimum number of levels", prm.n_minimum_levels);
  }
  internal::add_line(stream,
                     "Number of initial adapt. refinements",
                     prm.n_initial_adaptive_refinements);
  internal::add_line(stream,
                     "Number of initial global refinements",
                     prm.n_initial_global_refinements);
  internal::add_line(stream,
                     "Number of initial boundary refinements",
                     prm.n_initial_boundary_refinements);

  internal::add_header(stream);

  return (stream);
}



OutputControlParameters::OutputControlParameters()
:
graphical_output_frequency(100),
terminal_output_frequency(100),
graphical_output_directory("./")
{}


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
  internal::add_header(stream);
  internal::add_line(stream, "Output control parameters");
  internal::add_header(stream);

  internal::add_line(stream,
                     "Graphical output frequency",
                     prm.graphical_output_frequency);
  internal::add_line(stream,
                     "Terminal output frequency",
                     prm.terminal_output_frequency);
  internal::add_line(stream,
                     "Graphical output directory",
                     prm.graphical_output_directory);

  internal::add_header(stream);

  return (stream);
}


PreconditionBaseParameters::PreconditionBaseParameters
(const std::string        &name,
 const PreconditionerType &type)
:
preconditioner_type(type),
preconditioner_name(name)
{}


void PreconditionBaseParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Preconditioner type",
                    "ILU",
                    Patterns::Selection("ILU|AMG|GMG|Jacobi|SSOR"));
}

PreconditionerType PreconditionBaseParameters::parse_preconditioner_type(const ParameterHandler &prm)
{
  std::string name = prm.get("Preconditioner type");

  PreconditionerType  type;

  if (name == "AMG")
    type = PreconditionerType::AMG;
  else if (name == "GMG")
    type = PreconditionerType::GMG;
  else if (name == "ILU")
    type = PreconditionerType::ILU;
  else if (name == "Jacobi")
    type = PreconditionerType::Jacobi;
  else if (name == "SSOR")
    type = PreconditionerType::SSOR;
  else
    AssertThrow(false, ExcMessage("Preconditioner type is unknown."));

  return (type);
}

void PreconditionBaseParameters::parse_parameters(const ParameterHandler &prm)
{
  preconditioner_name = prm.get("Preconditioner type");

  if (preconditioner_name == "AMG")
    preconditioner_type = PreconditionerType::AMG;
  else if (preconditioner_name == "GMG")
      preconditioner_type = PreconditionerType::GMG;
  else if (preconditioner_name == "ILU")
      preconditioner_type = PreconditionerType::ILU;
  else if (preconditioner_name == "Jacobi")
      preconditioner_type = PreconditionerType::Jacobi;
  else if (preconditioner_name == "SSOR")
      preconditioner_type = PreconditionerType::SSOR;
  else
    AssertThrow(false, ExcMessage("Preconditioner type is unknown."));
}

PreconditionRelaxationParameters::PreconditionRelaxationParameters()
:
PreconditionBaseParameters("Jacobi", PreconditionerType::Jacobi),
omega(1.0),
overlap(0),
n_sweeps(1)
{}

void PreconditionRelaxationParameters::declare_parameters(ParameterHandler &prm)
{
  PreconditionBaseParameters::declare_parameters(prm);

  prm.declare_entry("Relaxation parameter",
                    "1.0",
                    Patterns::Double());

  prm.declare_entry("Overlap",
                    "0",
                    Patterns::Integer());


  prm.declare_entry("Number of sweeps",
                    "1",
                    Patterns::Integer());
}


void PreconditionRelaxationParameters::parse_parameters(const ParameterHandler &prm)
{
  PreconditionBaseParameters::parse_parameters(prm);

  AssertThrow(preconditioner_type == PreconditionerType::Jacobi ||
              preconditioner_type == PreconditionerType::SSOR,
              ExcMessage("Unexpected preconditioner type in "
                         "PreconditionRelaxationParameters."));

  omega = prm.get_double("Relaxation parameter");
  AssertThrow(omega > 0.0, ExcLowerRangeType<double>(omega, 0.0));

  switch (preconditioner_type)
  {
    case PreconditionerType::Jacobi:
    {
      AssertThrow(omega <= 1.0, ExcLowerRangeType<double>(1.0, omega));
      // Print a warning if PETSc is used
      #ifdef USE_PETSC_LA
      ConditionalOStream  pcout(std::cout,
                                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
     pcout << std::endl << std::endl
             << "----------------------------------------------------"
             << std::endl
             << "Warning in the PreconditionRelaxationParameters: " << std::endl
             << "    The program runs using the linear algebra methods of the PETSc "
                    "library. Therefore the following parameter will not have an "
                    "influence on the configuration of the preconditioner:\n"
                    "   - Relaxation parameter (omega)."
             << std::endl
             << "----------------------------------------------------"
             << std::endl << std::endl;
      #endif
      break;
    }
    case PreconditionerType::SSOR:
    {
      AssertThrow(omega <= 2.0, ExcLowerRangeType<double>(2.0, omega));

      overlap = prm.get_integer("Overlap");
      n_sweeps = prm.get_integer("Number of sweeps");

      // Print a warning if PETSc is used
      #ifdef USE_PETSC_LA
      ConditionalOStream  pcout(std::cout,
                                 Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
       pcout << std::endl << std::endl
             << "----------------------------------------------------"
             << std::endl
             << "Warning in the PreconditionRelaxationParameters: " << std::endl
             << "    The program runs using the linear algebra methods of the PETSc "
                    "library. Therefore the following parameter will not have an "
                    "influence on the configuration of the preconditioner:\n"
                    "   - Overlap parameter,\n"
                    "   - Number of sweeps parameter."
             << std::endl
             << "----------------------------------------------------"
             << std::endl << std::endl;
      #endif
      break;
    }
    default:
      AssertThrow(false,
                  ExcMessage("Unexpected preconditioner type in "
                             "PreconditionRelaxationParameters."));
      break;
  }
}

template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionRelaxationParameters &prm)
{
  internal::add_line(stream, "Preconditioner", prm.preconditioner_name);
  internal::add_line(stream, "  Relaxation parameter", prm.omega);
  internal::add_line(stream, "  Overlap", prm.overlap);
  internal::add_line(stream, "  Number of sweeps", prm.n_sweeps);

  return (stream);
}


PreconditionILUParameters::PreconditionILUParameters()
:
PreconditionBaseParameters("ILU", PreconditionerType::ILU),
relative_tolerance(1.0),
absolute_tolerance(0.0),
fill(1),
overlap(1)
{
  #ifdef USE_PETSC_LA
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage("PreconditionILU using the PETSc library "
                           "only works in serial. Please choose a different "
                           "preconditioner."));
  #endif
}



void PreconditionILUParameters::declare_parameters(ParameterHandler &prm)
{
  PreconditionBaseParameters::declare_parameters(prm);

  prm.declare_entry("Relative tolerance",
                    "1.0",
                    Patterns::Double());

  prm.declare_entry("Absolute tolerance",
                    "0.0",
                    Patterns::Double());

  prm.declare_entry("Fill-in level",
                    "1",
                    Patterns::Integer());

  prm.declare_entry("Overlap",
                    "1",
                    Patterns::Integer());
}


void PreconditionILUParameters::parse_parameters(const ParameterHandler &prm)
{
  PreconditionBaseParameters::parse_parameters(prm);

  AssertThrow(preconditioner_type == PreconditionerType::ILU,
              ExcMessage("Unexpected preconditioner type in "
                         "PreconditionILUParameters."));

  relative_tolerance = prm.get_double("Relative tolerance");
  AssertThrow(relative_tolerance >= 1.0,
              ExcLowerRangeType<double>(relative_tolerance, 1.0));

  absolute_tolerance = prm.get_double("Absolute tolerance");
  AssertThrow(absolute_tolerance >= 0,
              ExcLowerRangeType<double>(absolute_tolerance, 0.0));

  fill = prm.get_integer("Fill-in level");

  overlap = prm.get_integer("Overlap");

  // Print a warning if PETSc is used
  #ifdef USE_PETSC_LA
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
   pcout << std::endl << std::endl
         << "----------------------------------------------------"
         << std::endl
         << "Warning in the PreconditionILUParameters: " << std::endl
         << "    The program runs using the linear algebra methods of the PETSc "
                "library. Therefore the following parameter will not have an "
                "influence on the configuration of the preconditioner:\n"
                "   - Overlap,\n"
                "   - Absolute tolerance,\n"
                "   - Relative tolerance."
         << std::endl
         << "----------------------------------------------------"
         << std::endl << std::endl;
  #endif
}


template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionILUParameters &prm)
{
  internal::add_line(stream, "Preconditioner", "ILU");
  internal::add_line(stream, "  Fill-in level", prm.fill);
  internal::add_line(stream, "  Overlap", prm.overlap);
  internal::add_line(stream, "  Relative tolerance", prm.relative_tolerance);
  internal::add_line(stream, "  Absolute tolerance", prm.absolute_tolerance);

  return (stream);
}


PreconditionAMGParameters::PreconditionAMGParameters()
:
PreconditionBaseParameters("AMG", PreconditionerType::AMG),
strong_threshold(0.2),
elliptic(false),
higher_order_elements(false),
n_cycles(1),
aggregation_threshold(1e-4)
{}


void PreconditionAMGParameters::declare_parameters(ParameterHandler &prm)
{
  PreconditionBaseParameters::declare_parameters(prm);

  prm.declare_entry("Strong threshold (PETSc only)",
                    "0.2",
                    Patterns::Double(0.0));

  prm.declare_entry("Elliptic",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Number of cycles",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("Aggregation threshold",
                    "1.0e-4",
                    Patterns::Double(0.0));
}


void PreconditionAMGParameters::parse_parameters(const ParameterHandler &prm)
{
  PreconditionBaseParameters::parse_parameters(prm);

  AssertThrow(preconditioner_type == PreconditionerType::AMG,
              ExcMessage("Unexpected preconditioner type in PreconditionAMGParameters."));

  strong_threshold = prm.get_double("Strong threshold (PETSc only)");
  AssertThrow(strong_threshold > 0.0,
              ExcLowerRangeType<double>(strong_threshold, 0.0));
  AssertIsFinite(strong_threshold);

  elliptic = prm.get_bool("Elliptic");

  n_cycles = prm.get_integer("Number of cycles");
  AssertThrow(n_cycles > 0,
              ExcLowerRange(n_cycles, 1));
  AssertIsFinite(n_cycles);

  aggregation_threshold = prm.get_double("Aggregation threshold");
  AssertThrow(aggregation_threshold > 0.0,
              ExcLowerRangeType<double>(aggregation_threshold, 0.0));
  AssertIsFinite(aggregation_threshold);
}


template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionAMGParameters &prm)
{
  internal::add_line(stream, "Preconditioner", "AMG");
  internal::add_line(stream, "  Strong threshold", prm.strong_threshold);
  internal::add_line(stream, "  Elliptic", (prm.elliptic? "true": "false"));
  internal::add_line(stream, "  Number of cycles", prm.n_cycles);
  internal::add_line(stream, "  Aggregation threshold", prm.aggregation_threshold);

  return (stream);
}


LinearSolverParameters::LinearSolverParameters(const std::string &name)
:
relative_tolerance(1e-6),
absolute_tolerance(1e-9),
n_maximum_iterations(50),
preconditioner_parameters_ptr(nullptr),
solver_name(name)
{}





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

  prm.enter_subsection("Preconditioner parameters");
  {

    PreconditionRelaxationParameters::declare_parameters(prm);

    PreconditionAMGParameters::declare_parameters(prm);

    PreconditionILUParameters::declare_parameters(prm);
  }
  prm.leave_subsection();
}



void LinearSolverParameters::parse_parameters(ParameterHandler &prm)
{
  n_maximum_iterations = prm.get_integer("Maximum number of iterations");
  AssertThrow(n_maximum_iterations > 0, ExcLowerRange(n_maximum_iterations, 0));

  relative_tolerance = prm.get_double("Relative tolerance");
  AssertThrow(relative_tolerance > 0, ExcLowerRange(relative_tolerance, 0));

  absolute_tolerance = prm.get_double("Absolute tolerance");
  AssertThrow(relative_tolerance > absolute_tolerance,
              ExcLowerRangeType<double>(relative_tolerance , absolute_tolerance));

  prm.enter_subsection("Preconditioner parameters");
  {
    const PreconditionerType preconditioner_type =
        PreconditionBaseParameters::parse_preconditioner_type(prm);

    switch (preconditioner_type)
    {
      case PreconditionerType::AMG:
        preconditioner_parameters_ptr
          = std::make_shared<PreconditionAMGParameters>();
        static_cast<PreconditionAMGParameters*>(preconditioner_parameters_ptr.get())
            ->parse_parameters(prm);
        break;
      case PreconditionerType::ILU:
        preconditioner_parameters_ptr
          = std::make_shared<PreconditionILUParameters>();
        static_cast<PreconditionILUParameters*>(preconditioner_parameters_ptr.get())
            ->parse_parameters(prm);
        break;
      case PreconditionerType::Jacobi:
        preconditioner_parameters_ptr
          = std::make_shared<PreconditionJacobiParameters>();
        static_cast<PreconditionJacobiParameters*>(preconditioner_parameters_ptr.get())
            ->parse_parameters(prm);
        break;
      case PreconditionerType::SSOR:
        preconditioner_parameters_ptr
          = std::make_shared<PreconditionSSORParameters>();
        static_cast<PreconditionSSORParameters*>(preconditioner_parameters_ptr.get())
            ->parse_parameters(prm);
        break;
      case PreconditionerType::GMG:
        AssertThrow(false, ExcNotImplemented());
        break;
      default:
        AssertThrow(false, ExcMessage("Preconditioner type is unknown."));
        break;
    }
  }
  prm.leave_subsection();
}


template<typename Stream>
Stream& operator<<(Stream &stream, const LinearSolverParameters &prm)
{


  internal::add_header(stream);
  internal::add_line(stream, "Linear solver parameters" +
                              (prm.solver_name != "default" ?
                                " - " + prm.solver_name : ""));
  internal::add_header(stream);

  internal::add_line(stream,
                     "Maximum number of iterations",
                     prm.n_maximum_iterations);
  internal::add_line(stream,
                     "Relative tolerance",
                     prm.relative_tolerance);
  internal::add_line(stream,
                     "Absolute tolerance",
                     prm.absolute_tolerance);

  switch (prm.preconditioner_parameters_ptr->preconditioner_type)
  {
    case PreconditionerType::AMG:
      stream << *static_cast<const PreconditionAMGParameters*>(prm.preconditioner_parameters_ptr.get());
      break;
    case PreconditionerType::ILU:
      stream << *static_cast<const PreconditionILUParameters*>(prm.preconditioner_parameters_ptr.get());
      break;
    case PreconditionerType::Jacobi:
      stream << *static_cast<const PreconditionJacobiParameters*>(prm.preconditioner_parameters_ptr.get());
      break;
    case PreconditionerType::SSOR:
      stream << *static_cast<const PreconditionSSORParameters*>(prm.preconditioner_parameters_ptr.get());
      break;
    case PreconditionerType::GMG:
      AssertThrow(false, ExcNotImplemented());
      break;
    default:
      AssertThrow(false, ExcMessage("Preconditioner type is unknown."));
      break;
  }

  stream << "\r";

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



void DimensionlessNumbers::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Problem type",
                    "hydrodynamic",
                    Patterns::Selection("hydrodynamic|"
                                         "heat_convection_diffusion |"
                                         "boussinesq |"
                                         "rotating_boussinesq |"
                                         "rotating_magnetohydrodynamic"));

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
}


void DimensionlessNumbers::declare_parameters
(ParameterHandler &prm, const ProblemType  problem_type)
{
  prm.declare_entry("Problem type",
                    "hydrodynamic",
                    Patterns::Selection("hydrodynamic|"
                                         "heat_convection_diffusion |"
                                         "boussinesq |"
                                         "rotating_boussinesq |"
                                         "rotating_magnetohydrodynamic"));

  prm.enter_subsection("Dimensionless numbers");
  {
    switch (problem_type)
    {
      case ProblemType::hydrodynamic:
        prm.declare_entry("Reynolds number",
                          "1.0",
                          Patterns::Double(0));
        break;
      case ProblemType::heat_convection_diffusion:
        prm.declare_entry("Peclet number",
                          "1.0",
                          Patterns::Double(0));
        break;
      case ProblemType::boussinesq:
        prm.declare_entry("Prandtl number",
                          "1.0",
                          Patterns::Double(0));


        prm.declare_entry("Rayleigh number",
                          "1.0",
                          Patterns::Double(0));
        break;
      case ProblemType::rotating_boussinesq:
        prm.declare_entry("Prandtl number",
                          "1.0",
                          Patterns::Double(0));
        prm.declare_entry("Rayleigh number",
                          "1.0",
                          Patterns::Double(0));
        prm.declare_entry("Ekman number",
                          "1.0",
                          Patterns::Double(0));
        break;
      case ProblemType::rotating_magnetohydrodynamic:
        prm.declare_entry("Prandtl number",
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
        break;
      default:
        AssertThrow(false,
                    ExcMessage("Unexpected identifier for the problem type."));
        break;
    }
  }
  prm.leave_subsection();
}



void DimensionlessNumbers::parse_parameters(ParameterHandler &prm)
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
                ExcMessage("Unexpected identifier for the problem type."));

  prm.enter_subsection("Dimensionless numbers");
  {
    switch (problem_type)
    {
      case ProblemType::hydrodynamic:
        Re = prm.get_double("Reynolds number");
        AssertThrow(Re > 0.0, ExcLowerRangeType<double>(Re, 0.0));
        AssertIsFinite(Re);
        break;
      case ProblemType::heat_convection_diffusion:
        Pe = prm.get_double("Peclet number");
        AssertThrow(Pe > 0.0, ExcLowerRangeType<double>(Pe, 0.0));
        AssertIsFinite(Pe);
        break;
      case ProblemType::boussinesq:
        Pr = prm.get_double("Prandtl number");
        AssertThrow(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
        AssertIsFinite(Pr);

        Ra = prm.get_double("Rayleigh number");
        AssertThrow(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
        AssertIsFinite(Ra);
        break;
      case ProblemType::rotating_boussinesq:
        Pr = prm.get_double("Prandtl number");
        AssertThrow(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
        AssertIsFinite(Pr);

        Ra = prm.get_double("Rayleigh number");
        AssertThrow(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
        AssertIsFinite(Ra);

        Ek = prm.get_double("Ekman number");
        AssertThrow(Ek > 0.0, ExcLowerRangeType<double>(Ek, 0.0));
        AssertIsFinite(Ek);

        break;
      case ProblemType::rotating_magnetohydrodynamic:
        Pr = prm.get_double("Prandtl number");
        AssertThrow(Pr > 0.0, ExcLowerRangeType<double>(Pr, 0.0));
        AssertIsFinite(Pr);

        Ra = prm.get_double("Rayleigh number");
        AssertThrow(Ra > 0.0, ExcLowerRangeType<double>(Ra, 0.0));
        AssertIsFinite(Ra);

        Ek = prm.get_double("Ekman number");
        AssertThrow(Ek > 0.0, ExcLowerRangeType<double>(Ek, 0.0));
        AssertIsFinite(Ek);

        Pm = prm.get_double("magnetic Prandtl number");
        AssertThrow(Pm > 0.0, ExcLowerRangeType<double>(Pm, 0.0));
        AssertIsFinite(Pm);
        break;
      default:
        AssertThrow(false,
                    ExcMessage("Unexpected identifier for the problem type."));
        break;
    }
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Dimensionless numbers");
  internal::add_header(stream);

  switch (prm.problem_type)
  {
    case ProblemType::hydrodynamic:
      internal::add_line(stream, "Reynolds number", prm.Re);
      break;
    case ProblemType::heat_convection_diffusion:
      internal::add_line(stream, "Peclet number", prm.Pe);
      break;
    case ProblemType::boussinesq:
      internal::add_line(stream, "Prandtl number", prm.Pr);
      internal::add_line(stream, "Rayleigh number", prm.Ra);
      break;
    case ProblemType::rotating_boussinesq:
      internal::add_line(stream, "Prandtl number", prm.Pr);
      internal::add_line(stream, "Rayleigh number", prm.Ra);
      internal::add_line(stream, "Ekman number", prm.Ek);
      break;
    case ProblemType::rotating_magnetohydrodynamic:
      internal::add_line(stream, "Prandtl number", prm.Pr);
      internal::add_line(stream, "Rayleigh number", prm.Ra);
      internal::add_line(stream, "Ekman number", prm.Ek);
      internal::add_line(stream, "magnetic Prandtl number", prm.Pm);
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the problem type."));
      break;
  }

  internal::add_header(stream);

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
diffusion_step_solver_parameters("Diffusion step"),
projection_step_solver_parameters("Projection step"),
correction_step_solver_parameters("Correction step"),
poisson_prestep_solver_parameters("Poisson pre-step"),
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
    AssertThrow(preconditioner_update_frequency > 0,
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
  internal::add_header(stream);
  internal::add_line(stream, "Navier-Stokes discretization parameters");
  internal::add_header(stream);

  switch (prm.pressure_correction_scheme)
  {
    case PressureCorrectionScheme::standard:
      internal::add_line(stream, "Incremental pressure-correction scheme", "standard");
      break;
    case PressureCorrectionScheme::rotational:
      internal::add_line(stream, "Incremental pressure-correction scheme", "rotational");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                               "incremental pressure-correction scheme"));
      break;
  }

  switch (prm.convective_term_weak_form) {
    case ConvectiveTermWeakForm::standard:
      internal::add_line(stream, "Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      internal::add_line(stream, "Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      internal::add_line(stream, "Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      internal::add_line(stream, "Convective term weak form", "skew-symmetric");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                               "weak form of the convective term."));
      break;
  }

  switch (prm.convective_term_time_discretization) {
    case ConvectiveTermTimeDiscretization::semi_implicit:
      internal::add_line(stream, "Convective temporal form", "semi-implicit");
      break;
    case ConvectiveTermTimeDiscretization::fully_explicit:
      internal::add_line(stream, "Convective temporal form", "explicit");
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unexpected type identifier for the "
                             "time discretization of the convective term."));
      break;
  }

  internal::add_line(stream, "Preconditioner update frequency", prm.preconditioner_update_frequency);

  stream << prm.diffusion_step_solver_parameters;

  stream << "\r";

  stream << prm.projection_step_solver_parameters;

  stream << "\r";

  stream << prm.correction_step_solver_parameters;

  stream << "\r";

  stream << prm.poisson_prestep_solver_parameters;

  stream << "\r";

  internal::add_header(stream);

  return (stream);
}



HeatEquationParameters::HeatEquationParameters()
:
convective_term_weak_form(ConvectiveTermWeakForm::skewsymmetric),
convective_term_time_discretization(ConvectiveTermTimeDiscretization::semi_implicit),
C4(1.0),
solver_parameters("Heat equation"),
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
    AssertThrow(preconditioner_update_frequency > 0,
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
  internal::add_header(stream);
  internal::add_line(stream, "Heat equation solver parameters");
  internal::add_header(stream);

  switch (prm.convective_term_weak_form) {
    case ConvectiveTermWeakForm::standard:
      internal::add_line(stream, "Convective term weak form", "standard");
      break;
    case ConvectiveTermWeakForm::rotational:
      internal::add_line(stream, "Convective term weak form", "rotational");
      break;
    case ConvectiveTermWeakForm::divergence:
      internal::add_line(stream, "Convective term weak form", "divergence");
      break;
    case ConvectiveTermWeakForm::skewsymmetric:
      internal::add_line(stream, "Convective term weak form", "skew-symmetric");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                               "weak form of the convective term."));
      break;
  }

  switch (prm.convective_term_time_discretization) {
    case ConvectiveTermTimeDiscretization::semi_implicit:
      internal::add_line(stream, "Convective temporal form", "semi-implicit");
      break;
    case ConvectiveTermTimeDiscretization::fully_explicit:
      internal::add_line(stream, "Convective temporal form", "explicit");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                               "time discretization of the convective term."));
      break;
  }

  internal::add_line(stream,
                     "Preconditioner update frequency",
                     prm.preconditioner_update_frequency);

  stream << prm.solver_parameters;

  stream << "\r";

  internal::add_header(stream);

  return (stream);
}




ProblemBaseParameters::ProblemBaseParameters()
:
OutputControlParameters(),
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
verbose(false),
spatial_discretization_parameters(),
time_discretization_parameters()
{}



void ProblemBaseParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Spatial dimension",
                    "2",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Polynomial degree",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Apply to interior cells",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());

  OutputControlParameters::declare_parameters(prm);

  SpatialDiscretizationParameters::declare_parameters(prm);

  TimeDiscretization::
  TimeDiscretizationParameters::declare_parameters(prm);

}



void ProblemBaseParameters::parse_parameters(ParameterHandler &prm)
{
  dim = prm.get_integer("Spatial dimension");
  AssertThrow(dim > 0, ExcLowerRange(dim, 0) );
  AssertThrow(dim <= 3, ExcMessage("The spatial dimension is larger than three.") );

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  mapping_interior_cells = prm.get_bool("Mapping - Apply to interior cells");

  verbose = prm.get_bool("Verbose");

  OutputControlParameters::parse_parameters(prm);

  spatial_discretization_parameters.parse_parameters(prm);

  time_discretization_parameters.parse_parameters(prm);
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemBaseParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  internal::add_line(stream, "Spatial dimension", prm.dim);

  {
    std::stringstream strstream;

    strstream << "MappingQ<" << std::to_string(prm.dim) << ">"
              << "(" << std::to_string(prm.mapping_degree) << ")";
    internal::add_line(stream, "Mapping", strstream.str().c_str());
  }

  internal::add_line(stream,
                     "Mapping - Apply to interior cells",
                     (prm.mapping_interior_cells ? "true" : "false"));


  internal::add_line(stream, "Verbose", (prm.verbose? "true": "false"));

  stream << static_cast<const OutputControlParameters &>(prm);

  stream << "\r";

  stream << prm.spatial_discretization_parameters;

  stream << "\r";

  stream << prm.time_discretization_parameters;

  stream << "\r";

  return (stream);
}



HydrodynamicProblemParameters::HydrodynamicProblemParameters()
:
ProblemBaseParameters(),
DimensionlessNumbers(),
problem_type(ProblemType::hydrodynamic),
fe_degree_pressure(1),
fe_degree_velocity(2),
navier_stokes_parameters()
{}



HydrodynamicProblemParameters::HydrodynamicProblemParameters
(const std::string &parameter_filename)
:
HydrodynamicProblemParameters()
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

  switch (problem_type)
  {
    case ProblemType::hydrodynamic:
      navier_stokes_parameters.C1 = 0.0;
      navier_stokes_parameters.C2 = 1.0/Re;
      navier_stokes_parameters.C3 = 0.0;
      navier_stokes_parameters.C5 = 0.0;
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the problem type."));
      break;
  }

  AssertIsFinite(navier_stokes_parameters.C1);
  AssertIsFinite(navier_stokes_parameters.C2);
  AssertIsFinite(navier_stokes_parameters.C3);
  AssertIsFinite(navier_stokes_parameters.C5);
}




void HydrodynamicProblemParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("FE's polynomial degree - Pressure (Taylor-Hood)",
                    "1",
                    Patterns::Integer(1));

  ProblemBaseParameters::declare_parameters(prm);

  DimensionlessNumbers::declare_parameters(prm, ProblemType::hydrodynamic);

  NavierStokesParameters::declare_parameters(prm);
}




void HydrodynamicProblemParameters::parse_parameters(ParameterHandler &prm)
{
  ProblemBaseParameters::parse_parameters(prm);

  DimensionlessNumbers::parse_parameters(prm);

  fe_degree_pressure = prm.get_integer("FE's polynomial degree - Pressure (Taylor-Hood)");
  AssertThrow(fe_degree_pressure > 0,
              ExcLowerRange(fe_degree_pressure, 0));

  fe_degree_velocity = fe_degree_pressure + 1;

  navier_stokes_parameters.parse_parameters(prm);
}




template<typename Stream>
Stream& operator<<(Stream &stream, const HydrodynamicProblemParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  stream << static_cast<const ProblemBaseParameters &>(prm);

  stream << "\r";

  internal::add_line(stream, "Problem type", "hydrodynamic");

  {
    std::string fe_velocity = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_velocity) + ")^" + std::to_string(prm.dim);
    internal::add_line(stream, "Finite Element - Velocity", fe_velocity);

    std::string fe_pressure = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_pressure) + ")";
    internal::add_line(stream, "Finite Element - Pressure", fe_pressure);
  }

  stream << static_cast<const DimensionlessNumbers &>(prm);

  stream << "\r";

  stream << prm.navier_stokes_parameters;

  stream << "\r";

  return (stream);
}



BoussinesqProblemParameters::BoussinesqProblemParameters()
:
ProblemBaseParameters(),
DimensionlessNumbers(),
problem_type(ProblemType::boussinesq),
fe_degree_pressure(1),
fe_degree_velocity(2),
fe_degree_temperature(2),
navier_stokes_parameters(),
heat_equation_parameters()
{}



BoussinesqProblemParameters::BoussinesqProblemParameters
(const std::string &parameter_filename)
:
BoussinesqProblemParameters()
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

  switch (problem_type)
  {
    case ProblemType::boussinesq:
      navier_stokes_parameters.C1 = 0.0;
      navier_stokes_parameters.C2 = std::sqrt(Pr/Ra);
      navier_stokes_parameters.C3 = 1.0;
      heat_equation_parameters.C4 = 1.0/std::sqrt(Ra*Pr);
      navier_stokes_parameters.C5 = 0.0;
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the problem type."));
      break;
  }

  AssertIsFinite(navier_stokes_parameters.C1);
  AssertIsFinite(navier_stokes_parameters.C2);
  AssertIsFinite(navier_stokes_parameters.C3);
  AssertIsFinite(heat_equation_parameters.C4);
  AssertIsFinite(navier_stokes_parameters.C5);
}




void BoussinesqProblemParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("FE's polynomial degree - Pressure (Taylor-Hood)",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("FE's polynomial degree - Temperature",
                    "1",
                    Patterns::Integer(1));

  ProblemBaseParameters::declare_parameters(prm);

  DimensionlessNumbers::declare_parameters(prm, ProblemType::hydrodynamic);

  NavierStokesParameters::declare_parameters(prm);

  HeatEquationParameters::declare_parameters(prm);
}




void BoussinesqProblemParameters::parse_parameters(ParameterHandler &prm)
{
  ProblemBaseParameters::parse_parameters(prm);

  DimensionlessNumbers::parse_parameters(prm);

  fe_degree_pressure = prm.get_integer("FE's polynomial degree - Pressure (Taylor-Hood)");
  AssertThrow(fe_degree_pressure > 0,
              ExcLowerRange(fe_degree_pressure, 0));

  fe_degree_velocity = fe_degree_pressure + 1;

  fe_degree_temperature = prm.get_integer("FE's polynomial degree - Temperature");
  AssertThrow(fe_degree_temperature > 0,
              ExcLowerRange(fe_degree_temperature, 0));

  navier_stokes_parameters.parse_parameters(prm);

  heat_equation_parameters.parse_parameters(prm);
}




template<typename Stream>
Stream& operator<<(Stream &stream, const BoussinesqProblemParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  stream << static_cast<const ProblemBaseParameters &>(prm);

  stream << "\r";

  internal::add_line(stream, "Problem type", "Boussinesq");

  {
    std::string fe_velocity = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_velocity) + ")^" + std::to_string(prm.dim);
    internal::add_line(stream, "Finite Element - Velocity", fe_velocity);

    std::string fe_pressure = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_pressure) + ")";
    internal::add_line(stream, "Finite Element - Pressure", fe_pressure);

    std::string fe_temperature = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_temperature) + ")";
    internal::add_line(stream, "Finite Element - Temperature", fe_temperature);
  }

  stream << static_cast<const DimensionlessNumbers &>(prm);

  stream << "\r";

  stream << prm.navier_stokes_parameters;

  stream << "\r";

  stream << prm.heat_equation_parameters;

  stream << "\r";

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

  ConvergenceTest::ConvergenceTestParameters::declare_parameters(prm);

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
  AssertThrow(dim > 0, ExcLowerRange(dim, 0) );
  AssertThrow(dim <= 3, ExcMessage("The spatial dimension are larger than three.") );

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  mapping_interior_cells = prm.get_bool("Mapping - Apply to interior cells");

  fe_degree_pressure = prm.get_integer("FE's polynomial degree - Pressure (Taylor-Hood)");
  AssertThrow(fe_degree_pressure > 0,
              ExcLowerRange(fe_degree_pressure, 0));


  fe_degree_temperature = prm.get_integer("FE's polynomial degree - Temperature");
  AssertThrow(fe_degree_temperature > 0,
              ExcLowerRange(fe_degree_temperature, 0));

  fe_degree_velocity = fe_degree_pressure + 1;

  verbose = prm.get_bool("Verbose");

  OutputControlParameters::parse_parameters(prm);

  DimensionlessNumbers::parse_parameters(prm);

  if (flag_convergence_test)
    convergence_test_parameters.parse_parameters(prm);

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
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  switch (prm.problem_type)
  {
    case ProblemType::hydrodynamic:
      internal::add_line(stream, "Problem type", "hydrodynamic");
      break;
    case ProblemType::heat_convection_diffusion:
      internal::add_line(stream, "Problem type", "heat_convection_diffusion");
      break;
    case ProblemType::boussinesq:
      internal::add_line(stream, "Problem type", "boussinesq");
      break;
    case ProblemType::rotating_boussinesq:
      internal::add_line(stream, "Problem type", "rotating_boussinesq");
      break;
    case ProblemType::rotating_magnetohydrodynamic:
      internal::add_line(stream, "Problem type", "rotating_magnetohydrodynamic");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                               "problem type"));
      break;
  }

  internal::add_line(stream, "Spatial dimension", prm.dim);

  {
    std::stringstream strstream;

    strstream << "MappingQ<" << std::to_string(prm.dim) << ">"
              << "(" << std::to_string(prm.mapping_degree) << ")";
    internal::add_line(stream, "Mapping", strstream.str().c_str());
  }

  internal::add_line(stream,
                     "Mapping - Apply to interior cells",
                     (prm.mapping_interior_cells ? "true" : "false"));

  if (prm.problem_type != ProblemType::heat_convection_diffusion)
  {
    std::string fe_velocity = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_velocity) + ")^" + std::to_string(prm.dim);
    std::string fe_pressure = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_pressure) + ")";
    internal::add_line(stream, "Finite Element - Velocity", fe_velocity);
    internal::add_line(stream, "Finite Element - Pressure", fe_pressure);
  }

  if (prm.problem_type != ProblemType::hydrodynamic)
  {
    std::string fe_temperature = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree_temperature) + ")";
    internal::add_line(stream,
                       "Finite Element - Temperature",
                       fe_temperature);
  }

  internal::add_line(stream, "Verbose", (prm.verbose? "true": "false"));

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

  stream << std::endl << std::endl;

  stream << "+----------+----------+----------+----------+----------+----------+\n"
         << "|    C1    |    C2    |    C3    |    C4    |    C5    |    C6    |\n"
         << "+----------+----------+----------+----------+----------+----------+\n";

  stream << "| ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.navier_stokes_parameters.C1;
  stream << " | ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.navier_stokes_parameters.C2;
  stream << " | ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.navier_stokes_parameters.C3;
  stream << " | ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.heat_equation_parameters.C4;
  stream << " | ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.navier_stokes_parameters.C5;
  stream << " | ";
  stream << std::setw(8) << std::setprecision(1) << std::scientific << std::right << prm.navier_stokes_parameters.C6;
  stream << " |";

  stream << "\n+----------+----------+----------+----------+----------+----------+\n";

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
(std::ostream &, const RMHD::RunTimeParameters::PreconditionRelaxationParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::PreconditionRelaxationParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::PreconditionILUParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::PreconditionILUParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::PreconditionAMGParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::PreconditionAMGParameters &);

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
(std::ostream &, const RMHD::RunTimeParameters::ProblemBaseParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ProblemBaseParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::HydrodynamicProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::HydrodynamicProblemParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::BoussinesqProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::BoussinesqProblemParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::ProblemParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ProblemParameters &);
