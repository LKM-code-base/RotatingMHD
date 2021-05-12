/*
 * linear_solver_parameters.cc
 *
 *  Created on: May 5, 2021
 *      Author: sg
 */

#include <rotatingMHD/linear_solver_parameters.h>

#include <deal.II/base/conditional_ostream.h>

namespace RMHD
{

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

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

  return (stream);
}

} // namespace RunTimeParameters

} // namespace RMHD

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

