#include <rotatingMHD/convection_diffusion_solver.h>

#include <deal.II/fe/mapping_q.h>

#include <fstream>

namespace RMHD
{

ConvectionDiffusionParameters::ConvectionDiffusionParameters()
:
convective_term_weak_form(RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric),
convective_term_time_discretization(RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit),
equation_coefficient(1.0),
solver_parameters("Heat equation"),
preconditioner_update_frequency(10),
verbose(false)
{}



ConvectionDiffusionParameters::ConvectionDiffusionParameters
(const std::string &parameter_filename)
:
ConvectionDiffusionParameters()
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



void ConvectionDiffusionParameters::declare_parameters(ParameterHandler &prm)
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
      RunTimeParameters::LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void ConvectionDiffusionParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Heat equation solver parameters");
  {
    const std::string str_convective_term_weak_form(prm.get("Convective term weak form"));

    if (str_convective_term_weak_form == std::string("standard"))
      convective_term_weak_form
      = RunTimeParameters::ConvectiveTermWeakForm::standard;
    else if (str_convective_term_weak_form == std::string("skew-symmetric"))
      convective_term_weak_form
      = RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric;
    else if (str_convective_term_weak_form == std::string("divergence"))
      convective_term_weak_form
      = RunTimeParameters::ConvectiveTermWeakForm::divergence;
    else if (str_convective_term_weak_form == std::string("rotational"))
      convective_term_weak_form
      = RunTimeParameters::ConvectiveTermWeakForm::rotational;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the weak form "
                             "of the convective term."));

    const std::string str_convective_term_time_discretization(prm.get("Convective term time discretization"));

    if (str_convective_term_time_discretization == std::string("semi-implicit"))
      convective_term_time_discretization
      = RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit;
    else if (str_convective_term_time_discretization == std::string("explicit"))
      convective_term_time_discretization
      = RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit;
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
Stream& operator<<(Stream &stream, const ConvectionDiffusionParameters &prm)
{
  using namespace RunTimeParameters::internal;
  add_header(stream);
  add_line(stream, "Heat equation solver parameters");
  add_header(stream);

  switch (prm.convective_term_weak_form) {
    case RunTimeParameters::ConvectiveTermWeakForm::standard:
      add_line(stream, "Convective term weak form", "standard");
      break;
    case RunTimeParameters::ConvectiveTermWeakForm::rotational:
      add_line(stream, "Convective term weak form", "rotational");
      break;
    case RunTimeParameters::ConvectiveTermWeakForm::divergence:
      add_line(stream, "Convective term weak form", "divergence");
      break;
    case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
      add_line(stream, "Convective term weak form", "skew-symmetric");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "weak form of the convective term."));
      break;
  }

  switch (prm.convective_term_time_discretization) {
    case RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit:
      add_line(stream, "Convective temporal form", "semi-implicit");
      break;
    case RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit:
      add_line(stream, "Convective temporal form", "explicit");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "time discretization of the convective term."));
      break;
  }

  add_line(stream,
           "Preconditioner update frequency",
           prm.preconditioner_update_frequency);

  stream << prm.solver_parameters;

  stream << "\r";

  add_header(stream);

  return (stream);
}



template <int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionParameters              &parameters,
 const TimeDiscretization::VSIMEXMethod           &time_stepping,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &temperature,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
phi(temperature),
flag_matrices_were_updated(true)
{
  Assert(phi.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  Assert(parameters.equation_coefficient > 0.0,
         ExcLowerRangeType<double>(parameters.equation_coefficient, 0.0));
  AssertIsFinite(parameters.equation_coefficient);

  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));

  // Explicitly set the shared_ptr's to zero.
  source_term_ptr       = nullptr;
  velocity_function_ptr = nullptr;
  this->velocity        = nullptr;
}



template <int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionParameters              &parameters,
 const TimeDiscretization::VSIMEXMethod           &time_stepping,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &temperature,
 std::shared_ptr<Entities::VectorEntity<dim>>     &velocity,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
phi(temperature),
velocity(velocity),
flag_matrices_were_updated(true)
{
  Assert(phi.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));

  Assert(parameters.equation_coefficient > 0.0,
         ExcLowerRangeType<double>(parameters.equation_coefficient, 0.0));
  AssertIsFinite(parameters.equation_coefficient);

  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));

  // Explicitly set the shared_ptr's to zero.
  source_term_ptr       = nullptr;
  velocity_function_ptr = nullptr;
}



template <int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionParameters              &parameters,
 const TimeDiscretization::VSIMEXMethod           &time_stepping,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &temperature,
 std::shared_ptr<TensorFunction<1, dim>>          &velocity,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
phi(temperature),
velocity_function_ptr(velocity),
flag_matrices_were_updated(true)
{
  Assert(phi.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity function's shared pointer has not be"
                    " initialized."));

  Assert(parameters.equation_coefficient > 0.0,
         ExcLowerRangeType<double>(parameters.equation_coefficient, 0.0));
  AssertIsFinite(parameters.equation_coefficient);

  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));

  // Explicitly set the shared_ptr's to zero.
  source_term_ptr = nullptr;
  this->velocity  = nullptr;
}

template <int dim>
void ConvectionDiffusionSolver<dim>::clear()
{
  velocity_function_ptr = nullptr;
  source_term_ptr = nullptr;

  // preconditioners
  preconditioner.reset();

  // velocity matrices
  system_matrix.clear();
  mass_plus_stiffness_matrix.clear();
  stiffness_matrix.clear();
  advection_matrix.clear();
  mass_matrix.clear();

  // velocity vectors
  rhs.clear();

  // internal entity
  phi->clear();
  flag_matrices_were_updated = true;

}

}  // namespace RMHD

// explicit instantiations
template std::ostream & RMHD::operator<<
(std::ostream &, const RMHD::ConvectionDiffusionParameters &);
template dealii::ConditionalOStream & RMHD::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvectionDiffusionParameters &);

template class RMHD::ConvectionDiffusionSolver<2>;
template class RMHD::ConvectionDiffusionSolver<3>;
