#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>
#include <cmath>

namespace RMHD
{

template <int dim>
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::HeatEquationParameters  &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::FE_ScalarField<dim>>     &temperature,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
flag_matrices_were_updated(true)
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  Assert(parameters.C4 > 0.0,
         ExcLowerRangeType<double>(parameters.C4, 0.0));
  AssertIsFinite(parameters.C4);

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
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::HeatEquationParameters  &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::FE_ScalarField<dim>>     &temperature,
 std::shared_ptr<Entities::FE_VectorField<dim>>     &velocity,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
velocity(velocity),
flag_matrices_were_updated(true)
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));

  Assert(parameters.C4 > 0.0,
         ExcLowerRangeType<double>(parameters.C4, 0.0));
  AssertIsFinite(parameters.C4);

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
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::HeatEquationParameters  &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::FE_ScalarField<dim>>     &temperature,
 std::shared_ptr<TensorFunction<1, dim>>          &velocity,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
velocity_function_ptr(velocity),
flag_matrices_were_updated(true)
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity function's shared pointer has not be"
                    " initialized."));

  Assert(parameters.C4 > 0.0,
         ExcLowerRangeType<double>(parameters.C4, 0.0));
  AssertIsFinite(parameters.C4);

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

}  // namespace RMHD

// explicit instantiations

template class RMHD::HeatEquation<2>;
template class RMHD::HeatEquation<3>;
