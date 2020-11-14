#include <rotatingMHD/heat_equation.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>
#include <cmath>

namespace RMHD
{

template <int dim>
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::ParameterSet        &parameters,
 TimeDiscretization::VSIMEXMethod             &time_stepping,
 std::shared_ptr<Entities::ScalarEntity<dim>> &temperature,
 std::shared_ptr<Entities::VectorEntity<dim>> velocity,
 const std::shared_ptr<Mapping<dim>>          external_mapping,
 const std::shared_ptr<ConditionalOStream>    external_pcout,
 const std::shared_ptr<TimerOutput>           external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
flag_reinit_preconditioner(true),
flag_add_mass_and_stiffness_matrices(true),
flag_ignore_advection(false)
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  if (velocity.get() != nullptr)
    this->velocity        = velocity;
  else
  {
    this->velocity        = nullptr;
    flag_ignore_advection = true;
  }

  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(std::cout,
                                       Utilities::MPI::this_mpi_process(
                                        mpi_communicator) == 0));

  if (external_timer.get() != nullptr)
      computing_timer  = external_timer;
  else
      computing_timer.reset(new TimerOutput(*pcout,
                                            TimerOutput::summary,
                                            TimerOutput::wall_times));
  
  source_term_ptr       = nullptr;
  velocity_function_ptr = nullptr;
}

template <int dim>
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::ParameterSet        &parameters,
 TimeDiscretization::VSIMEXMethod             &time_stepping,
 std::shared_ptr<Entities::ScalarEntity<dim>> &temperature,
 std::shared_ptr<TensorFunction<1, dim>>      velocity,
 const std::shared_ptr<Mapping<dim>>          external_mapping,
 const std::shared_ptr<ConditionalOStream>    external_pcout,
 const std::shared_ptr<TimerOutput>           external_timer)
:
parameters(parameters),
mpi_communicator(temperature->mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
flag_reinit_preconditioner(true),
flag_add_mass_and_stiffness_matrices(true),
flag_ignore_advection(false)
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  if (velocity.get() != nullptr)
    velocity_function_ptr = velocity;
  else
  {
    velocity_function_ptr = nullptr;
    flag_ignore_advection = true;
  }

  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(std::cout,
                                       Utilities::MPI::this_mpi_process(
                                        mpi_communicator) == 0));

  if (external_timer.get() != nullptr)
      computing_timer  = external_timer;
  else
      computing_timer.reset(new TimerOutput(*pcout,
                                            TimerOutput::summary,
                                            TimerOutput::wall_times));
  
  source_term_ptr = nullptr;
  this->velocity  = nullptr;
}

}  // namespace RMHD

// explicit instantiations

template class RMHD::HeatEquation<2>;
template class RMHD::HeatEquation<3>;