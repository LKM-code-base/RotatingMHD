#include <rotatingMHD/heat_equation.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>
#include <cmath>

namespace RMHD
{

template <int dim>
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::ParameterSet   &parameters,
 TimeDiscretization::VSIMEXMethod        &time_stepping,
 Entities::ScalarEntity<dim>             &temperature,
 Entities::VectorEntity<dim>             &velocity,
 const std::shared_ptr<Mapping<dim>>      external_mapping,
 const std::shared_ptr<ConditionalOStream>external_pcout,
 const std::shared_ptr<TimerOutput>       external_timer)
:
parameters(parameters),
mpi_communicator(temperature.mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
velocity(&velocity),
flag_zero_mean_value(false),
flag_setup_solver(true),
flag_reinit_preconditioner(true),
flag_assemble_mass_plus_stiffness_matrix(true),
flag_ignore_advection(false)
{
  if (external_mapping.get() != 0)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  if (external_pcout.get() != 0)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(std::cout,
                                       Utilities::MPI::this_mpi_process(
                                        mpi_communicator) == 0));

  if (external_timer.get() != 0)
      computing_timer  = external_timer;
  else
      computing_timer.reset(new TimerOutput(*pcout,
                                            TimerOutput::summary,
                                            TimerOutput::wall_times));
  supply_term_ptr       = nullptr;
}

template <int dim>
HeatEquation<dim>::HeatEquation
(const RunTimeParameters::ParameterSet        &parameters,
 TimeDiscretization::VSIMEXMethod             &time_stepping,
 Entities::ScalarEntity<dim>                  &temperature,
 const std::shared_ptr<TensorFunction<1,dim>> velocity_function,
 const std::shared_ptr<Mapping<dim>>          external_mapping,
 const std::shared_ptr<ConditionalOStream>    external_pcout,
 const std::shared_ptr<TimerOutput>           external_timer)
:
parameters(parameters),
mpi_communicator(temperature.mpi_communicator),
time_stepping(time_stepping),
temperature(temperature),
flag_zero_mean_value(false),
flag_setup_solver(true),
flag_reinit_preconditioner(true),
flag_assemble_mass_plus_stiffness_matrix(true),
flag_ignore_advection(false)
{
  if (velocity_function.get() != 0)
    velocity_function_ptr = velocity_function;
  else
  {
    velocity_function_ptr.reset(new ZeroTensorFunction<1, dim>());
    flag_ignore_advection = true;
  }

  if (external_mapping.get() != 0)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  if (external_pcout.get() != 0)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(std::cout,
                                       Utilities::MPI::this_mpi_process(
                                        mpi_communicator) == 0));

  if (external_timer.get() != 0)
      computing_timer  = external_timer;
  else
      computing_timer.reset(new TimerOutput(*pcout,
                                            TimerOutput::summary,
                                            TimerOutput::wall_times));
  supply_term_ptr = nullptr;
  velocity        = nullptr;
}

}  // namespace RMHD

// explicit instantiations

template class RMHD::HeatEquation<2>;
template class RMHD::HeatEquation<3>;