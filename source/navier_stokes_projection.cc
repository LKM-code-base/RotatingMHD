#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::ParameterSet        &parameters,
 TimeDiscretization::VSIMEXMethod             &time_stepping,
 std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
 std::shared_ptr<Entities::ScalarEntity<dim>> &pressure,
 std::shared_ptr<Entities::ScalarEntity<dim>> temperature,
 const std::shared_ptr<ConditionalOStream>    external_pcout,
 const std::shared_ptr<TimerOutput>           external_timer)
:
phi(std::make_shared<Entities::ScalarEntity<dim>>(*pressure)),
parameters(parameters),
mpi_communicator(velocity->mpi_communicator),
velocity(velocity),
pressure(pressure),
temperature(temperature),
time_stepping(time_stepping),
flag_initializing(false),
flag_normalize_pressure(false),
flag_setup_phi(true),
flag_add_mass_and_stiffness_matrices(true),
flag_ignore_temperature(false)
{
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != nullptr,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));
  if (temperature.get() == nullptr)
    flag_ignore_temperature = true;

  // Initiating the internal ConditionalOStream and TimerOutput instances.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));
  
  // Explicitly set the body force pointer to null
  body_force_ptr = nullptr;
}

}  // namespace RMHD

// explicit instantiations
template class RMHD::NavierStokesProjection<2>;
template class RMHD::NavierStokesProjection<3>;

