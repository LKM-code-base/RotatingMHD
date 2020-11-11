#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::ParameterSet        &parameters,
 std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
 std::shared_ptr<Entities::ScalarEntity<dim>> &pressure,
 TimeDiscretization::VSIMEXMethod             &time_stepping,
 const std::shared_ptr<ConditionalOStream>    external_pcout,
 const std::shared_ptr<TimerOutput>           external_timer)
:
phi(std::make_shared<Entities::ScalarEntity<dim>>(*pressure)),
parameters(parameters),
mpi_communicator(velocity->mpi_communicator),
velocity(velocity),
pressure(pressure),
time_stepping(time_stepping),
flag_initializing(false),
flag_normalize_pressure(false),
flag_setup_phi(true),
flag_setup_solver(true),
flag_add_mass_and_stiffness_matrices(true)
{
  Assert(velocity.get() != 0,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != 0,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));

  // Initiating the internal ConditionalOStream and TimerOutput instances.
  if (external_pcout.get() != 0)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  if (external_timer.get() != 0)
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

