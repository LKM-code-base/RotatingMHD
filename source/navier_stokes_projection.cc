#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>

namespace RMHD
{

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::NavierStokesParameters  &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::VectorEntity<dim>>     &velocity,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &pressure,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
phi(std::make_shared<Entities::ScalarEntity<dim>>(*pressure)),
parameters(parameters),
mpi_communicator(velocity->mpi_communicator),
velocity(velocity),
pressure(pressure),
time_stepping(time_stepping),
flag_normalize_pressure(false),
flag_setup_phi(true),
flag_add_mass_and_stiffness_matrices(true),
flag_ignore_bouyancy_term(true)
{
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != nullptr,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));

  Assert(parameters.C2 > 0.0,
         ExcLowerRangeType<double>(parameters.C2, 0.0));
  AssertIsFinite(parameters.C2);

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

  // Explicitly set the body forces and the temperature pointer to null
  body_force_ptr              = nullptr;
  gravity_vector_ptr          = nullptr;
  angular_velocity_vector_ptr = nullptr;
  temperature                 = nullptr;
}

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::NavierStokesParameters  &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::VectorEntity<dim>>     &velocity,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &pressure,
 std::shared_ptr<Entities::ScalarEntity<dim>>     &temperature,
 const std::shared_ptr<Mapping<dim>>              external_mapping,
 const std::shared_ptr<ConditionalOStream>        external_pcout,
 const std::shared_ptr<TimerOutput>               external_timer)
:
phi(std::make_shared<Entities::ScalarEntity<dim>>(*pressure)),
parameters(parameters),
mpi_communicator(velocity->mpi_communicator),
velocity(velocity),
pressure(pressure),
temperature(temperature),
time_stepping(time_stepping),
flag_normalize_pressure(false),
flag_setup_phi(true),
flag_add_mass_and_stiffness_matrices(true),
flag_ignore_bouyancy_term(false)
{
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != nullptr,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  Assert(parameters.C2 > 0.0,
         ExcLowerRangeType<double>(parameters.C2, 0.0));
  AssertIsFinite(parameters.C2);

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
  if (external_timer.get() != nullptr)gravity_unit_vector_ptr
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));

  // Explicitly set the body forces pointer to null
  body_force_ptr              = nullptr;
  gravity_vector_ptr          = nullptr;
  angular_velocity_vector_ptr = nullptr;
}

}  // namespace RMHD

// explicit instantiations
template class RMHD::NavierStokesProjection<2>;
template class RMHD::NavierStokesProjection<3>;

