#include <rotatingMHD/magnetic_induction.h>

namespace RMHD
{



namespace Solvers
{



using namespace dealii;



template<int dim>
MagneticInduction<dim>::MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  const std::shared_ptr<Mapping<dim>>             external_mapping,
  const std::shared_ptr<ConditionalOStream>       external_pcout,
  const std::shared_ptr<TimerOutput>              external_timer)
:
ProjectionSolverBase<dim>(
  parameters,
  time_stepping,
  magnetic_field,
  external_mapping,
  external_pcout,
  external_timer),
parameters(parameters),
magnetic_field(magnetic_field)
{
  Assert(magnetic_field.get() != nullptr,
        ExcMessage("The magnetic field's shared pointer has not been"
                    " initialized."));

  Assert(parameters.C8 > 0.0,
        ExcLowerRangeType<double>(parameters.C8, 0.0));
  AssertIsFinite(parameters.C8);

  velocity              = nullptr;
  ptr_velocity_function = nullptr;
}



template<int dim>
MagneticInduction<dim>::MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &velocity,
  const std::shared_ptr<Mapping<dim>>             external_mapping,
  const std::shared_ptr<ConditionalOStream>       external_pcout,
  const std::shared_ptr<TimerOutput>              external_timer)
:
ProjectionSolverBase<dim>(
  parameters,
  time_stepping,
  magnetic_field,
  external_mapping,
  external_pcout,
  external_timer),
parameters(parameters),
magnetic_field(magnetic_field),
velocity(velocity)
{
  Assert(magnetic_field.get() != nullptr,
        ExcMessage("The magnetic field's shared pointer has not been"
                    " initialized."));

  Assert(velocity.get() != nullptr,
        ExcMessage("The magnetic field's shared pointer has not been"
                    " initialized."));
  Assert(parameters.C8 > 0.0,
        ExcLowerRangeType<double>(parameters.C8, 0.0));
  AssertIsFinite(parameters.C8);

  ptr_velocity_function = nullptr;
}



template<int dim>
MagneticInduction<dim>::MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  TensorFunction<1, dim>                          &velocity_function,
  const std::shared_ptr<Mapping<dim>>             external_mapping,
  const std::shared_ptr<ConditionalOStream>       external_pcout,
  const std::shared_ptr<TimerOutput>              external_timer)
:
ProjectionSolverBase<dim>(
  parameters,
  time_stepping,
  magnetic_field,
  external_mapping,
  external_pcout,
  external_timer),
parameters(parameters),
magnetic_field(magnetic_field),
ptr_velocity_function(*velocity_function)
{
  Assert(magnetic_field.get() != nullptr,
        ExcMessage("The magnetic field's shared pointer has not been"
                    " initialized."));

  Assert(parameters.C8 > 0.0,
        ExcLowerRangeType<double>(parameters.C8, 0.0));
  AssertIsFinite(parameters.C8);

  velocity = nullptr;
}



} // namespace Solvers

} // namespace RMHD