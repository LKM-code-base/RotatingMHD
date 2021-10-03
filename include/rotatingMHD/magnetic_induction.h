#ifndef INCLUDE_ROTATINGMHD_MAGNETIC_INDUCTIONN_H_
#define INCLUDE_ROTATINGMHD_MAGNETIC_INDUCTIONN_H_

#include <rotatingMHD/solver_class.h>

namespace RMHD
{



namespace Solvers
{



using namespace dealii;



template <int dim>
class MagneticInduction : public ProjectionSolverBase<dim>
{

public:

/*!
 * @brief Construct a new MagneticInduction instance
 * @details Sets up the solver for the special case of an static fluid,
 * i.e., advection-free.
 *
 * @param parameters The parameter struct pertinent to the magnetic
 * induction problem.
 * @param time_stepping See the documentation for @ref SolverBase
 * @param magnetic_field The shared pointer to the @ref FE_VectorField
 * instance, which represents the magnetic field
 * @param external_mapping See the documentation for @ref SolverBase
 * @param external_pcout See the documentation for @ref SolverBase
 * @param external_timer See the documentation for @ref SolverBase
 */
MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  const std::shared_ptr<Mapping<dim>>             external_mapping =
    std::shared_ptr<Mapping<dim>>(),
  const std::shared_ptr<ConditionalOStream>       external_pcout =
    std::shared_ptr<ConditionalOStream>(),
  const std::shared_ptr<TimerOutput>              external_timer =
    std::shared_ptr<TimerOutput>());

/*!
 * @brief Construct a new MagneticInduction instance
 * @details Sets up the solver for the general case, where the velocity
 * field is an unkown field.
 *
 * @param parameters The parameter struct pertinent to the magnetic
 * induction problem.
 * @param time_stepping See the documentation for @ref SolverBase
 * @param magnetic_field The shared pointer to the @ref FE_VectorField
 * instance, which represents the magnetic field
 * @param velocity The shared pointer to the @ref FE_VectorField
 * instance, which represents the velocity field
 * @param external_mapping See the documentation for @ref SolverBase
 * @param external_pcout See the documentation for @ref SolverBase
 * @param external_timer See the documentation for @ref SolverBase
 */
MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &velocity,
  const std::shared_ptr<Mapping<dim>>             external_mapping =
    std::shared_ptr<Mapping<dim>>(),
  const std::shared_ptr<ConditionalOStream>       external_pcout =
    std::shared_ptr<ConditionalOStream>(),
  const std::shared_ptr<TimerOutput>              external_timer =
    std::shared_ptr<TimerOutput>());

/*!
 * @brief Construct a new MagneticInduction instance
 * @details Sets up the solver for the general case, where the velocity
 * field is given by a known function.
 *
 * @param parameters The parameter struct pertinent to the magnetic
 * induction problem.
 * @param time_stepping See the documentation for @ref SolverBase
 * @param magnetic_field The shared pointer to the @ref FE_VectorField
 * instance, which represents the magnetic field
 * @param velocity The shared pointer to the @ref TensorFunction
 * instance, which represents the velocity field
 * @param external_mapping See the documentation for @ref SolverBase
 * @param external_pcout See the documentation for @ref SolverBase
 * @param external_timer See the documentation for @ref SolverBase
 */
MagneticInduction(
  const RunTimeParameters::MagneticInduction      &parameters,
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  TensorFunction<1, dim>                          &velocity_function,
  const std::shared_ptr<Mapping<dim>>             external_mapping =
    std::shared_ptr<Mapping<dim>>(),
  const std::shared_ptr<ConditionalOStream>       external_pcout =
    std::shared_ptr<ConditionalOStream>(),
  const std::shared_ptr<TimerOutput>              external_timer =
    std::shared_ptr<TimerOutput>());

/*!
 * @brief A shared pointer to the @ref FE_ScalarField, representing
 * the pseudo-pressure, i.e., the Lagrange multiplier, which enforces
 * the divergence-free condition on the magnetic field.
 *
 */
std::shared_ptr<Entities::FE_ScalarField<dim>>  pseudo_pressure;

/*!
 * @brief
 *
 */
void clear() override;

private:

/*!
 * @brief
 *
 */
const RunTimeParameters::MagneticInduction            &parameters;

/*!
 * @brief A shared pointer to the @ref FE_VectorField, representing
 * the magnetic field.
 *
 */
std::shared_ptr<Entities::FE_ScalarField<dim>>        magnetic_field;

/*!
 * @brief A shared pointer to the @ref FE_VectorField, representing
 * the velocity field.
 *
 */
std::shared_ptr<const Entities::FE_VectorField<dim>>  velocity;

/*!
 * @brief A shared pointer to the @ref TensorFunction, representing
 * the velocity field.
 *
 */
TensorFunction<1,dim>                                 *ptr_velocity_function;

/*!
 * @brief
 *
 */
void setup() override;

/*!
 * @brief
 *
 */
void setup_pseudo_pressure();

/*!
 * @brief Set the up auxiliary scalar object
 *
 */
void setup_auxiliary_scalar() override;

/*!
 * @brief A virtual method, which sets up the linear algebra of the
 * solver
 *
 */
void setup_matrices() override;

/*!
 * @brief A virtual method, which sets up the linear algebra of the
 * solver
 *
 */
void setup_vectors() override;

/*!
 * @brief A method that computes admissable initial
 * conditions for the pseudo-pressure by considering and solving
 * the system of equations in its steady-state at \f$ t = 0 \f$.
 *
 */
void zeroth_step() override;

/*!
 * @brief A method performing the diffusion step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 */
void diffusion_step(const bool reinit_preconditioner) override;

/*!
 * @brief A method performing the projection step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 */
void projection_step(const bool reinit_preconditioner) override;

/*!
 * @brief A method performing the correction step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 */
void correction_step(const bool reinit_preconditioner) override;

/*!
 * @brief
 *
 */
void assemble_diffusion_step() override;

/*!
 * @brief
 *
 */
void assemble_diffusion_step_rhs() override;

/*!
 * @brief This method assembles the local right-hand side of the
 * diffusion step on a single cell.
 */
void assemble_local_diffusion_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator              &cell,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Copy          &data);

/*!
 * @brief This method copies the local right-hand side of the diffusion
 * step into the global vector.
 */
void copy_local_to_global_diffusion_step_rhs(
  const AssemblyData::MagneticInduction::DiffusionStepRHS::Copy  &data);

/*!
 * @brief
 *
 */
void assemble_projection_step_rhs() override;

/*!
 * @brief This method assembles the local right-hand side of the
 * projection step on a single cell.
 */
void assemble_local_projection_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator              &cell,
  AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::ProjectionStepRHS::Copy          &data);

/*!
 * @brief This method copies the local right-hand side of the projection
 * step into the global vector.
 */
void copy_local_to_global_projection_step_rhs(
  const AssemblyData::MagneticInduction::ProjectionStepRHS::Copy  &data);

/*!
 * @brief
 *
 */
void assemble_zeroth_step_rhs() override;

/*!
 * @brief This method assembles the local right-hand side of the zeroth
 * step on a single cell.
 */
void assemble_local_zeroth_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator          &cell,
  AssemblyData::MagneticInduction::ZerothStepRHS::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::ZerothStepRHS::Copy          &data);

/*!
 * @brief This method copies the local right-hand side of the zeroth
 * step into the global vector.
 */
void copy_local_to_global_zeroth_step_rhs(
  const AssemblyData::MagneticInduction::ZerothStepRHS::Copy  &data);

/*!
 * @brief
 *
 */
void assemble_advection_matrix();

/*!
 * @brief This method assembles the local advection matrix on a
 * single cell.
 */
void assemble_local_advection_matrix(
  const typename DoFHandler<dim>::active_cell_iterator            &cell,
  AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::AdvectionMatrix::Copy          &data);

/*!
 * @brief This method copies the local advection matrix into the
 * global matrix.
 */
void copy_local_to_global_advection_matrix(
  const AssemblyData::MagneticInduction::AdvectionMatrix::Copy  &data);

};


} // namespace Solvers


} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_MAGNETIC_INDUCTIONN_H_ */
