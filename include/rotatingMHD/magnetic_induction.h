#ifndef INCLUDE_ROTATINGMHD_MAGNETIC_INDUCTIONN_H_
#define INCLUDE_ROTATINGMHD_MAGNETIC_INDUCTIONN_H_

#include <rotatingMHD/solver_class.h>

namespace RMHD
{



namespace Solvers
{



using namespace dealii;


/*!
 * @brief A class, which solves the hydrodynamic approximation of the
 * Maxwell equations using a projection scheme for the decoupling of
 * the numerically introduced pseudo-pressure.
 *
 * @details The projection scheme decouples the set of equations
 * \f[
 * \nabla \cdot \bs{B} = 0, \quad
 * \pd{\bs{B}}{t} = \nabla \times (\bs{v} \times \bs{B}) -
 * \nabla \times (\eta\nabla \times \bs{B}),
 * \qquad \forall (\bs{x},t) \in \Omega \times \mathbb{R}_0,
 * \f]
 * where \f$ \bs{B} \f$ is the magnetic field, on which a
 * divergence-free constraint is enforced, i.e. a solenoidal vector field;
 * \f$ \bs{x} \f$ is the position vector;
 * \f$ t \f$ is the time;
 * \f$ \Omega \f$ is the spatial domain;
 * \f$ \mathbb{R}_0 \f$ is the temporal domain, all real numbers
 * including zero. The operator separation is based on the
 * Helmholtz decomposition of an arbitrary vector field into
 * the sum of a solenoidal and a irrotational vector. To this end, the
 * non-solenoidal magnetic field \f$  \boldsymbol{\mathfrak{B}} \f$ and the
 * auxiliary scalar field \f$ \phi \f$ are introduced.
 * The scheme is divided into four steps:
 * - An initialization step, which computes an admissible initial
 * condition for the pseudo-pressure by solving
 * \f[
 *    -\nabla^2 \sigma^0 = \nabla \cdot \bs{f}^0 ,
 *    \qquad \forall\bs{x} \in \Omega^0.
 * \f]
 * - A diffusion step, which computes the  the
 * non-solenoidal magnetic field \f$  \boldsymbol{\mathfrak{B}} \f$ by
 * solving the advection-diffusion equation
 * \f{split}{
 *    \frac{\alpha_0^k}{\Delta t^k} \boldsymbol{\mathfrak{B}}^k +
 *    \sum_{j=1}^{s} \frac{\alpha_j^k}{\Delta t^k}
 *    \boldsymbol{\mathfrak{B}}^{k-j} = \sum_{j=1}^{s} \beta^k_j \nabla
 *    \times (\bs{v}^{k-j} \times \boldsymbol{\mathfrak{B}}^{k-j}) -
 *    \sum_{j=0}^{s}\frac{\gamma^k_j}{\magPrandtl} \nabla
 *    \times(\nabla \times \boldsymbol{\mathfrak{B}}^{k-j})+ \nabla
 *    \sigma^{\sharp,k} + \bs{f}^k, \qquad \forall \bs{x} \in \Omega^k,
 * \f}
 * where
 * \f[
 *    \sigma^{\sharp, k} = \sigma^{\star, k} - \sum_{j=1}^s
 *    \frac{\alpha_j^k}{\alpha_0^{k-j}} \frac{\Delta t^{k-j}}{\Delta t^k}
 *    \phi^{k-j}
 * \f]
 * The time stepping scheme is the family of \f$ s \f$-th order VSIMEX
 * methods, specifically the second order family (\f$ s = 2 \f$) is
 * implemented in the code.
 * - A projection step, which projects the
 * non-solenoidal magnetic field \f$  \boldsymbol{\mathfrak{B}} \f$
 * into its solenoidal space, i.e., \f$ \boldsymbol{B} \f$, by solving the
 * Poisson equation that follows from
 * \f[
 *    \nabla \cdot \bs{B} = 0, \quad \frac{\alpha^k_0}{\Delta t^k}
 *    \left( \bs{B}^k - \boldsymbol{\mathfrak{B}}^k \right) = \nabla \phi^k,
 *    \qquad \forall\bs{x} \in \Omega^k
 * \f]
 * - A correction step, which corrects the current value of the
 * pseudo-pressure through as
 * \f[
 *    \sigma^k = \sigma^{\star,k} + \phi^k,
 *    \qquad \forall \bs{x}\in \Omega^k.
 * \f]
 *
 * @tparam dim An integer indicating the spatial dimension of the
 * problem.
 * @attention The solenoidal vector field is eliminated in the internal
 * algorithm. Therefore, the computed vector field is not solenoidal but
 * it fulfills the boundary conditions.
 */
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
 * @brief A method which clear all the internal entities of the solver.
 *
 */
void clear() override;

private:

/*!
 * @brief A constant reference to the
 * @ref RunTimeParameters::MagneticInduction instance contained all of
 * the solver's parameters.
 *
 */
const RunTimeParameters::MagneticInduction            &parameters;

/*!
 * @brief A shared pointer to the @ref FE_VectorField, representing
 * the magnetic field.
 *
 */
std::shared_ptr<Entities::FE_VectorField<dim>>        magnetic_field;

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
 * @brief A method for the set-up of the solver's linear algebra.
 *
 */
void setup() override;

/*!
 * @brief A method for the set-up of the pseudo-pressure field.
 *
 */
void setup_pseudo_pressure();

/*!
 * @brief A method for the set-up of auxiliary scalar field.
 *
 */
void setup_auxiliary_scalar() override;

/*!
 * @brief A method, which sets up the matrices of the solver
 *
 */
void setup_matrices() override;

/*!
 * @brief A virtual method, which sets up the vectors of the solver
 *
 */
void setup_vectors() override;

/*!
 * @brief A method that computes admissable initial
 * conditions for the pseudo-pressure by considering and solving
 * the system of equations in its steady-state at \f$ t = 0 \f$.
 *
 * @details The initialization step consist of the numerical solution
 * of the continuous variational problem: Find \f$\sigma^0 \in V_\sigma \f$
 * such that
 * \f[
 * \hat{a}_\textrm{init}(q, \sigma^k) = \hat{L}_\textrm{init}(q),
 * \qquad \forall q \in V_{q},
 * \f]
 * with
 * \f[
 * \hat{a}_\textrm{init}(q, \sigma^0)
 * =
 * \int_\Omega \nabla q \cdot \nabla \sigma^0 \dint{v}
 * \quad \textrm{and}\quad
 * \hat{L}_\textrm{init}(q) =- \int_{\partial \Omega}
 * \frac{1}{\magPrandtl} q \nabla^2 \bs{B}^0 \cdot \bs{n }\dint{a} -
 * \int_\Omega \nabla q \cdot \bs{f}^0\dint{v};
 * \f]
 *
 */
void initialization_step() override;

/*!
 * @brief A method performing the diffusion step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 *
 * @details The diffusion step consist of the numerical solution of the
 * continuous variational problem: Find \f$\boldsymbol{\mathfrak{B}}^k
 * \in V_{\boldsymbol{\mathfrak{B}}}\f$ such that
 * \f[
 * \hat{a}_\textrm{diff}(\bs{q}, \boldsymbol{\mathfrak{B}}^k) =
 * \hat{L}_\textrm{diff}(\bs{q}), \qquad \forall \bs{q} \in V_{\bs{q}},
 * \f]
 * with
 * \f[
 * \hat{a}_\textrm{diff}(\bs{q}, \boldsymbol{\mathfrak{B}}^k)
    =
    \int_\Omega \left[ \frac{\alpha_0^k}{\Delta t^k} \bs{q} \cdot
    \boldsymbol{\mathfrak{B}}^k + \frac{\gamma_0^k}{\magPrandtl}
    (\nabla \times \bs{q}) \cdot (\nabla \times \boldsymbol{\mathfrak{B}}^k  )
    \right] \dint{v}
 * \f]
 * and
 * \f{align}{
 * \hat{L}_\textrm{diff}(\bs{q}) =& \int_\Omega \bs{q}
 * \cdot \nabla \sigma^{\sharp, k}\dint{v}\\& -
 * \int_\Omega \sum_{j=1}^s \frac{\alpha_j^{k}}{\Delta t^k} \bs{q}
 * \cdot \boldsymbol{\mathfrak{B}}^{k-j}  \dint{v} \\
 *   & + \int_\Omega \sum_{j=1}^s \beta_j^k \bs{q} \cdot [ (\nabla
 *    \cdot \boldsymbol{\mathfrak{B}}^{k-j}) \bs{v}^{k-j} +
 *    (\nabla \cdot \bs{v}^{k-j}) \boldsymbol{\mathfrak{B}}^{k-j}
 *    \\ & \qquad \qquad \qquad \quad + \boldsymbol{\mathfrak{B}}^{k-j}
 *    \cdot (\nabla \otimes \bs{v}^{k-j}) -
 *    \bs{v}^{k-j} \cdot (\nabla \otimes \boldsymbol{\mathfrak{B}}^{k-j})]
 *    \dint{v} \\
 *   & - \int_\Omega \sum_{j=1} \frac{\gamma_j^k}{\magPrandtl}
 *    (\nabla \times \bs{q}) \cdot (\nabla \times
 *    \boldsymbol{\mathfrak{B}}^{k-j})\dint{v} \\
 *   & + \int_\Omega \bs{q} \cdot \bs{f}^k \dint{v} \\
 *   & + \int_{\partial \Omega} \sum_{j=1} \frac{\gamma_j^k}{\magPrandtl}
 *    \bs{q} \cdot \left[ ( \nabla \times \boldsymbol{\mathfrak{B}}^{k-j} )
 *    \times \bs{n} \right]\dint{a};
 * \f}
 *
 */
void diffusion_step(const bool reinit_preconditioner) override;

/*!
 * @brief A method performing the projection step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 *
 * @details The projection step consist of the numerical solution of the
 * continuous variational problem:  Find \f$ \phi^k \in V_{\phi} \f$ such that
 * \f[
 *    \hat{a}_\textrm{proj}(q, \phi^k) = \hat{L}_\textrm{proj}(q),
 *    \qquad \forall q \in V_{q}
 * \f]
 * with
 * \f[
 *    \hat{a}_\textrm{proj}(q, \phi^k) =
 *    \int_\Omega \nabla q \cdot \nabla \phi^k \dint{v} \quad
 *    \textrm{and}\quad \hat{L}_\textrm{proj}(q) =
 *    \int_\Omega \frac{\alpha_0^k}{\Delta t^k} q
 *    (\nabla \cdot \mathfrak{B}^k)\dint{v};
 * \f]
 */
void projection_step(const bool reinit_preconditioner) override;

/*!
 * @brief A method performing the correction step.
 *
 * @param reinit_preconditioner A boolean indicating if the
 * preconditioner is to be re-built.
 *
 * @details The diffusion step consist of vector addition
 * \f[
 *    \Sigma^k = \Sigma^{k-1} + \Phi^k,
 * \f]
 * where \f$ \Sigma\f$ and \f$ \Phi \f$ contain the nodal values of
 * pseudo pressure field and the auxiliary scalar field, respectively.
 * The superindices indicate their respective time step.
 *
 */
void correction_step(const bool reinit_preconditioner) override;

/*!
 * @brief A method assembling the linear algebra of the diffusion step
 *
 */
void assemble_diffusion_step() override;

/*!
 * @brief A method assembling the right-hand side vector of the
 * diffusion step
 *
 */
void assemble_diffusion_step_rhs() override;

/*!
 * @brief This method assembles the local right-hand side of the
 * diffusion step on a single cell.
 */
void assemble_local_diffusion_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator             &cell,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::DiffusionStepRHS::Copy          &data);

/*!
 * @brief This method copies the local right-hand side of the diffusion
 * step into the global vector.
 */
void copy_local_to_global_diffusion_step_rhs(
  const AssemblyData::MagneticInduction::DiffusionStepRHS::Copy  &data);

/*!
 * @brief A method assembling the right-hand side vector of the
 * projection step
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
 * @brief A method assembling the right-hand side vector of the
 * initialization step
 *
 */
void assemble_initialization_step_rhs() override;

/*!
 * @brief This method assembles the local right-hand side of the
 * initialization step on a single cell.
 */
void assemble_local_initialization_step_rhs(
  const typename DoFHandler<dim>::active_cell_iterator                  &cell,
  AssemblyData::MagneticInduction::InitializationStepRHS::Scratch<dim>  &scratch,
  AssemblyData::MagneticInduction::InitializationStepRHS::Copy          &data);

/*!
 * @brief This method copies the local right-hand side of the
 * initialization step into the global vector.
 */
void copy_local_to_global_initialization_step_rhs(
  const AssemblyData::MagneticInduction::InitializationStepRHS::Copy  &data);

/*!
 * @brief A method assembling advection matrix of the diffusion step
 *
 * @details The advection matrix is given by
 * \f[
 * \mathcal{A}_{\boldsymbol{\mathfrak{B}}}
 * =
 * \int
 * \bs{q} \cdot [
 * (\nabla \cdot \boldsymbol{\mathfrak{B}}^{k}) \bs{v}^{k}
 * +
 * (\nabla \cdot \bs{v}^{k}) \boldsymbol{\mathfrak{B}}^{k}
 * +
 * \boldsymbol{\mathfrak{B}}^{k} \cdot (\nabla \otimes \bs{v}^{k})
 * -
 * \bs{v}^{k} \cdot (\nabla \otimes \boldsymbol{\mathfrak{B}}^{k})]
 * \dint{v},
 * \f]
 * for the case of a known velocity field. If the velocity field is a
 * variable all the of instances above are replaced by the second order
 * Taylor extrapolation of \f$ \bs{v}^k \f$.
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
