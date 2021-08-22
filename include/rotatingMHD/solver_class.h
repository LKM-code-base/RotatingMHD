#ifndef INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_
#define INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_

#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/mapping_q.h>

#include <memory>

namespace RMHD
{



using namespace dealii;




/**
 * @brief A class serving as a base for all the solvers in the library.
 *
 * @tparam dim An integer indicating the spatial dimension of the
 * problem.
 */
template<int dim>
class SolverBase
{

public:

/**
 * @brief Construct a new SolverBase instance
 *
 * @param time_stepping An instance of the
 *    @ref TimeDiscretization::VSIMEXMethod class. A reference to the
 *    instance is a member of the @ref SolverBase class
 * @param external_mapping A shared pointer to an instance of the
 *    Mapping class, which is copied in an internal member.
 *    If no pointer is passed on to the constructor, a
 *    first order instance of the MappingQ class is created.
 * @param external_pcout A shared pointer to an instance of the
 *    ConditionalOStream class, which is copied in an internal member.
 *    If no pointer is passed on to the constructor, an instance is
 *    created, which outputs its calls to the zeroth process.
 * @param external_timer A shared pointer to an instance of the
 *    TimerOutput class, which is copied in an internal member.
 *    If no pointer is passed on to the constructor, an instance
 *    is created, which prints a summary of the wall times upon the
 *    destruction of the instance.
 */
SolverBase(
  TimeDiscretization::VSIMEXMethod            &time_stepping,
  const std::shared_ptr<Mapping<dim>>         external_mapping =
    std::shared_ptr<Mapping<dim>>(),
  const std::shared_ptr<ConditionalOStream>   external_pcout =
    std::shared_ptr<ConditionalOStream>(),
  const std::shared_ptr<TimerOutput>          external_timer =
    std::shared_ptr<TimerOutput>());

/**
 * @brief A pure virtual method for the set-up of the solver's linear
 * algebra.
 *
 */
virtual void setup() = 0;

/**
 * @brief A pure virtual method, which performs the solve operation.
 *
 */
virtual void solve() = 0;

/**
 * @brief A pure virtual method, which clears the solver's linear
 * algebra and resets all internal booleans.
 *
 */
virtual void clear() = 0;


protected:

/**
 * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
 */
const MPI_Comm                          mpi_communicator;

/**
 * @brief A shared pointer to a ConditionalOStream instance.
 */
std::shared_ptr<ConditionalOStream>     pcout;

/**
 * @brief A shared pointer to a TimerOutput instance.
 */
std::shared_ptr<TimerOutput>            computing_timer;

/**
 * @brief A shared pointer to a Mapping instance.
 */
std::shared_ptr<Mapping<dim>>           mapping;

/**
 * @brief A shared pointer to a @ref TimeDiscretization::VSIMEXMethod
 * instance.
 */
const TimeDiscretization::VSIMEXMethod  &time_stepping;

/**
  * @brief A flag indicating if the internal matrices were updated.
  */
bool                                    flag_matrices_were_updated;

/**
 * @brief A pure virtual method to set-up the matrices of the solver
 *
 */
virtual void setup_matrices() = 0;

/**
 * @brief A pure virtual method to set-up the vectors of the solver
 *
 */
virtual void setup_vectors() = 0;
};



/**
 * @brief A class serving as a base for all the solvers in the library,
 * which are based on a projection scheme for solenoidal vector fields.
 *
 * @details The projection scheme decouples the set of equations
 * \f[
 *    \nabla \cdot \bs{a}_\textrm{s},
 *    \quad \mathcal{L}(\bs{a}_\textrm{s}) = \bs{f}, \qquad
 *    \forall (\bs{x}, t) \in \Omega \times \mathbb{R}_0
 * \f]
 * where \f$ \bs{a}_\textrm{s} \f$ is a vector, on which a divergence-free
 * constraint is enforced, i.e. a solenoidal vector;
 * \f$ \mathcal{L} \f$ is a differential operator;
 * \f$ \bs{x} \f$ is the position vector;
 * \f$ t \f$ is the time;
 * \f$ \Omega \f$ is the spatial domain;
 * \f$ \mathbb{R}_0 \f$ is the temporal domain, all real numbers
 * including zero. Based on a temporal decoupling of variables and
 * the \textsc{Helmholtz} decomposition of an arbitrary vector into
 * the sum of a solenoidal and a irrotational vector
 * \f[
 *    \bs{a} = \bs{a}_\textrm{s} + \nabla \phi,
 * \f]
 * the projection scheme decouples \f$ \mathcal{L}(\bs{a}_\textrm{s})\f$
 * from the divergence-free condition by solving a diffusion step
 * \f[
 *    \mathcal{L}(\bs{a}) = \bs{f};
 * \f]
 * a projection step, in which \f$ \bs{a} \f$ is projected
 * into the space of solenoidal vectors, i.e., \f$\bs{a}_\textrm{s}\f$
 * is computed; and a correction step, which updates \f$ \phi \f$ to
 * the current time step. A more detailed explanation, can be found in
 * the child classes of @ref ProjectionSolverBase.
 * @tparam dim An integer indicating the spatial dimension of the
 * problem.
 * @attention The solenoidal vector field is eliminated in the internal
 * algorithm. Therefore, the computed vector field is not solenoidal but
 * it fulfills the boundary conditions.
 */
template<int dim>
class ProjectionSolverBase : SolverBase<dim>
{
public:

/**
 * @brief Construct a new ProjectionSolverBase instance
 *
 * @param time_stepping See the documentation for @ref SolverBase
 * @param vector The vector field on which the divergence-free condition
 * is imposed
 * @param lagrange_multiplier The lagrange multiplier imposing the
 * divergence-free condition
 * @param external_mapping See the documentation for @ref SolverBase
 * @param external_pcout See the documentation for @ref SolverBase
 * @param external_timer See the documentation for @ref SolverBase
 */
ProjectionSolverBase(
  TimeDiscretization::VSIMEXMethod                &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>  &vector,
  std::shared_ptr<Entities::FE_ScalarField<dim>>  &lagrange_multiplier,
  const std::shared_ptr<Mapping<dim>>             external_mapping =
    std::shared_ptr<Mapping<dim>>(),
  const std::shared_ptr<ConditionalOStream>       external_pcout =
    std::shared_ptr<ConditionalOStream>(),
  const std::shared_ptr<TimerOutput>              external_timer =
    std::shared_ptr<TimerOutput>());


/**
 * @brief The auxiliary scalar field introduced to simplify the
 * diffusion and projection step.
 *
 */
std::shared_ptr<Entities::FE_ScalarField<dim>>  auxiliary_scalar;

/**
 * @brief Sets the internal supply term shared pointer equal to
 * @ref supply_term.
 *
 * @param supply_term The external supply term pointer.
 */
void set_supply_term(
  std::shared_ptr<TensorFunction<1, dim>> supply_term);

/**
 * @brief Returns the norm of the right hand side of the diffusion step.
 *
 * @return double \f$ L_2 \f$-Norm of the right-hand side of the
 * diffusion step.
 */
double get_diffusion_step_rhs_norm() const;

/**
 * @brief Returns the norm of the right hand side of the projection step.
 *
 * @return double \f$ L_2 \f$-Norm of the right-hand side of the
 * projection step.
 */
double get_projection_step_rhs_norm() const;

protected:

/**
 * @brief A shared pointer to the vector field on which the
 * divergence-free condition is imposed.
 *
 */
std::shared_ptr<Entities::FE_VectorField<dim>>  vector;

/**
 * @brief A shared pointer to the lagrange multiplier which imposes
 * the divergence-free condition on the @ref vector.
 *
 */
std::shared_ptr<Entities::FE_ScalarField<dim>>  lagrange_multiplier;

/**
 * @brief A shared pointer to the supply term function.
 *
 */
std::shared_ptr<TensorFunction<1, dim>>       supply_term;

/**
 * @brief The system matrix of the diffusion step.
 *
 * @details If the advection term is treated implicitly, the entries
 * of the system matrix change in every time step.
 *
 */
LinearAlgebra::MPI::SparseMatrix  diffusion_step_system_matrix;

/**
 * @brief The mass matrix of the diffusion step.
 *
 * @details The entries of the mass matrix only change if the spatial
 * discretization changes. It is stored in memory because otherwise an
 * assembly would be required if the timestep changes.
 *
 */
LinearAlgebra::MPI::SparseMatrix  diffusion_step_mass_matrix;

/**
 * @brief The stiffness matrix of the diffusion step.
 *
 * @details The entries of the stiffness matrix only change if the
 * spatial discretization changes. It is stored in memory because otherwise an
 * assembly would be required if the timestep changes.
 *
 */
LinearAlgebra::MPI::SparseMatrix  diffusion_step_stiffness_matrix;

/**
 * @brief The sum of the mass and stiffness matrix of the diffusion step.
 *
 * @details The entries of the stiffness matrix only change if the
 * spatial discretization changes. It is stored in memory because otherwise an
 * assembly would be required if the timestep changes.
 *
 */
LinearAlgebra::MPI::SparseMatrix  diffusion_step_mass_plus_stiffness_matrix;


/**
 * @brief The advection matrix of the diffusion step.
 *
 * @details The entries of the advection matrix change in every
 * time-step. Therefore, it has to be assembled in each time-step.
 *
 */
LinearAlgebra::MPI::SparseMatrix  diffusion_step_advection_matrix;


/**
 * @brief Vector representing the right-hand side of the linear system
 * of the diffusion step.
 *
 */
LinearAlgebra::MPI::Vector        diffusion_step_rhs;

/**
 * @brief The norm of the right hand side of the diffusion step.
 * @details Its value is that of the last computed pressure-correction
 * scheme step.
 *
 */
double                            norm_diffusion_rhs;

/**
 * @brief The preconditioner of the diffusion step.
 *
 */
std::shared_ptr<LinearAlgebra::PreconditionBase> diffusion_step_preconditioner;

/**
 * @brief The system matrix of the projection step.
 *
 * @details The system matrix corresponds to the stiffness matrix of the
 * auxiliary scalar field. Its entries only change if the
 * spatial discretization changes. It is stored in memory because
 * otherwise an assembly would be required if the timestep changes.
 *
 */
LinearAlgebra::MPI::SparseMatrix  projection_step_system_matrix;


/**
 * @brief Vector representing the right-hand side of the linear system
 * of the projection step.
 *
 */
LinearAlgebra::MPI::Vector        projection_step_rhs;

/**
 * @brief The norm of the right hand side of the projection step.
 * @details Its value is that of the last computed pressure-correction
 * scheme step.
 *
 */
double                            norm_projection_rhs;

/**
 * @brief The preconditioner of the projection step
 *
 */
std::shared_ptr<LinearAlgebra::PreconditionBase> projection_step_preconditioner;

/**
  * @brief Assemble the matrices which change only if the triangulation is
  * refined or coarsened.
  */
void assemble_constant_matrices();

};



} // namespace RMHD



#endif /*INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_*/
