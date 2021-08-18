#ifndef INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_
#define INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_

#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <rotatingMHD/finite_element_field.h>

namespace RMHD
{

using namespace dealii;

/*!
 * @struct SolutionTransferContainer
 * @brief An struct that groups together all the entities that
 * are to be considered during a solution transfer operation.
 * @details In case of the case of a refining and coarsening operation,
 * the container also keeps track as to which entities are to be
 * considered by calculation of the total error estimate.
 */
template <int dim>
struct SolutionTransferContainer
{
  /*!
   * @brief A typedef for the std::pair composed of a pointer to a
   * @ref Entities::FE_FieldBase instance and a boolean.
   * @details The boolean indicates wheter the entity is to be
   * considered by the error estimation or not.
   */
  using FE_Field = std::pair<Entities::FE_FieldBase<dim> *, bool>;

  /*!
   * @brief A std::vector with all the entities to be considered in
   * a solution transfer
   */
  std::vector<FE_Field>  entities;

  /*!
   * @brief Default constructor.
   */
  SolutionTransferContainer();

  /*!
   * @details Release all memory and return all objects to a state just like
   * after having called the default constructor.
   */
  void clear();

  /*!
   * @brief Inline returning the number of entities to be considered
   * by the error estimation.
   */
  unsigned int get_error_vector_size() const;

  /*!
   * @brief Indicates whether @ref entities is empty or not.
   */
  bool empty() const;

  /*!
   * @brief Adds the passed on FE_FieldBase instance and flag to the
   * entities struct member.
   * @details If no boolean is passed, it is assumed that the entity
   * is to be considered by the error estimation.
   */
  void add_entity(std::shared_ptr<Entities::FE_FieldBase<dim>> entity, bool flag = true);

private:

  /*!
   * @brief The size of the std::vector instance containing all the
   * Vector instances to be considered by the error estimation.
   */
  unsigned int              error_vector_size;
};

template <int dim>
inline unsigned int SolutionTransferContainer<dim>::get_error_vector_size() const
{
  return error_vector_size;
}

template <int dim>
inline bool SolutionTransferContainer<dim>::empty() const
{
  return entities.empty();
}

/*!
 * @class Problem
 * @brief The class containts instances and methods that are common
 * and/or useful for most problems to be formulated.
 */
template <int dim>
class Problem
{
public:
  /*!
   * @brief Default constructor which initializes the member variables.
   */
  Problem(const RunTimeParameters::ProblemBaseParameters &prm);

protected:
  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm  mpi_communicator;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const RunTimeParameters::ProblemBaseParameters  &prm;

  /*!
   * @brief Triangulation object of the problem.
   */
  parallel::distributed::Triangulation<dim> triangulation;

  /*!
   * @brief The shared pointer to the class describing the mapping from
   * the reference cell to the real cell.
   */
  std::shared_ptr<Mapping<dim>> mapping;

  /*!
   * @brief Stream object which only prints output for one MPI process.
   */
  std::shared_ptr<ConditionalOStream>         pcout;

  /*!
   * @brief Class member used to monitor the compute time of parts of the simulation.
   */
  std::shared_ptr<TimerOutput>                computing_timer;

  /*!
   * Struct containing all the entities to be considered during a
   * solution transfer.
   */
  SolutionTransferContainer<dim>              container;

  /*!
   * @details Release all memory and return all objects to a state just like
   * after having called the default constructor.
   */
  virtual void clear();

  /*!
   * @brief Interpolates the @p function on the finite element space contained
   * in the @p entity and saves the result in the @p vector.
   */
  void interpolate_function
  (const Function<dim>                             &function,
   const std::shared_ptr<Entities::FE_FieldBase<dim>> entity,
   LinearAlgebra::MPI::Vector                      &vector);

  /*!
   * @brief Loads the initial conditions to the pertinent solution
   * vector
   * @details Projects the @ref function at simulation's start time
   * to the old_solution vector of the @ref entity. If @ref boolean is
   * set to true, the @ref function evaluated at start time and start time
   * plus one time step will be projected to the old_old_solution and
   * old_solution vectors respectively.
   * @warning If @ref boolean is set to true, one has to manually
   * advance the time to the first time step or else the solver will
   * have a time offset.
   * @attention Would it be preferable to interpolate the functions and
   * apply the constraints instead of projection?
   */
  void set_initial_conditions
  (std::shared_ptr<Entities::FE_FieldBase<dim>> entity,
   Function<dim>                              &function,
   const TimeDiscretization::VSIMEXMethod     &time_stepping,
   const bool                                 boolean = false);

  /*!
   *  @brief Computes the next time step according to the
   *  Courant-Friedrichs-Lewy (CFL) condition.
   *
   *  @details The next time step is given by
   * \f[
   *    \Delta t^{n-1}_\textrm{new} = \frac{C_\max}{C} \Delta t^{n-1}
   * \f]
   * where \f$ C_\max \f$ is the maximum CFL number and \f$ C\f$ is the
   * CFL number computed from the current velocity field.
   *  @attention The maximum Courant-Friedrichs-Lewy number is assumed
   * to be 1.0 if no value is passed.
   */
  virtual double compute_next_time_step
  (const TimeDiscretization::VSIMEXMethod &time_stepping,
   const double                           cfl_number,
   const double                           max_cfl_number = 1.0) const;

  /*!
   * @brief Performs an adaptive mesh refinement.
   * @details
   */
  void adaptive_mesh_refinement();
};

// inline functions
template<int dim>
inline void SolutionTransferContainer<dim>::clear()
{
  entities.clear();
}

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_*/
