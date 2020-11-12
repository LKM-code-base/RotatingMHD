#ifndef INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_
#define INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_

#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

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
   * @ref Entities::EntityBase instance and a boolean.
   * @details The boolean indicates wheter the entity is to be 
   * considered by the error estimation or not.
   */ 
  using EntityEntry = std::pair<Entities::EntityBase<dim> *, bool>;

  /*!
   * @brief A std::vector with all the entities to be considered in
   * a solution transfer
   */ 
  std::vector<EntityEntry>  entities;

  /*!
   * @brief Default constructor.
   */ 
  SolutionTransferContainer();

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
   * @brief Adds the passed on EntityBase instance and flag to the
   * entities struct member.
   * @details If no boolean is passed, it is assumed that the entity
   * is to be considered by the error estimation.
   */ 
  void add_entity(std::shared_ptr<Entities::EntityBase<dim>> entity, bool flag = true);

private:

  /*!
   * @brief The side of the std::vector instance containing all the
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
  Problem(const RunTimeParameters::ParameterSet &prm);

protected:
  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                              mpi_communicator;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const RunTimeParameters::ParameterSet      &prm;

  /*!
   * @brief Triangulation object of the problem.
   */
  parallel::distributed::Triangulation<dim>   triangulation;

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

protected:

  /*!
   * @attention Purpose of the method is not clear...
   */
  void set_initial_conditions
  (std::shared_ptr<Entities::EntityBase<dim>> entity,
   Function<dim>                              &function,
   const TimeDiscretization::VSIMEXMethod     &time_stepping);

  void set_initial_conditions
  (std::shared_ptr<Entities::EntityBase<dim>>              entity,
   Function<dim>                          &function,
   const TimeDiscretization::VSIMEXMethod &time_stepping);

  /*!
   * @brief Computes the error of the numerical solution against
   * the analytical solution.
   * @details The error is calculated by subtracting the /f$ L_2/f$
   * projection of the given function from the solution vector and
   * computing the absolute value of the residum.
   */

  void compute_error
  (LinearAlgebra::MPI::Vector                 &error_vector,
   std::shared_ptr<Entities::EntityBase<dim>> entity,
   Function<dim>                              &exact_solution);

  void compute_error
  (LinearAlgebra::MPI::Vector  &error_vector,
   std::shared_ptr<Entities::EntityBase<dim>>   entity,
   Function<dim>               &exact_solution);

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

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_*/
