#ifndef INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_
#define INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_

#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

namespace RMHD
{

using namespace dealii;

template <int dim>
class Problem 
{
public:
  /*!
   * Default constructor which initializes the member variables.
   */
  Problem();

protected:
  /*!
   * The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm      mpi_communicator;

  /*!
   * Triangulation object of the problem.
   */
  parallel::distributed::Triangulation<dim>   triangulation;

  /*!
   * Stream object which only prints output for one MPI process.
   */
  std::shared_ptr<ConditionalOStream>         pcout;

  /*!
   * Class member used to monitor the compute time of parts of the simulation.
   */
  std::shared_ptr<TimerOutput>                computing_timer;

protected:

  /*!
   * @attention Purpose of the method is not clear...
   */
  void set_initial_conditions
  (Entities::EntityBase<dim>              &entity,
   Function<dim>                    &function,
   const TimeDiscretization::VSIMEXMethod &time_stepping);

  /*!
   * @brief Computes the error of the numerical solution against
   * the analytical solution.
   * @details The error is calculated by subtracting the /f$ L_2/f$
   * projection of the given function from the solution vector and
   * computing the absolute value of the residum.
   */
  void compute_error
  (LinearAlgebra::MPI::Vector  &error_vector,
   Entities::EntityBase<dim>   &entity,
   Function<dim>               &exact_solution);

};

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_*/
