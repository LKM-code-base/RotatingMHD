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
  ConditionalOStream  pcout;

  /*!
   * Class member used to monitor the compute time of parts of the simulation.
   */
  TimerOutput         computing_timer;

protected:

  /*!
   * @attention Purpose of the method is not clear...
   */
  void set_initial_conditions
  (Entities::EntityBase<dim>              &entity,
   Function<dim>                    &function,
   const TimeDiscretization::VSIMEXMethod &time_stepping);

};

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_PROBLEM_CLASS_H_*/
