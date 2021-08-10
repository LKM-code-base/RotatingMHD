#ifndef INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_
#define INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_

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



} // namespace RMHD



#endif /*INCLUDE_ROTATINGMHD_SOLVER_CLASS_H_*/
