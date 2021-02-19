#ifndef INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_
#define INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_

#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/navier_stokes_projection.h>

#include <memory>

namespace RMHD
{

/*!
 * @class HydrodynamicProblem
 *
 * @brief The class contains instances and methods that are common
 * and / or useful for most problems to be formulated.
 */
template <int dim>
class HydrodynamicProblem : public Problem<dim>
{
public:
  HydrodynamicProblem(const RunTimeParameters::HydrodynamicProblemParameters &parameters);

  /*
   * @brief This methods starts a simulation.
   */
  void run();

  /*!
   * @brief This methods continues a simulation until the final time or the
   * maximum number of steps is reached.
   *
   * @attention This method only continues a simulation if the parameters of the
   * @ref parameter are modified accordingly.
   */
  void continue_run();

protected:
  const RunTimeParameters::HydrodynamicProblemParameters &parameters;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  /*!
   * @brief The current Courant number.
   */
  double                                        cfl_number;

  /*!
   * @details Release all memory and return all objects to a state just like
   * after having called the default constructor.
   */
  virtual void clear() override;

  /*!
   * @brief This methods creates the initial mesh.
   *
   * @details This method is only called at the beginning of a simulation.
   */
  virtual void make_grid() = 0;

  /*!
   * @brief This methods restarts a simulation from a serialized checkpoint.
   *
   * @details The parameters are nevertheless those being read from the
   * parameter file. This method is only called at the beginning of a
   * simulation.
   */
  void restart(const std::string &fname);

  /*!
   * @brief This initializes the problem such that the solution at the fictitious
   * previous time step is specified according the @p function. The @ref
   * time_stepping is also initialized accordingly.
   *
   * @details This method is only called at the beginning of a simulation.
   *
   */
  void initialize_from_function(Function<dim> &velocity_function,
                                Function<dim> &pressure_function,
                                const double   previous_step_size);

  /*!
   * @brief Virtual method to allow the user to run some postprocessing methods.
   * The default implementation does nothing.
   *
   */
  virtual void postprocess_solution();

  /*!
   * @brief Virtual method to allow the user to run some postprocessing methods
   * at the end of a simulation. The default implementation does nothing.
   *
   * @details This method is called at the end of @ref run and @ref continue_run.
   *
   */
  virtual void save_postprocessing_results();

  /*!
   * @brief Purely virtual method to setup the boundary conditions of the
   * simulation.
   */
  virtual void setup_boundary_conditions() = 0;

  /*!
   * @brief Purely virtual method to setup the initial conditions of the @ref
   * velocity and, possibly, the @ref pressure.
   *
   * @details This method is only called at the beginning of a simulation.
   */
  virtual void setup_initial_conditions() = 0;

private:
  /*!
   * @brief This method loads a checkpoint of a previous simulation from the
   * file system. This process is referred to as deserialization.
   */
  void deserialize(const std::string &fname);

  /*!
   * @brief This method saves a previous simulation to the file system. This
   * process is referred to as serialization.
   */
  void serialize(const std::string &fname) const;

  /*!
   * @brief This method performs a setup of the degrees of freedom of the
   * @ref velocity and the @ref pressure.
   *
   * @details This method is called at the beginning of a simulation and when
   * the mesh is refined.
   */
  void setup_dofs();

  void time_loop(const unsigned int n_steps);

  /*!
   * @brief This method saves the current solution as VTK output.
   */
  void output_results() const;

  /*!
   * @brief This method updates the solution vectors of the @ref velocity and
   * the @ref pressure.
   *
   * @details This method is commonly called when the timestep is advanced.
   */
  void update_solution_vectors();

};

// inline functions
template<int dim>
inline void HydrodynamicProblem<dim>::postprocess_solution()
{
  return;
}

template<int dim>
inline void HydrodynamicProblem<dim>::save_postprocessing_results()
{
  return;
}

template<int dim>
inline void HydrodynamicProblem<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_ */
