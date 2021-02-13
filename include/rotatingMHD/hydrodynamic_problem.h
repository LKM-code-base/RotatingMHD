#ifndef INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_
#define INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_

#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/navier_stokes_projection.h>

#include <memory>

namespace RMHD
{

/*!
 * @class Problem
 * @brief The class containts instances and methods that are common
 * and/or useful for most problems to be formulated.
 */
template <int dim>
class HydrodynamicProblem : public Problem<dim>
{
public:
  HydrodynamicProblem(const RunTimeParameters::HydrodynamicProblemParameters &parameters);

  void run();

protected:
  const RunTimeParameters::HydrodynamicProblemParameters &parameters;

  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  double                                        cfl_number;

  virtual void make_grid() = 0;

  void restart(const std::string &fname);

  void restart_from_function(const double old_step_size,
                             const double old_old_step_size);

  void output_results() const;

  virtual void postprocess_solution();

  virtual void setup_boundary_conditions() = 0;

  void setup_dofs();

  virtual void setup_initial_conditions() = 0;

  void update_solution_vectors();
};

// inline functions
template<int dim>
inline void HydrodynamicProblem<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_HYDRODYNAMIC_PROBLEM_H_ */
