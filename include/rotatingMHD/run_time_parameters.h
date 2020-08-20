/*
 * data_storage.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace RMHD
{
  using namespace dealii;

namespace RunTimeParameters
{
enum class ProjectionMethod
{
  standard,
  rotational
};

class ParameterSet
{
public:
  ProjectionMethod  projection_method;
  double            dt;
  double            t_0;
  double            T;
  double            vsimex_input_gamma;
  double            vsimex_input_c;
  double            Re;
  unsigned int      n_global_refinements;
  unsigned int      p_fe_degree;
  unsigned int      solver_max_iterations;
  unsigned int      solver_krylov_size;
  unsigned int      solver_off_diagonals;
  unsigned int      solver_update_preconditioner;
  double            solver_tolerance;
  double            solver_diag_strength;
  bool              flag_verbose_output;
  bool              flag_adaptive_time_step;
  bool              flag_DFG_benchmark;
  unsigned int      graphical_output_interval;
  unsigned int      terminal_output_interval;

  ParameterSet();
  void read_data_from_file(const std::string &filename);

protected:
  ParameterHandler  prm;
};

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
