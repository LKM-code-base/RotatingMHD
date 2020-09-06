/*
 * data_storage.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

namespace RunTimeParameters
{

enum class ProjectionMethod
{
  standard,
  rotational
};

struct ParameterSet
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ParameterSet();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ParameterSet(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the time stepping scheme from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * Member variable which contains all parameters related to the time
   * discretization.
   */
  TimeDiscretization::TimeSteppingParameters  time_stepping_parameters;

  ProjectionMethod  projection_method;

  double            Re;

  unsigned int      n_global_refinements;

  unsigned int      p_fe_degree;

  unsigned int      n_maximum_iterations;
  unsigned int      solver_krylov_size;
  unsigned int      solver_off_diagonals;
  unsigned int      solver_update_preconditioner;
  double            relative_tolerance;
  double            solver_diag_strength;

  bool              flag_verbose_output;
  bool              flag_DFG_benchmark;

  unsigned int      graphical_output_interval;
  unsigned int      terminal_output_interval;

};

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
