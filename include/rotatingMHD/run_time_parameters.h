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

enum class ConvectionTermForm
{
  standard,
  skewsymmetric,
  divergence,
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

  ProjectionMethod    projection_method;
  ConvectionTermForm  convection_term_form;

  double              Re;
  double              Pe;

  unsigned int        n_global_refinements;

  unsigned int        p_fe_degree;
  unsigned int        temperature_fe_degree;

  unsigned int        n_maximum_iterations;
  unsigned int        solver_krylov_size;
  unsigned int        solver_off_diagonals;
  unsigned int        solver_update_preconditioner;
  double              relative_tolerance;
  double              solver_diag_strength;

  bool                verbose;
  bool                flag_semi_implicit_convection;

  unsigned int        graphical_output_interval;
  unsigned int        terminal_output_interval;
  unsigned int        adaptive_meshing_interval;
  
  unsigned int        refinement_and_coarsening_max_level;
  unsigned int        refinement_and_coarsening_min_level;

  bool                flag_spatial_convergence_test;
  unsigned int        initial_refinement_level;
  unsigned int        final_refinement_level;
  unsigned int        temporal_convergence_cycles;
  double              time_step_scaling_factor;
};

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
