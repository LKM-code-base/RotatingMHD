/*
 * navier_stokes_parameters.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PARAMETERS_H_

#include <rotatingMHD/general_parameters.h>

#include <rotatingMHD/time_discretization.h>

#include <string>

namespace RMHD
{

namespace RunTimeParameters
{

/*!
 * @brief Enumeration for the type of the pressure update of the pressure
 * projection algorithm.
 */
enum class ProjectionMethod
{
  /*!
   * @todo Documentation is missing.
   */
  standard,
  /*!
   * @todo Documentation is missing.
   */
  rotational
};

/*!
 * @brief Enumeration for the weak form of the non-linear convective term.
 */
enum class ConvectiveTermWeakForm
{
  /*!
   * @brief Standard form, *i. e.*, \f$\int\bs{v}\cdot
   * (\nabla\otimes\bs{v})\cdot\bs{w}\dint{V}\f$, where \f$\bs{w}\f$ denotes a test function.
   */
  standard,
  /*!
   * @todo Documentation is missing.
   */
  skewsymmetric,
  /*!
   * @todo Documentation is missing.
   */
  divergence,
  /*!
   * @todo Documentation is missing.
   */
  rotational
};

/*!
 * @brief Enumeration for the type of the temporal discretization of the
 * non-linear convective term.
 */
enum class ConvectiveTermTimeDiscretization
{
  /*!
   * @brief Semi-implicit treatment in time, *i. e.*, \f$\bs{v}^*\cdot
   * \nabla\phi^n\f$, where \f$\bs{v}^*\f$ is an extrapolated velocity.
   */
  semi_implicit,
  /*!
   * @brief Explicit treatment in time, *i. e.*, \f$\bs{v}^{n-1}\cdot
   * \nabla\phi^{n-1}+\cdots\f$.
   */
  fully_explicit
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
  ConvectiveTermWeakForm  convection_term_form;

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

/*!
 * @struct NavierStokesParameters
 *
 * @brief @ref NavierStokesParameters contains the parameters which control the
 * behavior of @ref NavierStokesProjection.
 */
struct NavierStokesDiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  NavierStokesDiscretizationParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  NavierStokesDiscretizationParameters(const std::string &parameter_filename);

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
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const NavierStokesDiscretizationParameters &prm);

  /*!
   * @brief Variable controlling the type of the pressure projection method.
   *
   * @attention SG thinks that this parameter just controls the type of the
   * pressure update. Maybe a re-naming is appropriate.
   *
   * @details See @ref ProjectionMethod for details.
   *
   */
  ProjectionMethod        projection_method;

  /*!
   * @brief Variable controlling the type of the weak form of the convective
   * term.
   *
   * @details See @ref ConvectiveTermWeakForm for details.
   *
   */
  ConvectiveTermWeakForm  convective_weak_form;

  /*!
   * @brief Variable controlling the type of the temporal discretization of
   * the convective term.
   *
   * @details See @ref ConvectiveTermTimeDiscretization for details.
   *
   */
  ConvectiveTermTimeDiscretization  convective_temporal_form;

  /*!
   * @brief Specifies the frequency of the update of the diffusion
   * preconditioner.
   */
  unsigned int  preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool    verbose;

  /*!
   * @brief The Reynolds number of the problem.
   *
   * @details This parameter is not set by
   * @ref parse_parameters and not declared by @ref declare_paramters. Instead,
   * the Reynolds number should be declared and set in a superordinate parameter
   * object.
   *
   * @todo SG thinks that we want to have equation coefficients instead of
   * a single parameter. This would allow us to use several types of dimensionless
   * equations.
   *
   */
  double  Re;

  /*!
   * @brief A structure containing all parameters relevant for the solution of
   * linear systems using a Krylov subspace method.
   *
   * @details See @ref LinearSolverParameters for details.
   *
   */
  LinearSolverParameters  linear_solver_control;

};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream,
                   const NavierStokesDiscretizationParameters &prm);

/*!
 * @struct NavierStokesProblemParameters
 *
 * @brief Structure containing all parameter which control a Navier-Stokes
 * problem. This set of parameters is used for example in @ref DFG and
 * @ref Step35.
 *
 */
struct NavierStokesProblemParameters : public ProblemParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  NavierStokesProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  NavierStokesProblemParameters(const std::string &parameter_filename);

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
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const NavierStokesProblemParameters &prm);

  /*!
   *
   * @brief Polynomial degree \f$ p \f$ of the \f$Q_{p+1}-Q_p\f$ finite element
   * of the velocity and the pressure space.
   * @brief Specifies the frequency of the update of the diffusion
   * preconditioner.
   */
  unsigned int  fe_degree;

  /*!
   * @brief Controls the discretization of the problem.
   *
   * @details See @ref NavierStokesDiscretizationParameters for details.
   */
  NavierStokesDiscretizationParameters  navier_stokes_discretization;

};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const NavierStokesProblemParameters &prm);

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
