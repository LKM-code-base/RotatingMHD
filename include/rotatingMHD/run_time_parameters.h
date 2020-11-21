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
enum class ConvectionTermForm
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
enum class ConvectiveTermDiscretizationType
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

/*!
 * @struct DiscretizationParameters
 *
 * @brief @ref DiscretizationParameters contains parameters which are related to
 * the output of the solver and the control of the refinement of the
 * mesh.
 */
struct DiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  DiscretizationParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  DiscretizationParameters(const std::string &parameter_filename);

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
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const DiscretizationParameters &prm);

  /*!
   * @brief The frequency at which a graphical output file (vtk output) is
   * written to the file system.
   */
  unsigned int  graphical_output_frequency;

  /*!
   * @brief The frequency at which diagnostics are written the terminal.
   */
  unsigned int  terminal_output_frequency;

  /*!
   * @brief Directory where the graphical output should be written.
   */
  std::string   graphical_output_directory;

  /*!
   * @brief Boolean flag to enable or disable adaptive mesh refinement.
   */
  bool          adaptive_mesh_refinement;

  /*!
   * @brief The frequency at which an adaptive mesh refinement is performed.
   */
  unsigned int  adaptive_mesh_refinement_frequency;

  /*!
   * @brief The number of maximum levels of the mesh. This parameter prohibits
   * a further refinement of the mesh.
   */
  unsigned int  n_maximum_levels;

  /*!
   * @brief The number of minimum levels of the mesh. This parameter prohibits
   * a further coarsening of the mesh.
   */
  unsigned int  n_minimum_levels;

  /*!
   * @brief The number of adaptive initial refinement steps. This parameter
   * determines the number of refinements which are based on the spatial structure
   * of the initial condition.
   */
  unsigned int  n_adaptive_initial_refinements;

  /*!
   * @brief The number of global initial refinement steps.
   */
  unsigned int  n_global_initial_refinements;

   /*!
   * @brief The number of initial refinement steps of cells at the boundary.
   */
  unsigned int  n_boundary_initial_refinements;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool          verbose;

};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const DiscretizationParameters &prm);

/*!
 * @struct LinearSolverParameters
 *
 * @brief A structure containing all parameters relevant for the solution of
 * linear systems using a Krylov subspace method.
 */
struct  LinearSolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  LinearSolverParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  LinearSolverParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the time stepping scheme from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const LinearSolverParameters &prm);

  /*!
   * @brief Relative tolerance.
   *
   * @details This parameter specifies the desired relative reduction of the
   * initial residual of the linear system. The initial residual is given by
   *
   * \f[
   *    r_0=|\!| \mathbf{A}\cdot\mathbf{x}_0-\mathbf{b}|\!|_2\,.
   * \f]
   *
   * If the relative tolerance is applied, the final solution of the linear
   * system is such that
   *
   * \f[
   *    |\!| \mathbf{A}\cdot\mathbf{x}-\mathbf{b}|\!|_2\leq\mathrm{tol}\,r_0\,.
   * \f]
   *
   */
  double  relative_tolerance;

  /*!
   * @brief Absolute tolerance.
   *
   * @details This parameter specifies the desired absolute value of the
   * residual of the linear system.
   *
   * If the absolute tolerance is applied, the final solution of the linear
   * system is such that
   *
   * \f[
   *    |\!| \mathbf{A}\cdot\mathbf{x}-\mathbf{b}|\!|_2\leq\mathrm{tol}\,.
   * \f]
   *
   */
  double  absolute_tolerance;

  /*!
   * @brief Maximum number of iterations to be performed by the linear solver.
   *
   * @details If the tolerance is fulfilled within the maximum number of
   * iterations, an error is thrown and the simulation is aborted. In this case,
   * the preconditioner might require some optimization.
   *
   */
  unsigned int  n_maximum_iterations;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const LinearSolverParameters &prm);

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
