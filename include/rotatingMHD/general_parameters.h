/*
 * general_parameters.h
 *
 *  Created on: Nov 23, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_GENERAL_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_GENERAL_PARAMETERS_H_

#include <rotatingMHD/time_discretization.h>

#include <string>

namespace RMHD
{

using namespace dealii;

namespace RunTimeParameters
{

using namespace TimeDiscretization;

/*!
 * @struct DiscretizationParameters
 *
 * @brief @ref DiscretizationParameters contains parameters which are related to
 * the output of the solver and the control of the refinement of the
 * mesh.
 */
struct RefinementParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  RefinementParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  RefinementParameters(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const RefinementParameters &prm);

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
  unsigned int  n_initial_adaptive_refinements;

  /*!
   * @brief The number of global initial refinement steps.
   */
  unsigned int  n_initial_global_refinements;

   /*!
   * @brief The number of initial refinement steps of cells at the boundary.
   */
  unsigned int  n_initial_boundary_refinements;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const RefinementParameters &prm);

/*!
 * @struct OutputControlParameters
 *
 */
struct OutputControlParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  OutputControlParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  OutputControlParameters(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const OutputControlParameters &prm);

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
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const OutputControlParameters &prm);

/*!
 * @struct ProblemParameters
 */
struct ProblemParameters
    : public OutputControlParameters,
      public RefinementParameters,
      public TimeSteppingParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ProblemParameters(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const ProblemParameters &prm);

  /*!
   * @brief Spatial dimension of the problem.
   */
  unsigned int dim;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool          verbose;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm);

/*!
 * @struct LinearSolverParameters
 *
 * @brief A structure containing all parameters relevant for the solution of
 * linear systems using a Krylov subspace method.
 */
struct LinearSolverParameters
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
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
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
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const LinearSolverParameters &prm);

/*!
 * @struct ConvergenceAnalysisParameters
 *
 * @brief @ref ConvergenceAnalysisParameters contains all parameters which
 * control those of a spatial and/or a temporal convergence analysis.
 *
 */
struct ConvergenceAnalysisParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvergenceAnalysisParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ConvergenceAnalysisParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   *
   * @todo Implementation is missing.
   *
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the time stepping scheme from
   * the ParameterHandler object @p prm.
   *
   * @todo Implementation is missing.
   *
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Flag indicating whether a spatial convergence test should be
   * performed.
   */
  bool        spatial_convergence_test;

  /*!
   * @brief Number of initial global mesh refinements.
   */
  unsigned int  n_global_initial_refinements;

  /*!
   * Number of spatial convergence cycles.
   */
  unsigned int  n_spatial_convergence_cycles;

  /*!
   * @brief Flag indicating whether a temporal convergence test should be
   * performed.
   */
  bool        temporal_convergence_test;

  /*!
   * @brief Number of temporal convergence cycles.
   */
  unsigned int  n_temporal_convergence_cycles;

  /*!
   * @brief Factor \f$ s \f$ of the reduction of the timestep between two
   * subsequent levels, *i. e.*, \f$ \Delta t_{l+1} = s \Delta t_l\f$.
   *
   * @details The factor \f$ s \f$ must be positive and less than unity.
   */
  double  timestep_reduction_factor;

};

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_GENERAL_PARAMETERS_H_ */
