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
 * @brief Enumeration for the different types of problems.
 */
enum class ProblemType
{
  /*!
   * @todo Documentation is missing.
   */
  hydrodynamic,

  /*!
   * @todo Documentation is missing.
   */
  heat_convection_diffusion,

  /*!
   * @todo Documentation is missing.
   */
  boussinesq,

  /*!
   * @todo Documentation is missing.
   */
  rotating_boussinesq,

  /*!
   * @todo Documentation is missing.
   */
  rotating_magnetohydrodynamic
};



/*!
 * @brief Enumeration for convergence test type.
 */
enum class ConvergenceTestType
{
  /*!
   * @todo Documentation is missing.
   */
  spatial,

  /*!
   * @todo Documentation is missing.
   */
  temporal
};



/*!
 * @brief Enumeration for incremental pressure-correction scheme types.
 */
enum class PressureCorrectionScheme
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
   * (\nabla\otimes\bs{v})\cdot\bs{w}\dint{V}\f$,
   * where \f$\bs{w}\f$ denotes a test function.
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



/*!
 * @struct RefinementParameters
 *
 * @brief @ref RefinementParameters contains parameters which are
 * related to the adaptive refinement of the mesh.
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
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const RefinementParameters &prm);

  /*!
   * @brief Boolean flag to enable or disable adaptive mesh refinement.
   */
  bool          adaptive_mesh_refinement;

  /*!
   * @brief The frequency at which an adaptive mesh refinement is performed.
   */
  unsigned int  adaptive_mesh_refinement_frequency;

  /*!
   * @brief The upper fraction of the total number of cells set to
   * coarsen.
   */
  double        cell_fraction_to_coarsen;

  /*!
   * @brief The lower fraction of the total number of cells set to
   * refine.
   */
  double        cell_fraction_to_refine;

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
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const RefinementParameters &prm);



/*!
 * @struct OutputControlParameters
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
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const OutputControlParameters &prm);

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
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const OutputControlParameters &prm);



/*!
 * @struct ConvergenceTestParameters
 */
struct ConvergenceTestParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvergenceTestParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ConvergenceTestParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const ConvergenceTestParameters &prm);

  /*!
   * @brief The type of convergence test (spatial or temporal).
   */
  ConvergenceTestType convergence_test_type;

  /*!
   * @brief Number of initial global mesh refinements.
   */
  unsigned int        n_global_initial_refinements;

  /*!
   * Number of spatial convergence cycles.
   */
  unsigned int        n_spatial_convergence_cycles;

  /*!
   * @brief Factor \f$ s \f$ of the reduction of the timestep between two
   * subsequent levels, *i. e.*, \f$ \Delta t_{l+1} = s \Delta t_l\f$.
   *
   * @details The factor \f$ s \f$ must be positive and less than unity.
   */
  double              timestep_reduction_factor;

  /*!
   * @brief Number of temporal convergence cycles.
   */
  unsigned int        n_temporal_convergence_cycles;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvergenceTestParameters &prm);



/*!
 * @struct LinearSolverParameters
 *
 * @brief A structure containing all parameters relevant for the solution of
 * linear systems using a Krylov subspace method.
 *
 * @todo Proper initiation of the solver_name string without constructor
 * ambiguity
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
  double        relative_tolerance;

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
  double        absolute_tolerance;

  /*!
   * @brief Maximum number of iterations to be performed by the linear solver.
   *
   * @details If the tolerance is fulfilled within the maximum number of
   * iterations, an error is thrown and the simulation is aborted. In this case,
   * the preconditioner might require some optimization.
   *
   */
  unsigned int  n_maximum_iterations;

private:

  /*!
   * @brief The name of the solver.
   */
  std::string   solver_name;
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
 * @struct DimensionlessNumbers
 *
 * @brief A structure containing all the dimensionless numbers.
 */
struct DimensionlessNumbers
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  DimensionlessNumbers();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  DimensionlessNumbers(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm);

  /*!
   * @brief The Reynolds number
   */
  double  Re;

  /*!
   * @brief The Prandtl number
   */
  double  Pr;

  /*!
   * @brief The Peclet number
   */
  double  Pe;

  /*!
   * @brief The Rayleigh number
   */
  double  Ra;

  /*!
   * @brief The Ekman number
   */
  double  Ek;

  /*!
   * @brief The magnetic Prandtl number
   */
  double  Pm;

private:

  ProblemType problem_type;
};



/*!
 * @struct NavierStokesParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct NavierStokesParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  NavierStokesParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  NavierStokesParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const NavierStokesParameters &prm);


  /*!
   * @brief Enumerator controlling the incremental pressure-correction
   * scheme is to be implemented.
   */
  PressureCorrectionScheme          pressure_correction_scheme;

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is to be implemented
   */
  ConvectiveTermWeakForm            convective_term_weak_form;

  /*!
   * @brief Enumerator controlling which time discretization of the
   * convective term is to be implemented
   */
  ConvectiveTermTimeDiscretization  convective_term_time_discretization;

  /*!
   * @brief The factor multiplying the Coriolis acceleration.
   */
  double                            C1;

    /*!
   * @brief The factor multiplying the velocity's laplacian.
   */
  double                            C2;

    /*!
   * @brief The factor multiplying the bouyancy term.
   */
  double                            C3;

    /*!
   * @brief The factor multiplying the electromagnetic force.
   */
  double                            C5;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            diffusion_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            projection_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            correction_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * poisson pre-step.
   */
  LinearSolverParameters            poisson_prestep_solver_parameters;

  /*!
   * @brief Specifies the frequency of the update of the diffusion
   * preconditioner.
   */
  unsigned int                      preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                              verbose;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const NavierStokesParameters &prm);



/*!
 * @struct HeatEquationParameters
 *
 * @brief A structure containing all the parameters of the heat
 * equation solver.
 */
struct HeatEquationParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  HeatEquationParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  HeatEquationParameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const HeatEquationParameters &prm);

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is to be implemented
   */
  ConvectiveTermWeakForm            convective_term_weak_form;

  /*!
   * @brief Enumerator controlling which time discretization of the
   * convective term is to be implemented
   */
  ConvectiveTermTimeDiscretization  convective_term_time_discretization;

    /*!
   * @brief The factor multiplying the temperature's laplacian.
   */
  double                            C4;

  /*!
   * @brief The parameters for the linear solver.
   */
  LinearSolverParameters            solver_parameters;

  /*!
   * @brief Specifies the frequency of the update of the solver's
   * preconditioner.
   */
  unsigned int                      preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                              verbose;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const HeatEquationParameters &prm);



/*!
 * @struct ProblemParameters
 */
struct ProblemParameters
    : public OutputControlParameters,
      public DimensionlessNumbers
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ProblemParameters(const std::string &parameter_filename,
                    const bool        flag = false);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
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
  ProblemType                                 problem_type;

  /*!
   * @brief Spatial dimension of the problem.
   */
  unsigned int                                dim;

  /*!
   * @brief Polynomial degree of the mapping.
   */
  unsigned int                                mapping_degree;

  /*!
   * @brief Boolean indicating if the mapping is to be asigned to
   * the interior cells too.
   */
  bool                                        mapping_interior_cells;

  /*!
   * @brief The polynomial degree of the pressure's finite element.
   */
  unsigned int                                fe_degree_pressure;

  /*!
   * @brief The polynomial degree of the velocity's finite element.
   */
  unsigned int                                fe_degree_velocity;

  /*!
   * @brief The polynomial degree of the temperature's finite element.
   */
  unsigned int                                fe_degree_temperature;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                                        verbose;

  /*!
   * @brief Parameters of the convergence test.
   */
  ConvergenceTestParameters                   convergence_test_parameters;

  /*!
   * @brief Parameters of the adaptive mesh refinement.
   */
  RefinementParameters                        refinement_parameters;

  /*!
   * @brief Parameters of the time stepping scheme.
   */
  TimeDiscretization::TimeSteppingParameters  time_stepping_parameters;

  /*!
   * @brief Parameters of the Navier-Stokes solver.
   */
  NavierStokesParameters                      navier_stokes_parameters;

  /*!
   * @brief Parameters of the heat equation solver.
   */
  HeatEquationParameters                      heat_equation_parameters;

private:

  bool                                        flag_convergence_test;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm);



} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
