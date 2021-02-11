#ifndef INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/time_discretization.h>

#include <memory>

namespace RMHD
{


/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{
/*!
 * @brief Enumeration for the different types of problems.
 *
 * @details The different problems derive from the stencil
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + C_1 \bs{\Omega} \times \bs{v}
    = -C_6 \nabla \bar{p}
    + C_ 2\nabla^2 \bs{v}
    - C_3 \vartheta \bs{g}
    + C_5 (\nabla \times \bs{B}) \times \bs{B},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    C_4 \nabla^2 \vartheta, \\
    \nabla \cdot \bs{B} = 0, \quad
    \pd{\bs{B}}{t} = \nabla \times ( \bs{v} \times \bs{B} ) + C_5 \nabla \times ( \nabla \times \bs{B} ),    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \bs{B} \f$ the magnetic field and
  * \f$ {C_i} \f$ the dimensionless coefficients. These coefficients
  * are given by a combination of dimensionless numbers depending of
  * the problem type.
  * @note For a definition of the pertinent dimensionless numbers see
  * @ref DimensionlessNumbers
 */
enum class ProblemType
{
  /*!
   * @brief The Navier-Stokes equations in an intertial frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
    = - \nabla \bar{p}
    + \frac{1}{\Reynolds} \nabla^2 \bs{v}
    + \bs{f},
    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \rho_0 \f$ the density of the reference state,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Reynolds \f$ the Reynolds number and
  * \f$ \bs{f} \f$ the body force,
  */
  hydrodynamic,

  /*!
   * @brief The heat convection-difussion equation.
   * @details Given by
  * \f[
  * \begin{gather*}
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Peclet} \nabla^2 \vartheta + r, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \Peclet \f$ the Peclet number and
  * \f$ r \f$ the internal heat generation.
  */
  heat_convection_diffusion,

  /*!
   * @brief Boussinesq-approximation in an interial frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
    = - \nabla \bar{p}
    + \sqrt{\frac{\Prandtl}{\Rayleigh}} \nabla^2 \bs{v}
    - \vartheta \bs{g},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\sqrt{\Rayleigh \Prandtl}} \nabla^2 \vartheta, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \Rayleigh \f$ the Rayleigh number and
  * \f$ \vartheta \f$ the temperature.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  boussinesq,

  /*!
   * @brief Boussinesq-approximation in an rotating frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + \frac{2}{\Ekman} \bs{\Omega} \times \bs{v}
    = -\dfrac{1}{\Ekman} \nabla \bar{p}
    + \nabla^2 \bs{v}
    - \frac{\Rayleigh}{\Prandtl} \vartheta \bs{g},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Prandtl} \nabla^2 \vartheta, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bs{\Omega} \f$ the angular velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Ekman \f$ the Ekman number,
  * \f$ \Rayleigh \f$ the Rayleigh number,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \vartheta \f$ the temperature.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  rotating_boussinesq,

  /*!
   * @brief Boussinesq-approximation with electromagnetic forces in an rotating frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + \frac{2}{\Ekman} \bs{\Omega} \times \bs{v}
    = -\dfrac{1}{\Ekman} \nabla \bar{p}
    + \nabla^2 \bs{v}
    - \frac{\Rayleigh}{\Prandtl} \vartheta \bs{g}
    + \frac{1}{\magPrandtl}(\nabla \times \bs{B}) \times \bs{B},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Prandtl} \nabla^2 \vartheta, \\
    \nabla \cdot \bs{B} = 0, \quad
    \pd{\bs{B}}{t} = \nabla \times ( \bs{v} \times \bs{B} ) + \frac{1}{\magPrandtl} \nabla \times ( \nabla \times \bs{B} ),    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bs{\Omega} \f$ the angular velocity,
  * \f$ \Ekman \f$ the Ekman number,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Rayleigh \f$ the Rayleigh number,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \bs{B} \f$ the magnetic field and
  * \f$ \magPrandtl \f$ the magnetic Prandtl number.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  rotating_magnetohydrodynamic
};



/*!
 * @brief Enumeration for convergence test type.
 */
enum class ConvergenceTestType
{
  /*!
   * @brief Spatial convergence test.
   * @details Test to study the spatial discretization dependence of
   * convergence for a given problem.
   * @note Spatial convergence tests should be performed with a fine
   * time discretization, *i. e.*, a small enough time step.
   */
  spatial,

  /*!
   * @brief Temporal convergence test.
   * @details Test to study the temporal discretization dependence of
   * convergence for a given problem.
   * @note Temporal convergence tests should be performed with a fine
   * spatial discretization, *i. e.*, a triangulation with small enough cells.
   */
  temporal
};



/*!
 * @brief Enumeration for incremental pressure-correction scheme types.
 */
enum class PressureCorrectionScheme
{
  /*!
   * @brief Standard incremental pressure-correction scheme.
   * @details The pressure update is given by
   * \f[
   * p^{k} = p^{k-1} + \phi^{k}
   * \f]
   * @note See @ref NavierStokesProjection for a complete explanation
   * of the incremental pressure-correction scheme.
   */
  standard,

  /*!
   * @brief Rotational incremental pressure-correction scheme.
   * @details The pressure update is given by
   * \f[
   * p^{k} = p^{k-1} + \phi^{k} - \nu \nabla \cdot \bs{u}^k
   * \f]
   * @note See @ref NavierStokesProjection for a complete explanation
   * of the incremental pressure-correction scheme.
   */
  rotational
};



/*!
 * @brief Enumeration for the weak form of the non-linear convective term.
 * @attention These definitions are the ones I see the most in the literature.
 * Nonetheless Volker John and Helene Dallmann define the skew-symmetric
 * and the divergence form differently.
 */
enum class ConvectiveTermWeakForm
{
  /*!
   * @brief The standard form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v})] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  standard,

  /*!
   * @brief The skew-symmetric form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * (\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  skewsymmetric,

  /*!
   * @brief The divergence form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * \frac{1}{2}(\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  divergence,

  /*!
   * @brief The rotational form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ ( \nabla \times\bs{v}) \times \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   * @note This form modifies the pressure, *i. e.*,
   * \f[
   * \bar{p} = p + \frac{1}{2} \bs{v} \cdot \bs{v}.
   * \f]
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
   * @brief Semi-implicit treatment in time of the convective term, *i. e.*,
   * \f[
   * \bs{v}^{\star,k} \cdot
   * (\nabla \otimes\bs{\phi}^{k}),
   * \f]
   * where \f$\bs{v}^{\star, k}\f$ is the second order Taylor extrapolation
   * of the velocity at \f$t^k \f$ and \f$ \bs{\phi}^{k} \f$ is the
   * field being transported by the velocity.
   * @note The semi-implicit treatment for non-standard weak forms of
   * the convective terms is given
   * by
   * \f[
   * \bs{v}^{k} \cdot ( \nabla \otimes \bs{v}^{\star,k}) +
   * \frac{1}{2}(\nabla \cdot \bs{v}^{\star,k}) \bs{v}^{k},
   * \qquad
   * \bs{v}^{k} \cdot ( \nabla \otimes \bs{v}^{\star,k}) +
   * (\nabla \cdot \bs{v}^{\star,k}) \bs{v}^{k},
   * \qquad
   * ( \nabla \times\bs{v}^{\star,k}) \times \bs{v}^{k},
   * \f]
   * for the skew-symmetric, divergence and rotational form respectively.
   */
  semi_implicit,

  /*!
   * @brief Explicit treatment in time of the convective term, *i. e.*,
   * \f[
   * \sum_{j = 1} \beta_j
   * \bs{v}^{k-j} \cdot
   * (\nabla \otimes\bs{\phi}^{k-j}),
   * \f]
   * where \f$ \beta_j \f$ are time discretization coefficientes,
   * \f$\bs{v}^{k-j}\f$ the previous values of the velocity field and
   * \f$ \bs{\phi}^{k-j} \f$ the previous values of the
   * field being transported by the velocity.
   * @attention Why fully_explicit and not just explicit?
   */
  fully_explicit
};

/*!
 * @brief Enumeration for the type of the preconditioner to be used.
 */
enum class PreconditionerType
{
  /*!
   * @brief Incomplete LU decomposition preconditioning.
   */
  ILU,

  /*!
   * @brief Geometric multigrid preconditioning.
   *
   * @attention This is not implemented yet but may be helpful once the
   * MatrixFree framework is used.
   */
  GMG,

  /*!
   * @brief Algebraic multigrid preconditioning.
   */
  AMG,

  /*!
   * @brief Jacobi preconditioning.
   */
  Jacobi,

  /*!
   * @brief Symmetric Sucessive overrelaxation preconditioning. The system
   * matrix must be symmetric to apply this preconditioner.
   */
  SSOR

};

/*!
 * @struct SpatialDiscretizationParameters
 *
 * @brief @ref SpatialDiscretizationParameters contains parameters which are
 * related to the adaptive refinement of the mesh.
 */
struct SpatialDiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SpatialDiscretizationParameters();

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
                            const SpatialDiscretizationParameters &prm);

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
Stream& operator<<(Stream &stream, const SpatialDiscretizationParameters &prm);



/*!
 * @struct OutputControlParameters
 * @brief @ref OutputControlParameters contains parameters which are
 * related to the graphical and terminal output.
 */
struct OutputControlParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  OutputControlParameters();

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
 *
 * @brief @ref ConvergenceTestParameters contains parameters which are
 * related to convergence tests.
 */
struct ConvergenceTestParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvergenceTestParameters();

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
 * @struct PreconditionerParametersBase
 *
 * @brief A structure from which all other preconditioner parameters structures
 * are derived.
 */
struct PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionBaseParameters(const std::string        &name = "ILU",
                             const PreconditionerType &type = PreconditionerType::ILU);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the only the type parameters of
   * ::PreconditionBaseParameters from the ParameterHandler object @p prm.
   */
  static PreconditionerType parse_preconditioner_type(const ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of ::PreconditionBaseParameters
   * from the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief The type of the preconditioner used for solving the linear system.
   */
  PreconditionerType preconditioner_type;

  /*!
   * @brief The name of the preconditioner.
   */
  std::string       preconditioner_name;
};


/*!
 * @struct PreconditionRelaxationParameters
 *
 * @brief A structure containing all parameters relevant for Jacobi
 * preconditioning.
 */
struct PreconditionRelaxationParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionRelaxationParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the ILU preconditioner from
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
  friend Stream& operator<<(Stream &stream, const PreconditionRelaxationParameters &prm);

  /*!
   * @brief The relaxation parameter in the preconditioner.
   *
   * @attention For a Jacobi preconditioner, this parameter is only meaningful if the library is
   * compiled using the Trilinos linear algebra package.
   *
   */
  double        omega;

  /*!
   * @brief A parameter specifying how large the overlap of the local matrix
   * portions on each processor in a parallel application should be.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package and if an SOR or SSOR
   * preconditioner is used.
   *
   */
  unsigned int  overlap;


  /*!
   * @brief A parameter specifying how many times the given preconditioner
   * should be applied at each call.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  unsigned int  n_sweeps;
};

using PreconditionJacobiParameters = PreconditionRelaxationParameters;

using PreconditionSSORParameters = PreconditionRelaxationParameters;

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionRelaxationParameters &prm);

/*!
 * @struct PreconditionILUParameters
 *
 * @brief A structure containing all parameters relevant for the computation of
 * an incomplete LU preconditioner.
 *
 */
struct PreconditionILUParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionILUParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the ILU preconditioner from
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
  friend Stream& operator<<(Stream &stream, const PreconditionILUParameters &prm);

  /*!
   * @brief Relative tolerance used for the perturbation of the diagonal entries
   * of the diagonal matrix which meet yield a better preconditioning. For the
   * relative tolerance \f$\beta\f$ the condition \f$\beta\geq 1\f$ must hold.
   * The suggested value is \f$\beta\approx 1.01\f$.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  double        relative_tolerance;

  /*!
   * @brief Absolute tolerance used for the perturbation of the diagonal entries
   * of the diagonal matrix which meet yield a better preconditioning. For the
   * relative tolerance \f$\alpha\f$ the condition \f$\alpha>0\f$ must hold. The
   * suggested value is \f$10^{-5}\alpha\leq 10^{-2}\f$.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   */
  double        absolute_tolerance;

  /*!
   * @brief Fill-in level of the ILU. The ILU uses only the entries of the
   * sparsity pattern of the matrix \f$ A^k\f$ where \f$ k\f$ is the fill-in
   * level.
   */
  unsigned int  fill;

  /*!
   * @brief Overlap between the different MPI processes used for computing
   * the ILU.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   */
  unsigned int  overlap;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionILUParameters &prm);


/*!
 * @struct PreconditionAMGParameters
 *
 * @brief A structure containing the common parameters used in AMG
 * preconditioning.
 */
struct PreconditionAMGParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionAMGParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the AMG preconditioner from
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
  friend Stream& operator<<(Stream &stream, const PreconditionAMGParameters &prm);

  /*
   * @brief A control parameter for AMG preconditioning using the PETSc library.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the PETSc linear algebra package.
   *
   */
  double strong_threshold;

  /*
   * @brief A control parameter to control whether smoothed aggregation or
   * non-smoothed aggregration should be applied.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  bool  elliptic;

  /*
   * @brief A control parameter to control whether higher order finite elements
   * are used.
   *
   * @details The parameters cannot be specified through the parameter file but
   * is set according to the polynomial degree of the finite element.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  bool  higher_order_elements;

  /*
   * @brief A control parameter to control how many multigrid cycles should be
   * performed.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  unsigned int  n_cycles;

  /*
   * @brief A control parameter for AMG preconditioning using the Trilinos
   * library to control how the coarsening should be performed.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  double  aggregation_threshold;

};


/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionAMGParameters &prm);


/*!
 * @struct LinearSolverParameters
 *
 * @brief A structure containing all parameters relevant for the solution of
 * linear systems using a Krylov subspace method.
 *
 * @todo Proper initiation of the solver_name string without constructor
 * ambiguity. The string is included in order for the stream to print
 * also the names of the solvers to the terminal. As of now one can not
 * tell one from the other (On the prm file they are properly differentiated)
 */
struct LinearSolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  LinearSolverParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the linear solver from
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

  /*!
   * @brief Pointer to the parameter of the preconditioners
   */
  std::shared_ptr<PreconditionBaseParameters> preconditioner_parameters_ptr;

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
 * @brief @ref DimensionlessNumbers contains all the dimensionless
 * numbers relevant to the different problem types.
 */
struct DimensionlessNumbers
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  DimensionlessNumbers();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the dimensionless numbers from
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
   * @details Defined as
   * \f[
   * \Reynolds = \frac{u D}{\nu},
   * \f]
   * where \f$ u \f$ is the characteristic velocity, \f$ D \f$ the
   * characteristic length and \f$ \nu \f$ the kinematic viscosity.
   */
  double  Re;

  /*!
   * @brief The Prandtl number
   * @details Defined as
   * \f[
   * \Prandtl = \frac{\nu}{\kappa},
   * \f]
   * where \f$ \nu \f$ is the kinematic viscosity and \f$ \kappa \f$ the
   * thermal diffusivity.
   */
  double  Pr;

  /*!
   * @brief The Peclet number
   * @details Defined as
   * \f[
   * \Peclet = \frac{u D}{\kappa},
   * \f]
   * where \f$ u \f$ is the characteristic velocity, \f$ D \f$ the
   * characteristic length and \f$ \kappa \f$ the thermal diffusivity.
   */
  double  Pe;

  /*!
   * @brief The Rayleigh number
   * @details Defined as
   * \f[
   * \Rayleigh = \frac{\alpha g D^3}{\nu \kappa},
   * \f]
   * where \f$ \alpha \f$ is the thermal expansion coefficients,
   * \f$ g \f$ the reference gravity magnitude,
   * \f$ D \f$ the characteristic length,
   * \f$ \nu \f$ the kinematic viscosity and
   * \f$ \kappa \f$ the thermal diffusivity.
   */
  double  Ra;

  /*!
   * @brief The Ekman number
   * @details Defined as
   * \f[
   * \Ekman = \frac{\nu}{\Omega D^2},
   * \f]
   * where \f$ nu \f$ is the kinematic viscosity, \f$ \Omega \f$ the
   * characteristic angular velocity and \f$ D \f$ the characteristic
   * length.
   */
  double  Ek;

  /*!
   * @brief The magnetic Prandtl number
   * @details Defined as
   * \f[
   * \magPrandtl = \frac{\nu}{\eta},
   * \f]
   * where \f$ nu \f$ is the kinematic viscosity and
   * \f$ \eta \f$ the magnetic diffusivity.
   */
  double  Pm;

private:
  /*!
   * @brief The problem type
   * @details Its inclusion in @ref DimensionlessNumbers is for the
   * sole purpose to control the stream output, effectively printing only
   * the relevant numbers.
   */
  ProblemType problem_type;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm);



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
   * @attention This needs further work as the weak forms are note
   * one to one in the Navier Stokes and heat equations
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
   * @attention I do not like the flag. It was my quick fix in order to
   * print either the refinement parameters or the convergence test
   * parameters. As when one is needed the other is not.
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
   * @brief Problem type.
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
  SpatialDiscretizationParameters             spatial_discretization_parameters;

  /*!
   * @brief Parameters of the time stepping scheme.
   */
  TimeDiscretization::TimeDiscretizationParameters  time_discretization_parameters;

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
