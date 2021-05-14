#ifndef INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_SOLVER_H_
#define INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_SOLVER_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>

#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/basic_parameters.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/linear_solver_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <memory>
#include <string>
#include <vector>

namespace RMHD
{

using namespace dealii;


/*!
 * @struct ConvectionDiffusionParameters
 *
 * @brief A structure containing all the parameters of the heat
 * equation solver.
 */
struct ConvectionDiffusionParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  ConvectionDiffusionParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ConvectionDiffusionParameters(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const ConvectionDiffusionParameters &prm);

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is to be implemented
   * @attention This needs further work as the weak forms are note
   * one to one in the Navier Stokes and heat equations
   */
  RunTimeParameters::ConvectiveTermWeakForm            convective_term_weak_form;

  /*!
   * @brief Enumerator controlling which time discretization of the
   * convective term is to be implemented
   */
  RunTimeParameters::ConvectiveTermTimeDiscretization  convective_term_time_discretization;

    /*!
   * @brief The factor multiplying the diffusion term.
   */
  double  equation_coefficient;

  /*!
   * @brief The parameters for the linear solver.
   */
  RunTimeParameters::LinearSolverParameters solver_parameters;

  /*!
   * @brief Specifies the frequency of the update of the solver's
   * preconditioner.
   */
  unsigned int  preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool  verbose;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvectionDiffusionParameters &prm);

/*!
 * @class ConvectionDiffusionSolver
 *
 * @brief Solves a convection-diffusion problem.
 *
 * @details This version is parallelized using deal.ii's MPI facilities and
 * relies either on the Trilinos or the PETSc library. Moreover, for the time
 * discretization an implicit-explicit scheme (IMEX) with variable step size is
 * used.
 * The heat equation solved is derived from the balance of internal
 * energy
 * \f[
 *  \pd{u}{\d t} + \bs{u}\cdot\nabla u = - C \nabla^2 u + s\,,
 *  \quad \forall (\bs{x}, t) \in \Omega \times \left[0, T \right]
 * \f]
 * where \f$ u \f$ is a scalar field, for example, the temperature or a
 * compositional variable, \f$ C \f$ is dimensionless number,
 * \f$ s \f$ a source term. Moreover, \f$ \bs{v} \f$ denotes the velocity.
 *
 * @todo Documentation
 *
 * @todo Add consistent forms of the advection term.
 */
template <int dim>
class ConvectionDiffusionSolver
{

public:
  /*!
   * @brief The constructor of the HeatEquation class for the case
   * when there is no advection.
   *
   * @details Stores a local reference to the input parameters and
   * pointers for the mapping and terminal output entities.
   */
  ConvectionDiffusionSolver
  (const ConvectionDiffusionParameters              &parameters,
   const TimeDiscretization::VSIMEXMethod           &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>>     &phi,
   const std::shared_ptr<Mapping<dim>>              external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>        external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>               external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The constructor of the HeatEquation class for the case
   * when the velocity field is given by a VectorEntity instance.
   *
   * @details Stores a local reference to the input parameters and
   * pointers for the mapping and terminal output entities.
   */
  ConvectionDiffusionSolver
  (const ConvectionDiffusionParameters              &parameters,
   const TimeDiscretization::VSIMEXMethod           &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>>     &phi,
   std::shared_ptr<Entities::VectorEntity<dim>>     &velocity,
   const std::shared_ptr<Mapping<dim>>              external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>        external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>               external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The constructor of the HeatEquation class for the case
   *  where the velocity is given by a TensorFunction.
   *
   * @details Stores a local reference to the input parameters and
   * pointers for the mapping and terminal output entities.
   */
  ConvectionDiffusionSolver
  (const ConvectionDiffusionParameters              &parameters,
   const TimeDiscretization::VSIMEXMethod           &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>>     &phi,
   std::shared_ptr<TensorFunction<1, dim>>          &velocity,
   const std::shared_ptr<Mapping<dim>>              external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>        external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>               external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the heat equation problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the ScalarEntity and VectorEntity structs passed on
   *  in the constructor (The temperature and the velocity respectively).
   */
  void setup();

  /*!
   *  @brief Sets the source term of the problem.
   *
   *  @details Stores the memory address of the source term function in
   *  the pointer @ref suppler_term_ptr.
   */
  void set_source_term(Function<dim> &source_term);

  /*!
   *  @brief Sets the source term of the problem.
   *
   *  @details Stores the memory address of the source term function in
   *  the pointer @ref suppler_term_ptr.
   */
  void set_velocity(const std::shared_ptr<TensorFunction<1,dim>> &velocity);

  /*!
   *  @brief Sets the velocity field field of the problem.
   *
   *  @details Stores the memory address of the velocity in
   *  the pointer @ref velocity_function_ptr.
   */
  void set_velocity(const std::shared_ptr<Entities::VectorEntity<dim>> &velocity);

  /*!
   *  @brief Solves the heat equation problem for one single timestep.
   */
  void solve();

  /*!
   * @details Release all memory and return all objects to a state just like
   * after having called the default constructor.
   */
  void clear();

private:
  /*!
   * @brief A reference to the parameters which control the solution process.
   */
  const ConvectionDiffusionParameters           &parameters;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                                &mpi_communicator;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod        &time_stepping;

  /*!
   * @brief A shared pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>           pcout;

  /*!
   * @brief A shared pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>                  computing_timer;

  /*!
   * @brief A shared pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>                 mapping;

  /*!
   * @brief A shared pointer to the entity of the scalar field.
   */
  std::shared_ptr<Entities::ScalarEntity<dim>>        phi;

  /*!
   * @brief A shared pointer to the entity of velocity field.
   */
  std::shared_ptr<const Entities::VectorEntity<dim>>  velocity;

  /*!
   * @brief A shared pointer to the TensorFunction of the velocity field.
   */
  std::shared_ptr<TensorFunction<1,dim>>        velocity_function_ptr;

  /*!
   * @brief A pointer to the supply term function.
   */
  Function<dim>                                 *source_term_ptr;

  /*!
   * @brief System matrix for the heat equation.
   * @details For
   */
  LinearAlgebra::MPI::SparseMatrix              system_matrix;

  /*!
   * @brief Mass matrix
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   *
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              mass_matrix;

  /*!
   * @brief Stiffness matrix.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              stiffness_matrix;

  /*!
   * @brief Sum of the mass and stiffness matrix.
   *
   * @details If the time step size is constant, this matrix does not
   * change each step.
   *
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              mass_plus_stiffness_matrix;

  /*!
   * @brief Advection matrix.
   *
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   *
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              advection_matrix;

  /*!
   * @brief Vector representing the right-hand side of the linear system.
   *
   * @todo Add formulas
   */
  LinearAlgebra::MPI::Vector                    rhs;

  /*!
   * @brief The preconditioner of the linear system.
   */
  std::shared_ptr<LinearAlgebra::PreconditionBase> preconditioner;

  /*!
   * @brief A flag indicating if the matrices were updated.
   */
  bool  flag_matrices_were_updated;

  /*!
   * @brief Setup of the sparsity spatterns of the matrices.
   */
  void setup_matrices();

  /*!
   * @brief Setup of the right-hand side and the auxiliary vectors.
   */
  void setup_vectors();

  /*!
   * @brief Assemble the matrices which change only if the triangulation is
   * refined or coarsened.
   * @todo Add formulas
   */
  void assemble_constant_matrices();

  /*!
   * @brief Assemble the advection matrix.
   * @todo Add formulas
   */
  void assemble_advection_matrix();


  /*!
   * @brief Assembles the right-hand side.
   * @todo Add formulas
   */
  void assemble_rhs();

  /*!
   * @brief Assembles the linear system.
   * @todo Add formulas
   */
  void assemble_linear_system();

  /*!
   * @brief Solves the linear system.
   * @details Pending.
   */
  void solve_linear_system(const bool reinit_preconditioner);

  /*!
   * @brief This method assembles the mass matrix on a single cell.
   */
  void assemble_local_constant_matrices(
    const typename DoFHandler<dim>::active_cell_iterator        &cell,
    AssemblyData::HeatEquation::ConstantMatrices::Scratch<dim>  &scratch,
    AssemblyData::HeatEquation::ConstantMatrices::Copy          &data);

  /*!
   * @brief This method copies the mass matrix into its global
   * conterpart.
   */
  void copy_local_to_global_constant_matrices(
    const AssemblyData::HeatEquation::ConstantMatrices::Copy    &data);

  /*!
   * @brief This method assembles the advection matrix on a single cell.
   */
  void assemble_local_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator        &cell,
    AssemblyData::HeatEquation::AdvectionMatrix::Scratch<dim>   &scratch,
    AssemblyData::HeatEquation::AdvectionMatrix::Copy           &data);

  /*!
   * @brief This method copies the local advection matrix into their
   * global conterparts.
   */
  void copy_local_to_global_advection_matrix(
    const AssemblyData::HeatEquation::AdvectionMatrix::Copy &data);


  /*!
   * @brief This method assembles the right-hand side on a single cell.
   */
  void assemble_local_rhs(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    AssemblyData::HeatEquation::RightHandSide::Scratch<dim> &scratch,
    AssemblyData::HeatEquation::RightHandSide::Copy         &data);
  /*!
   * @brief This method copies the local right-hand side into its global
   * conterpart.
   */
  void copy_local_to_global_rhs(
    const AssemblyData::HeatEquation::RightHandSide::Copy   &data);
};

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_SOLVER_H_ */
