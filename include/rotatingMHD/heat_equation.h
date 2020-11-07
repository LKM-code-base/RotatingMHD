#ifndef INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_
#define INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>

#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <memory>
#include <string>
#include <vector>

namespace RMHD
{

using namespace dealii;

/*!
 * @class HeatEquation
 * 
 * @brief Solves the heat equation.
 * 
 * @details This version is parallelized using deal.ii's MPI facilities and
 * relies either on the Trilinos or the PETSc library. Moreover, for the time
 * discretization an implicit-explicit scheme (IMEX) with variable step size is
 * used.
 * The heat equation solved is derived from ... pending...
 * @todo Documentation
 */

template <int dim>
class HeatEquation
{

public:
  /*!
   * @brief The constructor of the HeatEquation class where the velocity
   * a VectorEntitty instance is.
   * 
   * @details Stores local references to the input parameters and 
   * pointers for terminal output entities.
   */
  HeatEquation
  (const RunTimeParameters::ParameterSet   &parameters,
   TimeDiscretization::VSIMEXMethod        &time_stepping,
   Entities::ScalarEntity<dim>             &temperature,
   Entities::VectorEntity<dim>             &velocity,
   const std::shared_ptr<Mapping<dim>>     external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>       external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the heat equation problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the VectorEntity and ScalarEntity structs passed on
   *  in the constructor (The velocity and the temperature respectively).
   */
  void setup();

  /*!
   *  @brief Sets the supply term of the problem.
   *
   *  @details Stores the memory address of the supply term function in 
   *  the pointer @ref suppler_term_ptr.
   */
  void set_supply_term(Function<dim> &supply_term);

  /*!
   * @brief Initializes the old_solution of the temperature struct for
   * a time discretization scheme of second order.
   */
  void initialize();

  /*!
   *  @brief Solves the heat equation problem for one single timestep.
   */
  void solve();

  /*!
   *  @brief Returns the norm of the right hand side for the last solved
   * step.
   */
  double get_rhs_norm() const;
  
  /*!
   *  @brief Indicates the solver to reinitialize the matrices and
   *  vectors before solving. 
   */ 
  void set_linear_algebra_to_reset();

private:
  /*!
   * @brief A reference to the parameters which control the solution process.
   */
  const RunTimeParameters::ParameterSet  &parameters;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                         &mpi_communicator;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod &time_stepping;

  /*!
   * @brief Pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>     pcout;

  /*!
   * @brief Pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>            computing_timer;

  /*!
   * @brief Pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>           mapping;

  /*!
   * @brief A reference to the entity of the temperature field.
   */
  Entities::ScalarEntity<dim>            &temperature;

  /*!
   * @brief A pointer to the entity of velocity field.
   */
  Entities::VectorEntity<dim>            *velocity;

  /*!
   * @brief A pointer to the supply term function.
   */
  Function<dim>                          *supply_term_ptr;

  /*!
   * @brief System matrix for the heat equation.
   * @details The matrix changes in every timestep due to the mass
   * and stiffness matrices being weighted by the VSIMEX coefficients.
   */
  LinearAlgebra::MPI::SparseMatrix      system_matrix;

  /*!
   * @brief Mass matrix of the temperature.
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix      mass_matrix;

  /*!
   * @brief Stiffness matrix of the temperature.
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix      stiffness_matrix;

  /*!
   * @brief Sum of the mass and stiffness matrix of the temperature.
   * @details If the time step size is constant, this matrix does not 
   * change each step.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix      mass_plus_stiffness_matrix;

  /*!
   * @brief Advection matrix of the temperature.
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix      advection_matrix;


  /*!
   * @brief Vector representing the right-hand side of the linear system.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::Vector            rhs;

  /*!
   * @brief The \f$ L_2 \f$ norm of the right hand side
   */
  double                                rhs_norm;

  /*!
   * @brief Vector representing the sum of the time discretization terms
   * that belong to the right hand side of the equation.
   * @details For example: A BDF2 scheme with a constant time step
   * expands the time derivative in three terms
   * \f[
   * \frac{\partial u}{\partial t} \approx 
   * \frac{1.5}{\Delta t} u^{n} - \frac{2}{\Delta t} u^{n-1}
   * + \frac{0.5}{\Delta t} u^{n-2},
   * \f] 
   * the last two terms are known quantities so they belong to the 
   * right hand side of the equation. Therefore, we define
   * \f[
   * u_\textrm{tmp} = - \frac{2}{\Delta t} u^{n-1}
   * + \frac{0.5}{\Delta t} u^{n-2},
   * \f].
   * which we use when assembling the right-hand side of the linear
   * system
   */
  LinearAlgebra::MPI::Vector            temperature_tmp;

  /*!
   * @brief Preconditioner of the linear system.
   */
  LinearAlgebra::MPI::PreconditionILU   preconditioner;

  /*!
   * @brief Absolute tolerance of the Krylov solver.
   */
  const double                          absolute_tolerance = 1.0e-9;

  /*!
   * @brief A flag to normalize the temperature field.
   * @details In the case of an unconstrained formulation in the 
   * temperature space, i.e. no Dirichlet boundary conditions, this flag
   * has to be set to true in order to constraint the temperature field.
   */
  bool                                  flag_zero_mean_value;

  /*!
   * @brief A flag indicating if the solver is to be set up, i.e.
   * the matrices and vectors to be initialize.
   */ 
  bool                                  flag_setup_solver;

  /*!
   * @brief A flag indicating if the preconditioner is to be
   * initiated.
   */ 
  bool                                  flag_reinit_preconditioner;

  /*!
   * @brief A flag indicating if the sum of the mass and stiffness matrix
   * is to be performed.
   */ 
  bool                                  flag_assemble_mass_plus_stiffness_matrix;

  /*!
   * @brief A flag indicating if the advection term is to be ignored.
   */ 
  bool                                  flag_ignore_advection;

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
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureConstantMatricesAssembly::LocalCellData<dim> &scratch,
    TemperatureConstantMatricesAssembly::MappingData<dim>   &data);

  /*!
   * @brief This method copies the mass matrix into its global
   * conterpart.
   */
  void copy_local_to_global_constant_matrices(
    const TemperatureConstantMatricesAssembly::MappingData<dim>  &data);

  /*!
   * @brief This method assembles the advection matrix on a single cell.
   */
  void assemble_local_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureAdvectionMatrixAssembly::LocalCellData<dim>  &scratch,
    TemperatureAdvectionMatrixAssembly::MappingData<dim>    &data);

  /*!
   * @brief This method copies the local advection matrix into their 
   * global conterparts.
   */
  void copy_local_to_global_advection_matrix(
    const TemperatureAdvectionMatrixAssembly::MappingData<dim>  &data);


  /*!
   * @brief This method assembles the right-hand side on a single cell.
   */
  void assemble_local_rhs(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureRightHandSideAssembly::LocalCellData<dim> &scratch,
    TemperatureRightHandSideAssembly::MappingData<dim>   &data);

  /*!
   * @brief This method copies the local right-hand side into its global
   * conterpart.
   */
  void copy_local_to_global_rhs(
    const TemperatureRightHandSideAssembly::MappingData<dim>  &data);
};

template <int dim>
inline double HeatEquation<dim>::get_rhs_norm() const
{
  return (rhs_norm);
}

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_ */
