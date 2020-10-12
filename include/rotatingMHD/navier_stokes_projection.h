#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

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
 * @brief Solves the Navier Stokes equations with the incremental pressure
 * projection scheme.
 *
 * @details This version is parallelized using deal.ii's MPI facilities and
 * relies either on the Trilinos or the PETSc library. Moreover, for the time
 * discretization an implicit-explicit scheme (IMEX) with variable step size is
 * used. 
 * 
 * The class is coded to solve a diffusion step given by
 * \f[
 * \begin{equation}
 * \dfrac{\alpha_0^n}{\Delta t_{n-1}} \bs{u}^{n} -\dfrac{1}{\textrm{Re}} 
 * \Delta \bs{u}^{n} + \bs{u}^{\star, n} \cdot (\nabla \otimes \bs{u}^{n})
 * + \frac{1}{2} (\nabla \cdot \bs{u}^{\star, n}) \bs{u}^{n} = -\nabla p^{\sharp, n}
 * - \sum_{j=1}^{2} \dfrac{\alpha_j^n}{\Delta t_{n-1}}\bs{u}^{n-j}, 
 * \qquad \forall(\bs{x},t)\in \Omega \times [0, T]
 * \end{equation}
 * \f]
 * with
 * \f[ 
 * \begin{equation}
 *  p^\sharp = p^{n-1} + 
 *  \sum_{j=1}^{2} \frac{\Delta t_{n-1-j}}{\Delta t_{n-1}}
 *  \frac{\alpha_j^n}{\alpha_0^{n-j}}\phi^{n-j}
 * \end{equation}
 * \f]
 * a projection step given by
 * \f[ 
 * \begin{equation}
 * \Delta \phi^{n} = \dfrac{\alpha_0^n}{\Delta t_{n-1}} \nabla \cdot 
 * \bs{u}^{n},  \qquad \forall (\bs{x}, t) 
 * \in \Omega \times \left[0, T \right]
 * \end{equation}
 * \f]
 * and a pressure correction step given by
 * \f[
 * \begin{equation*}
 * p^{n} = p^{n-1} + \phi^{n} - \frac{\chi}{\textrm{Re}} \nabla \cdot \bs{u}^{n}
 * \end{equation*}
 * \f]
 * where \f$ \chi \f$ is either 0 or 1 denoting the standard or rotational
 * incremental scheme respectively.
 * @todo Implement the complete generalized VSIMEX method.
 * @todo Implement a generalized extrapolation scheme.
 * @todo Implement the different forms of the convective term.
 * @attention The code is hardcoded for a second order time discretization
 * scheme. Setting a first order scheme in the parameter file will cause
 * errors.
 */
template <int dim>
class NavierStokesProjection
{

public:
  NavierStokesProjection
  (const RunTimeParameters::ParameterSet   &parameters,
   Entities::VectorEntity<dim>             &velocity,
   Entities::ScalarEntity<dim>             &pressure,
   TimeDiscretization::VSIMEXMethod        &time_stepping,
   const std::shared_ptr<ConditionalOStream>external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>       external_timer =
       std::shared_ptr<TimerOutput>());

  double                            norm_diffusion_rhs;

  double                            norm_projection_rhs;


  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the projection method problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the VectorEntity and ScalarEntity structs passed on
   *  in the constructor (The velocity and the pressure respectively).
   *  The boolean passed as argument control if the pressure is to be
   *  normalized.
   */
  void setup(const bool normalize_pressure = false);

  /*!
   *  @brief Sets the body force of the problem.
   *
   *  @details Stores the memory address of the body force function in 
   *  the pointer @ref body_force.
   */
  void set_body_force(RMHD::EquationData::BodyForce<dim> &body_force);

  /*!
   * @brief Currently this method only sets the vector of the two pressure
   * updates @ref old_phi and @ref old_old_phi to zero.
   */
  void initialize();


  /*!
   *  @brief Solves the problem for one single timestep.
   *
   *  @details Performs the diffusion and the projection step for one single
   *  time step and updates the member variables at the end.
   */
  void solve(const unsigned int step);

  /*!
   *  @brief Prepares the internal entities for the next time step.
   *
   *  @details The internal vectors of the variable \f$ \phi \f$ are
   *  updated by taking the values from the vector of the next time step
   *  relative to them, i.e.
   *  \f{eqnarray*}{
   *  \phi^{n-2} &=& \phi^{n-1}, \\
   *  \phi^{n-1} &=& \phi^{n}.
   *  \f}
   */
  void update_internal_entities();

  /*!
   *  @brief Restarts the internal entities.
   *
   *  @details The internal vectors of the variable \f$ \phi \f$ are
   *  set to zero and @ref flag_diffusion_matrix_assembled is set
   *  to false.
   */
  void reinit_internal_entities();

  /*!
   *  @brief Computes the next time step according to the Courant-Friedrichs-Lewy
   *  condition.
   *
   *  @details The next time step is given by 
   * \f[
   *    \Delta t_{n-1} = C \min_{K \in \Omega_\textrm{h}}
   *    \left\lbrace \frac{\max_{P \in K} { \left\lVert \bs{v} \right\rVert}}{h_K} \right\rbrace
   * \f] 
   * where \f$ C \f$ is the Courant number, \f$ K\f$ denotes the 
   * \f$ K\f$-th cell of the tessallation, \f$ \Omega_\textrm{h}\f$ the tessallation,
   * \f$ P \f$ a quadrature point inside the \f$ K\f$-th cell,
   * \f$ \bs{v} \f$ the velocity, \f$ h_K\f$ the largest diagonal of the \f$ K\f$-th
   * cell.
   *  @attention The Courant number is hardcoded to 1.
   */
  double compute_next_time_step();

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
   * @brief Pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>     pcout;

  /*!
   * @breif Pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>            computing_timer;

  /*!
   * @brief A reference to the entity of velocity field.
   */
  Entities::VectorEntity<dim>            &velocity;

  /*!
   * @brief A reference to the entity of the pressure field.
   */
  Entities::ScalarEntity<dim>            &pressure;

  /*!
   * @brief A pointer to the body force function.
   */
  RMHD::EquationData::BodyForce<dim>    *body_force_ptr;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod &time_stepping;

  /*!
   * @brief System matrix used to solve for the velocity field in the diffusion
   * step.
   *
   * @details This matrix changes in every timestep because the convective term
   * needs to be assembled in every timestep. However, not the entire system
   * matrix is assembled in every timestep but only the part due to the
   * convective term.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_system_matrix;

  /*!
   * @brief Mass matrix of the velocity.
   *
   * @details This matrix does change not in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_mass_matrix;

  /*!
   * @brief Stiffness matrix of the velocity. Assembly of  the weak of the
   * Laplace operator.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_laplace_matrix;

  /*!
   * @brief Sum of the mass and the stiffness matrix of the velocity.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_mass_plus_laplace_matrix;

  /*!
   * @brief Matrix representing the assembly of the skew-symmetric formm of the
   * convective term.
   *
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_advection_matrix;

  /*!
   * @brief A vector representing the extrapolated velocity at the
   * current timestep using a Taylor expansion
   * @details The Taylor expansion is given by
   * \f{eqnarray*}{
   * u^{n} &\approx& u^{n-1} + \frac{\partial u^{n-1}}{\partial t} \Delta t \\
   *         &\approx& u^{n-1} + \frac{u^{n-1} - u^{n-2}}{\Delta t} \Delta t \\
   *         &\approx& 2 u^{n-1} - u^{n-2}.
   * \f}
   * In the case of a variable time step the approximation is given by
   * \f[
   * u^{n} \approx (1 + \omega) u^{n-1} - \omega u^{n-2}
   * \f] 
   * where  \f$ \omega = \frac{\Delta t_{n-1}}{\Delta t_{n-2}}.\f$
   * @attention The extrapolation is hardcoded to the second order described
   * above. First and higher order are pending.
   */
  LinearAlgebra::MPI::Vector        extrapolated_velocity;

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
   * which we use when assembling the right hand side of the diffusion
   * step.
   */
  LinearAlgebra::MPI::Vector        velocity_tmp;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * diffusion step.
   */
  LinearAlgebra::MPI::Vector        velocity_rhs;

  /*!
   * @brief Mass matrix of the pressure field.
   */
  LinearAlgebra::MPI::SparseMatrix  pressure_mass_matrix;

  /*!
   * @brief Stiffness matrix of the pressure field. Assembly of  the weak of the
   * Laplace operator.
   */
  LinearAlgebra::MPI::SparseMatrix  pressure_laplace_matrix;

  /*!
   * @brief Vector representing the pressure used in the diffusion step.
   * @details The pressure is given by
   * \f[
   * p_\textrm{tmp} = p^{n-1} + 
   *                  \sum_{j=1}^{2} \frac{\Delta t_{n-1-j}}{\Delta t_{n-1}}
   *                  \frac{\alpha_j^n}{\alpha_0^{n-j}}\phi^\textrm{n-j} 
   * \f] 
   */  
  LinearAlgebra::MPI::Vector        pressure_tmp;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * projection step.
   */
  LinearAlgebra::MPI::Vector        pressure_rhs;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * poisson prestep.
   */
  LinearAlgebra::MPI::Vector        poisson_prestep_rhs;

  /*!
   * @brief Vector representing the pressure update of the current timestep.
   */
  LinearAlgebra::MPI::Vector        phi;

  /*!
   * @brief Vector representing the pressure update of the previous timestep.
   */
  LinearAlgebra::MPI::Vector        old_phi;

  /*!
   * @brief Vector representing the pressure update of two timesteps prior.
   */
  LinearAlgebra::MPI::Vector        old_old_phi;

  /*!
   * @brief The preconditioner of the diffusion step.
   * @attention Hardcoded for a ILU preconditioner.
   */
  LinearAlgebra::MPI::PreconditionILU     diffusion_step_preconditioner;
  
  /*!
   * @brief The preconditioner of the projection step.
   * @attention Hardcoded for a ILU preconditioner.
   */
  LinearAlgebra::MPI::PreconditionILU     projection_step_preconditioner;
  
  /*!
   * @brief The preconditioner of the correction step.
   * @attention Hardcoded for a Jacobi preconditioner.
   */
  LinearAlgebra::MPI::PreconditionJacobi  correction_step_preconditioner;

  // SG thinks that all of these parameters can go into a parameter structure.
  const double                          absolute_tolerance = 1.0e-9;
  
  /*!
   * @brief A flag for the assembly of the diffusion step.
   * @details In the case of a constant time step, this flags avoids
   * assembling the system matrix in each time step.
   */
  bool                                  flag_diffusion_matrix_assembled;

  /*!
   * @brief A flag for the initializing of the solver.
   * @details It is only set as true while initializing in order to 
   * reuse methods of the normal solve procedure.
   */
  bool                                  flag_initializing;
  
  /*!
   * @brief A flag to normalize the pressure field.
   * @details In the case of an unconstrained formulation in the 
   * pressure space, i.e. no Dirichlet boundary conditions, this flag
   * has to be set to true in order to constraint the pressure field.
   */
  bool                                  flag_normalize_pressure;

  /*!
   * @brief Setup of the sparsity spatterns of the matrices of the diffusion and
   * projection steps.
   */
  void setup_matrices();

  /*!
   * @brief Setup of the right-hand side and the auxiliary vector of the
   * diffusion and projection step.
   */
  void setup_vectors();

  /*!
   * @brief Assemble the matrices which change only if the triangulation is
   * refined or coarsened.
   */
  void assemble_constant_matrices();

  /*!
   * @brief This method performs the poisson prestep.
   */
  void poisson_prestep();

  /*!
   * @brief This method assembles the linear system of the poisson
   * prestep.
   */
  void assemble_poisson_prestep();

  /*!
   * @brief This method assembles the right-hand side of the poisson
   * prestep.
   */
  void assemble_poisson_prestep_rhs();

  /*!
   * @brief This method solves the linear system of the poisson prestep.
   */
  void solve_poisson_prestep();

  /*!
   * @brief This method performs the diffusion prestep.
   */
  void diffusion_prestep();

  /*!
   * @brief This method assembles the system matrix of the diffusion
   * prestep and calls assemble_diffusion_prestep_rhs
   * @details The system matrix \f$\bs{A}^{(\bs{v})}\f$ is constructed from the mass
   * \f$\bs{M}^{(\bs{v})}\f$, the stiffness \f$\bs{K}^{(\bs{v})}\f$ and
   * the advection matrices \f$\bs{C}^{(\bs{v})}\f$ as follows
   *
   * \f[
   * \bs{A} = \frac{1}{\Delta t_{n-1}} \bs{M}^{(\bs{v})}+ \frac{1}{\Reynolds}
   * \bs{K}^{(\bs{v})} + \bs{C}^{(\bs{v})} \,.
   * \f]
   */
  void assemble_diffusion_prestep();

  /*!
   * @brief This method performs the projection prestep.
   */
  void projection_prestep();

  /*!
   * @brief This method performs the pressure correction prestep.
   */
  void pressure_correction_prestep();

  /*!
   * @brief This method performs one complete diffusion step.
   */
  void diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the system matrix and the right-hand side of
   * the diffusion step.
   *
   * The system matrix \f$\bs{A}^{(\bs{v})}\f$ is constructed from the mass
   * \f$\bs{M}^{(\bs{v})}\f$, the stiffness \f$\bs{K}^{(\bs{v})}\f$ and
   * the advection matrices \f$\bs{C}^{(\bs{v})}\f$ as follows
   *
   * \f[
   * \bs{A} = \frac{\alpha_0}{\Delta t_{n-1}} \bs{M}^{(\bs{v})}+ \frac{1}{\Reynolds}
   * \bs{K}^{(\bs{v})} + \bs{C}^{(\bs{v})} \,.
   * \f]
   */
  void assemble_diffusion_step();

  /*!
   * @brief This method solves the linear system of the diffusion step.
   */
  void solve_diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method performs one complete projection step.
   */
  void projection_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the linear system of the projection step.
   */
  void assemble_projection_step();

  /*!
   * @brief This method solves the linear system of the projection step.
   */
  void solve_projection_step(const bool reinit_prec);

  /*!
   * @brief This method performs the pressure update of the projection step.
   */
  void pressure_correction(const bool reinit_prec);

  /*!
   * @brief This method assembles the mass \f$\bs{M}^{(\bs{v})}\f$ and the
   * stiffness matrix \f$\bs{K}^{(\bs{v})}\f$ of the velocity field using
   * the WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\bs{M}^{(\bs{v})}\f$ and \f$\bs{K}^{(\bs{v})}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \bs{M}^{(\bs{v})}_{ij} &= \int\limits_\Omega \bs{\varphi}_i\cdot\bs{\varphi}_j\dint{V}\,,\\
   * \bs{K}^{(\bs{v})}_{ij} &= \int\limits_\Omega (\nabla\otimes\bs{\varphi}_i)\cdott
   * (\nabla\otimes\bs{\varphi}_j)\dint{V}\,.
   * \end{aligned}
   * \end{equation*}
   * \f]
   */
  void assemble_velocity_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_velocity_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityMatricesAssembly::LocalCellData<dim>          &scratch,
    VelocityMatricesAssembly::MappingData<dim>            &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the velocity field on a single cell into the global matrices.
   */
  void copy_local_to_global_velocity_matrices(
    const VelocityMatricesAssembly::MappingData<dim>      &data);

  /*!
   * @brief This method assembles the mass \f$\bs{M}^{(p)}\f$ and the
   * stiffness matrices \f$\bs{K}^{(p)}\f$ of the pressure field using the
   * WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\bs{M}^{(p)}\f$ and \f$\bs{K}^{(p)}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \bs{M}^{(p)}_{ij} &= \int\limits_\Omega \varphi_i\varphi_j\dint{V}\,,\\
   * \bs{K}^{(p)}_{ij} &= \int\limits_\Omega \nabla\varphi_i\cdot\nabla\varphi_j\dint{V}\,.
   * \end{aligned}
   * \end{equation*}
   * \f]
   */
  void assemble_pressure_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_pressure_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureMatricesAssembly::LocalCellData<dim>          &scratch,
    PressureMatricesAssembly::MappingData<dim>            &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the pressure field on a single cell into the global matrices.
   */
  void copy_local_to_global_pressure_matrices(
    const PressureMatricesAssembly::MappingData<dim>      &data);

  /*!
   * @brief This method assembles the right-hand side of the poisson
   * prestep using the WorkStream approach.
   */
  void assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<dim>::active_cell_iterator        &cell,
    PoissonPrestepRightHandSideAssembly::LocalCellData<dim>     &scratch,
    PoissonPrestepRightHandSideAssembly::MappingData<dim>       &data);

  /*!
   * @brief This method assembles the local right-hand side of the poisson
   * prestep on a single cell.
   */
  void copy_local_to_global_poisson_prestep_rhs(
    const PoissonPrestepRightHandSideAssembly::MappingData<dim> &data);

  /*!
   * @brief This method assembles the right-hand side of the diffusion step
   * using the WorkStream approach.
   *
   * This method assembles the following weak form into the vector representing
   * the right-hand side \f$\bs{b}\f$
   * \f[
   * \bs{b}_i = -\int\limits_\Omega (\frac{\alpha_1}{\Delta t_{n-1}} 
   * \bs{v}^{n-1} + \frac{\alpha_2}{\Delta t_{n-1}} \bs{v}^{n-2}) \cdot
   * \bs{\varphi}_i \dint{V} + \int\limits_\Omega p^\star
   * (\nabla\cdot\bs{\varphi}_i) \dint{V}\,,
   * \f]
   * where \f$\bs{\varphi}_i\f$ is a test function of the velocity space.
   * Furthermore, \f$p^\star\f$ denotes the extrapolated pressure.
   */
  void assemble_diffusion_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the diffusion
   * step on a single cell.
   */
  void assemble_local_diffusion_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
    VelocityRightHandSideAssembly::MappingData<dim>       &data);

  /*!
   * @brief This method copies the local right-hand side of the diffusion step
   * into the global vector.
   */
  void copy_local_to_global_diffusion_step_rhs(
    const VelocityRightHandSideAssembly::MappingData<dim> &data);

  /*!
   * @brief This method assembles the right-hand side of the projection step
   * using the WorkStream approach.
   *
   * This method assembles the following weak form into the vector representing
   * the right-hand side \f$\bs{b}\f$
   * \f[
   * \bs{b}_i = -\int\limits_\Omega (\nabla\cdot\bs{v}) \varphi_i \dint{V}\,,
   * \f]
   * where \f$\varphi_i\f$ is a test function of the pressure space.
   */
  void assemble_projection_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the projection
   * step on a single cell.
   */
  void assemble_local_projection_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureRightHandSideAssembly::LocalCellData<dim>     &scratch,
    PressureRightHandSideAssembly::MappingData<dim>       &data);

  /*!
   * @brief This method copies the local right-hand side of the projection step
   * on a single cell to the global vector.
   */
  void copy_local_to_global_projection_step_rhs(
    const PressureRightHandSideAssembly::MappingData<dim> &data);

  /*!
   * @brief This method assembles the velocity advection matrix using the
   * WorkStream approach.
   *
   * This method assembles the following skew-symmetric weak form into the
   * advection matrix \f$\bs{C}^{(\bs{v})}\f$
   * \f[
   * \bs{C}^{(\bs{v})}_{ij} = \int\limits_\Omega
   * \bs{v}^\star\cdot(\nabla\otimes\bs{\varphi}_j)\cdot \bs{\varphi}_i \dint{V} +
   * \int\limits_\Omega \tfrac{1}{2}(\nabla\cdot\bs{v}^\star)
   * \bs{\varphi}_j\cdot\bs{\varphi}_i \dint{V}\,,
   * \f]
   * where \f$\varphi_j\f$ and \f$\varphi_i\f$ are the trial and test functions
   * of the velocity space. Furthermore, \f$\bs{v}^\star\f$ denotes
   * the extrapolated velocity.
   */
  void assemble_velocity_advection_matrix();

  /*!
   * @brief This method assembles the local velocity advection matrix on a
   * single cell.
   */
  void assemble_local_velocity_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionAssembly::LocalCellData<dim>                 &scratch,
    AdvectionAssembly::MappingData<dim>                   &data);

  /*!
   * @brief This method copies the local velocity advection matrix into the
   * global matrix.
   */
  void copy_local_to_global_velocity_advection_matrix(
    const AdvectionAssembly::MappingData<dim>             &data);
  
};

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_ */
