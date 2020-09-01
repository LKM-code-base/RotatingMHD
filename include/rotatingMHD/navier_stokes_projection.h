#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_

#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

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
 */
template <int dim>
class NavierStokesProjection
{

public:
  NavierStokesProjection(
                const RunTimeParameters::ParameterSet   &parameters,
                Entities::VectorEntity<dim>             &velocity,
                Entities::ScalarEntity<dim>             &pressure,
                TimeDiscretization::VSIMEXCoefficients  &VSIMEX,
                TimeDiscretization::VSIMEXMethod        &time_stepping);

  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the projection method problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the VectorEntity and ScalarEntity structs passed on
   *  in the constructor (The velocity and the pressure respectively).
   */
  void setup();

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
  double compute_next_time_step();

private:

  /*!
   *  @brief A parameter which determines the type of the pressure update.
   *
   *  @details For the pressure update after the projection step, this parameter
   *  determines whether the irrotational form of the pressure update is used or
   *  not.
   */
  RunTimeParameters::ProjectionMethod     projection_method;


  /*!
   * @brief The Reynolds number.
   *
   *  @details A parameter which determines the ratio of convection to viscous
   *  diffusion.
   */
  const double                            Re;

  /*!
   * @brief A reference to the entity of velocity field.
   */
  Entities::VectorEntity<dim>             &velocity;

  /*!
   * @brief A reference to the entity of the pressure field.
   */
  Entities::ScalarEntity<dim>             &pressure;

  TimeDiscretization::VSIMEXCoefficients  &VSIMEX;

  TimeDiscretization::VSIMEXMethod        &time_stepping;

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
   * p_\textrm{tmp} = p^{n-1} + \frac{4}{3} \phi^\textrm{n-1} 
   *            - \frac{1}{3} \phi^{n-2}. 
   * \f] 
   * The formula is taken from the dealii tutorial 
   * <a href="https://www.dealii.org/current/doxygen/deal.II/step_35.html#Projectionmethods">step-35</a> , 
   * from which this class is based upon.
   * @attention In the Guermond paper this is an extrapolated pressure,
   * and it is also called like that in the step-35 documentation, but 
   * I do not see how the formula above is an extrapolation.
   */  
  LinearAlgebra::MPI::Vector        pressure_tmp;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * projection step.
   */
  LinearAlgebra::MPI::Vector        pressure_rhs;

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

  LinearAlgebra::MPI::PreconditionILU     diffusion_step_preconditioner;
  LinearAlgebra::MPI::PreconditionILU     projection_step_preconditioner;
  LinearAlgebra::MPI::PreconditionJacobi  correction_step_preconditioner;

  // SG thinks that all of these parameters can go into a parameter structure.
  unsigned int                          solver_max_iterations;
  unsigned int                          solver_krylov_size;
  unsigned int                          solver_off_diagonals;
  unsigned int                          solver_update_preconditioner;
  double                                relative_tolerance;
  const double                          absolute_tolerance = 1.0e-9;
  double                                solver_diag_strength;
  bool                                  flag_adpative_time_step;


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
   * @brief Currently this method only sets the vector of the two pressure
   * updates @ref phi_n and @ref phi_n_minus_1 to zero.
   */
  void initialize();

  /*!
   * @brief This method performs one complete diffusion step.
   */
  void diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the system matrix and the right-hand side of
   * the diffusion step.
   *
   * The system matrix \f$\mathbf{A}^{(\bs{v})}\f$ is constructed from the mass
   * \f$\mathbf{M}^{(\bs{v})}\f$, the stiffness \f$\mathbf{K}^{(\bs{v})}\f$ and
   * the advection matrices \f$\mathbf{C}^{(\bs{v})}\f$ as follows
   *
   * \f[
   * \mathbf{A} = \alpha_2 \mathbf{M}^{(\bs{v})}+ \frac{1}{\Reynolds}
   * \mathbf{K}^{(\bs{v})} + \mathbf{C}^{(\bs{v})} \,.
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
   * @brief This method assembles the mass \f$\mathbf{M}^{(\bs{v})}\f$ and the
   * stiffness matrix \f$\mathbf{K}^{(\bs{v})}\f$ of the velocity field using
   * the WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\mathbf{M}^{(\bs{v})}\f$ and \f$\mathbf{K}^{(\bs{v})}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \mathbf{M}^{(\bs{v})}_{ij} &= \int\limits_\Omega \bs{\varphi}_i\cdot\bs{\varphi}_j\dint{V}\,,\\
   * \mathbf{K}^{(\bs{v})}_{ij} &= \int\limits_\Omega (\nabla\otimes\bs{\varphi}_i)\cdott
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
   * @brief This method assembles the mass \f$\mathbf{M}^{(p)}\f$ and the
   * stiffness matrices \f$\mathbf{K}^{(p)}\f$ of the pressure field using the
   * WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\mathbf{M}^{(p)}\f$ and \f$\mathbf{K}^{(p)}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \mathbf{M}^{(p)}_{ij} &= \int\limits_\Omega \varphi_i\varphi_j\dint{V}\,,\\
   * \mathbf{K}^{(p)}_{ij} &= \int\limits_\Omega \nabla\varphi_i\cdot\nabla\varphi_j\dint{V}\,.
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
   * @brief This method assembles the right-hand side of the diffusion step
   * using the WorkStream approach.
   *
   * This method assembles the following weak form into the vector representing
   * the right-hand side \f$\mathbf{b}\f$
   * \f[
   * \mathbf{b}_i = -\int\limits_\Omega (\alpha_1 \bs{v}^{n-1}
   * + \alpha_0 \bs{v}^{n-2}) \cdot
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
   * the right-hand side \f$\mathbf{b}\f$
   * \f[
   * \mathbf{b}_i = -\int\limits_\Omega (\nabla\cdot\bs{v}) \varphi_i \dint{V}\,,
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
   * advection matrix \f$\mathbf{C}^{(\bs{v})}\f$
   * \f[
   * \mathbf{C}^{(\bs{v})}_{ij} = \int\limits_\Omega
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
