/*
 * projection_solver.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_
#define INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_

#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

namespace Step35
{

  using namespace dealii;

template <int dim>
class NavierStokesProjection
{
public:
  NavierStokesProjection(const RunTimeParameters::ParameterSet &data);
  void run(const bool flag_verbose_output = false, 
           const unsigned int n_plots = 10);

private:
  MPI_Comm                            mpi_communicator;
  ConditionalOStream                  pcout;

  RunTimeParameters::ProjectionMethod projection_method;
  double                              dt_n;
  double                              dt_n_minus_1;
  const double                        t_0;
  const double                        T;
  const double                        Re;
  
  TimeDiscretization::VSIMEXCoefficients                  
                                      VSIMEX;
  TimeDiscretization::VSIMEXMethod    time_stepping;
  EquationData::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;
  EquationData::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;
  
  std::vector<types::boundary_id>     boundary_ids;

  parallel::distributed::Triangulation<dim>          
                                      triangulation;

  const unsigned int                  pressure_fe_degree;
  FE_Q<dim>                           pressure_fe;
  DoFHandler<dim>                     pressure_dof_handler;
  AffineConstraints<double>           pressure_constraints;
  QGauss<dim>                         pressure_quadrature_formula;

  IndexSet                            locally_owned_pressure_dofs;
  IndexSet                            locally_relevant_pressure_dofs;

  TrilinosWrappers::SparseMatrix      pressure_mass_matrix;
  TrilinosWrappers::SparseMatrix      pressure_laplace_matrix;

  TrilinosWrappers::MPI::Vector       pressure_n;
  TrilinosWrappers::MPI::Vector       pressure_n_minus_1;
  TrilinosWrappers::MPI::Vector       pressure_tmp;
  TrilinosWrappers::MPI::Vector       pressure_rhs;
  TrilinosWrappers::MPI::Vector       phi_n;
  TrilinosWrappers::MPI::Vector       phi_n_minus_1;

  const unsigned int                  velocity_fe_degree;
  FESystem<dim>                       velocity_fe;
  DoFHandler<dim>                     velocity_dof_handler;
  AffineConstraints<double>           velocity_constraints;
  QGauss<dim>                         velocity_quadrature_formula;

  IndexSet                            locally_owned_velocity_dofs;
  IndexSet                            locally_relevant_velocity_dofs;

  TrilinosWrappers::SparseMatrix      velocity_system_matrix;
  TrilinosWrappers::SparseMatrix      velocity_mass_matrix;
  TrilinosWrappers::SparseMatrix      velocity_laplace_matrix;
  TrilinosWrappers::SparseMatrix      velocity_mass_plus_laplace_matrix;
  TrilinosWrappers::SparseMatrix      velocity_advection_matrix;

  TrilinosWrappers::MPI::Vector       velocity_n;
  TrilinosWrappers::MPI::Vector       velocity_n_minus_1;
  TrilinosWrappers::MPI::Vector       extrapolated_velocity;
  TrilinosWrappers::MPI::Vector       velocity_tmp;
  TrilinosWrappers::MPI::Vector       velocity_rhs;

  TrilinosWrappers::PreconditionILU     diffusion_step_preconditioner;
  TrilinosWrappers::PreconditionILU     projection_step_preconditioner;
  TrilinosWrappers::PreconditionJacobi  correction_step_preconditioner;

  unsigned int                        solver_max_iterations;
  unsigned int                        solver_krylov_size;
  unsigned int                        solver_off_diagonals;
  unsigned int                        solver_update_preconditioner;
  double                              solver_tolerance;
  double                              solver_diag_strength;
  bool                                flag_adpative_time_step;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void make_grid(const unsigned int n_global_refinements);
  void setup_dofs();
  void setup_matrices_and_vectors();
  void assemble_constant_matrices();
  void initialize();
  void diffusion_step(const bool reinit_prec);
  void assemble_diffusion_step();
  void solve_diffusion_step(const bool reinit_prec);
  void projection_step(const bool reinit_prec);
  void assemble_projection_step();
  void solve_projection_step(const bool reinit_prec);
  void pressure_correction(const bool reinit_prec);
  void update_time_step();
  void output_results(const unsigned int step);

  void setup_velocity_matrices();
  void setup_pressure_matrices();

  void assemble_velocity_matrices();
  void assemble_local_velocity_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityMatricesAssembly::LocalCellData<dim>          &scratch,
    VelocityMatricesAssembly::MappingData<dim>            &data);
  void copy_local_to_global_velocity_matrices(
    const VelocityMatricesAssembly::MappingData<dim>      &data);

  void assemble_pressure_matrices();
  void assemble_local_pressure_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureMatricesAssembly::LocalCellData<dim>          &scratch,
    PressureMatricesAssembly::MappingData<dim>            &data);
  void copy_local_to_global_pressure_matrices(
    const PressureMatricesAssembly::MappingData<dim>      &data);

  void assemble_diffusion_step_rhs();
  void assemble_local_diffusion_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
    VelocityRightHandSideAssembly::MappingData<dim>       &data);
  void copy_local_to_global_diffusion_step_rhs(
    const VelocityRightHandSideAssembly::MappingData<dim> &data);

  void assemble_projection_step_rhs();
  void assemble_local_projection_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureRightHandSideAssembly::LocalCellData<dim>     &scratch,
    PressureRightHandSideAssembly::MappingData<dim>       &data);
  void copy_local_to_global_projection_step_rhs(
    const PressureRightHandSideAssembly::MappingData<dim> &data);

  void assemble_velocity_advection_matrix();
  void assemble_local_velocity_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionAssembly::LocalCellData<dim>                 &scratch,
    AdvectionAssembly::MappingData<dim>                   &data);
  void copy_local_to_global_velocity_advection_matrix(
    const AdvectionAssembly::MappingData<dim>             &data);
  
  double compute_max_velocity();
  void point_evaluation(const Point<dim>  &point,
                        unsigned int      time_step,
                        DiscreteTime      time) const;
};

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_ */
