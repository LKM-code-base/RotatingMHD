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
  RunTimeParameters::ProjectionMethod projection_method;
  double                              dt_n;
  double                              dt_n_minus_1;
  const double                        t_0;
  const double                        T;
  const double                        Re;

  EquationData::VelocityInflowBoundaryCondition<dim>  
                                      inflow_boundary_condition;
  EquationData::VelocityInitialCondition<dim>         
                                      velocity_initial_conditions;
  EquationData::PressureInitialCondition<dim>         
                                      pressure_initial_conditions;
  
  std::map<types::global_dof_index, double> 
                                      boundary_values;
  std::vector<types::boundary_id>     boundary_ids;

  Triangulation<dim>                  triangulation;

  const unsigned int                  pressure_fe_degree;
  FE_Q<dim>                           pressure_fe;
  DoFHandler<dim>                     pressure_dof_handler;
  AffineConstraints<double>           pressure_constraints;
  QGauss<dim>                         pressure_quadrature_formula;

  SparsityPattern                     pressure_sparsity_pattern;
  SparsityPattern                     mixed_sparsity_pattern;
  SparseMatrix<double>                pressure_mass_matrix;
  SparseMatrix<double>                pressure_laplace_matrix;
  SparseMatrix<double>                pressure_gradient_matrix;
  //SparseMatrix<double>                pressure_system_matrix;

  Vector<double>                      pressure_n;
  Vector<double>                      pressure_n_minus_1;
  Vector<double>                      pressure_tmp;
  Vector<double>                      pressure_rhs;
  Vector<double>                      phi_n;
  Vector<double>                      phi_n_minus_1;

  const unsigned int                  velocity_fe_degree;
  FESystem<dim>                       velocity_fe;
  DoFHandler<dim>                     velocity_dof_handler;
  AffineConstraints<double>           velocity_constraints;
  QGauss<dim>                         velocity_quadrature_formula;

  SparsityPattern                     velocity_sparsity_pattern;
  SparseMatrix<double>                velocity_system_matrix;
  SparseMatrix<double>                velocity_mass_matrix;
  SparseMatrix<double>                velocity_laplace_matrix;
  SparseMatrix<double>                velocity_mass_plus_laplace_matrix;
  SparseMatrix<double>                velocity_advection_matrix;

  Vector<double>                      velocity_n;
  Vector<double>                      velocity_n_minus_1;
  Vector<double>                      extrapolated_velocity;
  Vector<double>                      velocity_tmp;
  Vector<double>                      velocity_rhs;

  SparseILU<double>                   diffusion_step_preconditioner;
  SparseILU<double>                   projection_step_preconditioner;
  SparseDirectUMFPACK                 pressure_correction_preconditioner;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  using IteratorTuple =
    std::tuple<typename DoFHandler<dim>::active_cell_iterator,
               typename DoFHandler<dim>::active_cell_iterator>;
  using IteratorPair = SynchronousIterators<IteratorTuple>;

  unsigned int                        solver_max_iterations;
  unsigned int                        solver_krylov_size;
  unsigned int                        solver_off_diagonals;
  unsigned int                        solver_update_preconditioner;
  double                              solver_tolerance;
  double                              solver_diag_strength;
  bool                                flag_adpative_time_step;

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
  void setup_pressure_gradient_matrix();
  void assemble_velocity_matrices();
  void assemble_pressure_matrices();
  void assemble_pressure_gradient_matrix();
  void assemble_local_pressure_gradient_matrix(
    const IteratorPair                                    &SI,
    PressureGradientAssembly::LocalCellData<dim>          &scratch,
    PressureGradientAssembly::MappingData<dim>            &data);
  void copy_loca_to_global_pressure_gradient_matrix(
    const PressureGradientAssembly::MappingData<dim>      &data);

  void assemble_velocity_advection_matrix();
  void assemble_local_velocity_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionAssembly::LocalCellData<dim>                 &scratch,
    AdvectionAssembly::MappingData<dim>                   &data);
  void copy_local_to_global_velocity_advection_matrix(
    const AdvectionAssembly::MappingData<dim>             &data);
  
  double compute_max_velocity();
};

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_ */
