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

  protected:
    RunTimeParameters::ProjectionMethod       projection_method;
    const unsigned int                        p_fe_degree;
    const unsigned int                        v_fe_degree;
    const double                              dt;
    const double                              t_0;
    const double                              T;
    const double                              Re;

    EquationData::VelocityInflowBC<dim>       inflow_bc;
    EquationData::VelocityIC<dim>             v_initial_conditions;
    EquationData::PressureIC<dim>             p_initial_conditions;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>           boundary_ids;

    Triangulation<dim>                        triangulation;
    FESystem<dim>                             v_fe;
    FE_Q<dim>                                 p_fe;
    DoFHandler<dim>                           v_dof_handler;
    DoFHandler<dim>                           p_dof_handler;
    AffineConstraints<double>                 v_constraints;
    AffineConstraints<double>                 p_constraints;
    QGauss<dim>                               p_quadrature_formula;
    QGauss<dim>                               v_quadrature_formula;

    SparsityPattern                           v_sparsity_pattern;
    SparsityPattern                           p_sparsity_pattern;
    SparsityPattern                           mixed_sparsity_pattern;
    SparseMatrix<double>                      v_system_matrix;
    SparseMatrix<double>                      v_mass_matrix;
    SparseMatrix<double>                      v_laplace_matrix;
    SparseMatrix<double>                      v_mass_plus_laplace_matrix;
    SparseMatrix<double>                      v_advection_matrix;
    SparseMatrix<double>                      p_system_matrix;
    SparseMatrix<double>                      p_mass_matrix;
    SparseMatrix<double>                      p_laplace_matrix;
    SparseMatrix<double>                      p_gradient_matrix;

    Vector<double>                            v_n;
    Vector<double>                            v_n_m1;
    Vector<double>                            v_extrapolated;
    Vector<double>                            v_tmp;
    Vector<double>                            v_rot;
    Vector<double>                            v_rhs;
    Vector<double>                            p_n;
    Vector<double>                            p_n_m1;
    Vector<double>                            p_tmp;
    Vector<double>                            p_rhs;
    Vector<double>                            phi_n;
    Vector<double>                            phi_n_m1;

    SparseILU<double>                         v_preconditioner;
    SparseILU<double>                         p_preconditioner;
    SparseDirectUMFPACK                       p_update_preconditioner;
    SparseDirectUMFPACK                       v_rot_preconditioner;

    double                                    dt_n;
    double                                    dt_n_m1;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void make_grid(const unsigned int n_global_refinements);
    void setup_dofs();
    void initialize();
    void diffusion_step_assembly();
    void diffusion_step_solve(const bool reinit_prec);
    void projection_step_assembly(const bool reinit_prec);
    void projection_step_solve(const bool reinit_prec);
    void correction_step(const bool reinit_prec);
    void update_time_step();
    void output_results(const unsigned int step);

  private:
    using IteratorTuple =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator>;
    using IteratorPair = SynchronousIterators<IteratorTuple>;

    unsigned int                              solver_max_iterations;
    unsigned int                              solver_krylov_size;
    unsigned int                              solver_off_diagonals;
    unsigned int                              solver_update_preconditioner;
    double                                    solver_tolerance;
    double                                    solver_diag_strength;
    
    void setup_v_matrices();
    void setup_p_matrices();
    void setup_p_gradient_matrix();
    void assemble_v_matrices();
    void assemble_p_matrices();
    void assemble_p_gradient_matrix();
    void local_assemble_p_gradient_matrix(
      const IteratorPair                                    &SI,
      PressureGradientTermAssembly::LocalCellData<dim>      &scratch,
      PressureGradientTermAssembly::MappingData<dim>        &data);
    void mapping_p_gradient_matrix(
      const PressureGradientTermAssembly::MappingData<dim>  &data);

    void assemble_v_advection_matrix();
    void local_assemble_v_advection_matrix(
      const typename DoFHandler<dim>::active_cell_iterator  &cell,
      AdvectionTermAssembly::LocalCellData<dim>             &scratch,
      AdvectionTermAssembly::MappingData<dim>               &data);
    void mapping_v_advection_matrix(
      const AdvectionTermAssembly::MappingData<dim>         &data);
    
    double compute_max_velocity();
  };

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_ */
