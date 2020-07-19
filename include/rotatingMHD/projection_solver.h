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
  NavierStokesProjection(const RunTimeParameters::Data_Storage &data);

  void run(const bool verbose = false, const unsigned int n_plots = 10);

protected:
  RunTimeParameters::Method type;

  const unsigned int deg;
  const double       dt;
  const double       t_0;
  const double       T;
  const double       Re;

  EquationData::VelocityInflowBC<dim>               vel_exact;
  EquationData::VelocityIC<dim>             vel_ic;
  std::map<types::global_dof_index, double> boundary_values;
  std::vector<types::boundary_id>           boundary_ids;

  Triangulation<dim> triangulation;

  FESystem<dim> fe_velocity;
  FE_Q<dim> fe_pressure;

  DoFHandler<dim> dof_handler_velocity;
  DoFHandler<dim> dof_handler_pressure;

  QGauss<dim> quadrature_pressure;
  QGauss<dim> quadrature_velocity;

  SparsityPattern sparsity_pattern_velocity;
  SparsityPattern sparsity_pattern_pressure;
  SparsityPattern sparsity_pattern_pres_vel;

  SparseMatrix<double> vel_Laplace_plus_Mass;
  SparseMatrix<double> vel_it_matrix;
  SparseMatrix<double> vel_Mass;
  SparseMatrix<double> vel_Laplace;
  SparseMatrix<double> vel_Advection;
  SparseMatrix<double> pres_Laplace;
  SparseMatrix<double> pres_Mass;
  SparseMatrix<double> pres_Diff;
  SparseMatrix<double> pres_iterative;

  Vector<double> pres_n;
  Vector<double> pres_n_minus_1;
  Vector<double> phi_n;
  Vector<double> phi_n_minus_1;
  Vector<double> u_n;
  Vector<double> u_n_minus_1;
  Vector<double> u_star;
  Vector<double> force;
  Vector<double> v_tmp;
  Vector<double> pres_tmp;
  Vector<double> rot_u;

  SparseILU<double>   prec_velocity;
  SparseILU<double>   prec_pres_Laplace;
  SparseDirectUMFPACK prec_mass;
  SparseDirectUMFPACK prec_vel_mass;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation_and_dofs(const unsigned int n_refines);

  void initialize();

  void interpolate_velocity();

  void diffusion_step(const bool reinit_prec);

  void projection_step(const bool reinit_prec);

  void update_pressure(const bool reinit_prec);

private:

  unsigned int vel_max_its;
  unsigned int vel_Krylov_size;
  unsigned int vel_off_diagonals;
  unsigned int vel_update_prec;
  double       vel_eps;
  double       vel_diag_strength;

  void initialize_velocity_matrices();

  void initialize_pressure_matrices();

  using IteratorTuple =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator>;

  using IteratorPair = SynchronousIterators<IteratorTuple>;

  void initialize_gradient_operator();

  struct InitGradPerTaskData
  {
    unsigned int                         vel_dpc;
    unsigned int                         pres_dpc;

    FullMatrix<double>                   local_grad;

    std::vector<types::global_dof_index> vel_local_dof_indices;
    std::vector<types::global_dof_index> pres_local_dof_indices;

    InitGradPerTaskData(const unsigned int vdpc,
                        const unsigned int pdpc);
  };

  struct InitGradScratchData
  {
    unsigned int  nqp;

    FEValues<dim> fe_val_vel;
    FEValues<dim> fe_val_pres;

    InitGradScratchData(const FESystem<dim> &  fe_v,
                        const FE_Q<dim> &  fe_p,
                        const QGauss<dim> &quad,
                        const UpdateFlags  flags_v,
                        const UpdateFlags  flags_p);

    InitGradScratchData(const InitGradScratchData &data);
  };

  void assemble_one_cell_of_gradient(const IteratorPair & SI,
                                     InitGradScratchData &scratch,
                                     InitGradPerTaskData &data);

  void copy_gradient_local_to_global(const InitGradPerTaskData &data);

  void assemble_advection_term();

  struct AdvectionPerTaskData
  {

    FullMatrix<double>                   local_advection;
    std::vector<types::global_dof_index> local_dof_indices;
    AdvectionPerTaskData(const unsigned int dpc);

  };

  struct AdvectionScratchData
  {

    unsigned int                nqp;
    unsigned int                dpc;
    /*
     *

      std::vector<Point<dim>>     u_star_local;
      std::vector<Tensor<1, dim>> grad_u_star;
      std::vector<double>         u_star_tmp;

     *
     */
    std::vector<double>         div_u_star;
    std::vector<Tensor<1,dim>>  u_star_local;
    FEValues<dim>               fe_val;

    AdvectionScratchData(const FESystem<dim> &fe,
                         const QGauss<dim>   &quad,
                         const UpdateFlags    flags);

    AdvectionScratchData(const AdvectionScratchData &data);
  };

  void assemble_one_cell_of_advection(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      AdvectionScratchData &                                scratch,
      AdvectionPerTaskData &                                data);

  void copy_advection_local_to_global(const AdvectionPerTaskData &data);

  void diffusion_component_solve();

  void output_results(const unsigned int step);

//  void assemble_vorticity(const bool reinit_prec);
};

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_PROJECTION_SOLVER_H_ */
