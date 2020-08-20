#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_

#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/benchmark_data.h>

#include <deal.II/lac/trilinos_precondition.h>

#include <string>
#include <vector>

namespace RMHD
{

  using namespace dealii;

template <int dim>
class NavierStokesProjection
{
public:
  NavierStokesProjection(
                const RunTimeParameters::ParameterSet   &parameters,
                Entities::Velocity<dim>                 &velocity,
                Entities::Pressure<dim>                 &pressure,
                TimeDiscretization::VSIMEXCoefficients  &VSIMEX,
                TimeDiscretization::VSIMEXMethod        &time_stepping);
  void setup();
  void solve(const unsigned int step);
  
private:
  RunTimeParameters::ProjectionMethod     projection_method;
  const double                            Re;
  double                                  dt_n;
  double                                  dt_n_minus_1;
  Entities::Velocity<dim>                 &velocity;
  Entities::Pressure<dim>                 &pressure;

  TimeDiscretization::VSIMEXCoefficients  &VSIMEX;
  TimeDiscretization::VSIMEXMethod        &time_stepping;

  TrilinosWrappers::SparseMatrix        velocity_system_matrix;
  TrilinosWrappers::SparseMatrix        velocity_mass_matrix;
  TrilinosWrappers::SparseMatrix        velocity_laplace_matrix;
  TrilinosWrappers::SparseMatrix        velocity_mass_plus_laplace_matrix;
  TrilinosWrappers::SparseMatrix        velocity_advection_matrix;

  TrilinosWrappers::MPI::Vector         extrapolated_velocity;
  TrilinosWrappers::MPI::Vector         velocity_tmp;
  TrilinosWrappers::MPI::Vector         velocity_rhs;

  TrilinosWrappers::SparseMatrix        pressure_mass_matrix;
  TrilinosWrappers::SparseMatrix        pressure_laplace_matrix;

  TrilinosWrappers::MPI::Vector         pressure_tmp;
  TrilinosWrappers::MPI::Vector         pressure_rhs;
  TrilinosWrappers::MPI::Vector         phi_n;
  TrilinosWrappers::MPI::Vector         phi_n_minus_1;

  TrilinosWrappers::PreconditionILU     diffusion_step_preconditioner;
  TrilinosWrappers::PreconditionILU     projection_step_preconditioner;
  TrilinosWrappers::PreconditionJacobi  correction_step_preconditioner;

  unsigned int                          solver_max_iterations;
  unsigned int                          solver_krylov_size;
  unsigned int                          solver_off_diagonals;
  unsigned int                          solver_update_preconditioner;
  double                                solver_tolerance;
  double                                solver_diag_strength;
  bool                                  flag_adpative_time_step;

  void setup_matrices();
  void setup_vectors();
  void assemble_constant_matrices();
  void initialize();

  void diffusion_step(const bool reinit_prec);
  void assemble_diffusion_step();
  void solve_diffusion_step(const bool reinit_prec);
  void projection_step(const bool reinit_prec);
  void assemble_projection_step();
  void solve_projection_step(const bool reinit_prec);
  void pressure_correction(const bool reinit_prec);

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
};

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_ */