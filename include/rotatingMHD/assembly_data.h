#ifndef INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_q.h>

namespace Step35
{
  using namespace dealii;
namespace AdvectionAssembly
{
template <int dim>
struct MappingData
{
  FullMatrix<double>                   local_matrix;
  std::vector<types::global_dof_index> local_dof_indices;
  MappingData(const unsigned int dofs_per_cell);
};

template <int dim>  
struct LocalCellData
{
  FEValues<dim>               fe_values;
  unsigned int                n_q_points;
  unsigned int                dofs_per_cell;
  std::vector<double>         extrapolated_velocity_divergences;
  std::vector<Tensor<1,dim>>  extrapolated_velocity_values;
  std::vector<Tensor<1,dim>>  phi_velocity;
  std::vector<Tensor<2,dim>>  grad_phi_velocity;

  LocalCellData(const FESystem<dim> &fe,
                const QGauss<dim>   &quadrature_formula,
                const UpdateFlags   flags);
  LocalCellData(const LocalCellData &data);
};
}// namespace AdvectionAssembly

namespace PressureGradientAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                         velocity_dofs_per_cell;
  unsigned int                         pressure_dofs_per_cell;
  FullMatrix<double>                   local_matrix;
  std::vector<types::global_dof_index> local_velocity_dof_indices;
  std::vector<types::global_dof_index> local_pressure_dof_indices;

  MappingData(const unsigned int velocity_dofs_per_cell,
              const unsigned int pressure_dofs_per_cell);
};
template <int dim>
struct LocalCellData
{
  FEValues<dim>         velocity_fe_values;
  FEValues<dim>         pressure_fe_values;
  unsigned int          n_q_points;
  unsigned int          velocity_dofs_per_cell;
  unsigned int          pressure_dofs_per_cell;
  std::vector<double>   div_phi_velocity;
  std::vector<double>   phi_pressure;

  LocalCellData(const FESystem<dim> &velocity_fe,
                const FE_Q<dim>     &pressure_fe,
                const QGauss<dim>   &quadrature_formula,
                const UpdateFlags   velocity_flags,
                const UpdateFlags   pressure_flags);
  LocalCellData(const LocalCellData &data);
};
} // namespace PressureGradientAssembly

namespace VelocityMassLaplaceAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          velocity_dofs_per_cell;
  FullMatrix<double>                    local_velocity_mass_matrix;
  FullMatrix<double>                    local_velocity_laplace_matrix;
  std::vector<types::global_dof_index>  local_velocity_dof_indices;

  MappingData(const unsigned int velocity_dofs_per_cell);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>               velocity_fe_values;
  unsigned int                n_q_points;
  unsigned int                velocity_dofs_per_cell;
  std::vector<Tensor<1,dim>>  phi_velocity;
  std::vector<Tensor<2,dim>>  grad_phi_velocity;

  LocalCellData(const FESystem<dim> &velocity_fe,
                const QGauss<dim>   &velocity_quadrature_formula,
                const UpdateFlags   velocity_update_flags);
  LocalCellData(const LocalCellData &data);
};
} // namespace VelocityMassLaplaceAssembly

namespace PressureMassLaplaceAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          pressure_dofs_per_cell;
  FullMatrix<double>                    local_pressure_mass_matrix;
  FullMatrix<double>                    local_pressure_laplace_matrix;
  std::vector<types::global_dof_index>  local_pressure_dof_indices;

  MappingData(const unsigned int pressure_dofs_per_cell);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>               pressure_fe_values;
  unsigned int                n_q_points;
  unsigned int                pressure_dofs_per_cell;
  std::vector<double>         phi_pressure;
  std::vector<Tensor<1,dim>>  grad_phi_pressure;

  LocalCellData(const FE_Q<dim>     &pressure_fe,
                const QGauss<dim>   &pressure_quadrature_formula,
                const UpdateFlags   pressure_update_flags);
  LocalCellData(const LocalCellData &data);
};
} // namespace PressureMassLaplaceAssembly

namespace DiffusionStepRightHandSideAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          velocity_dofs_per_cell;
  Vector<double>                        local_diffusion_step_rhs;
  std::vector<types::global_dof_index>  local_velocity_dof_indices;

  MappingData(const unsigned int velocity_dofs_per_cell);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>                         velocity_fe_values;
  unsigned int                          n_q_points;
  unsigned int                          velocity_dofs_per_cell;
  std::vector<double>                   pressure_tmp_values;
  std::vector<Tensor<1, dim>>           velocity_tmp_values;

  LocalCellData(const FESystem<dim>     &velocity_fe,
                const QGauss<dim>       &velocity_quadrature_formula,
                const UpdateFlags       velocity_update_flags);
  LocalCellData(const LocalCellData     &data);
};
} // namespace DiffusionStepRightHandSideAssembly

namespace ProjectionStepRightHandSideAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          pressure_dofs_per_cell;
  Vector<double>                        local_projection_step_rhs;
  std::vector<types::global_dof_index>  local_pressure_dof_indices;

  MappingData(const unsigned int velocity_dofs_per_cell);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>                         pressure_fe_values;
  unsigned int                          n_q_points;
  unsigned int                          pressure_dofs_per_cell;
  std::vector<double>                   velocity_n_divergence_values;

  LocalCellData(const FE_Q<dim>         &pressure_fe,
                const QGauss<dim>       &pressure_quadrature_formula,
                const UpdateFlags       pressure_update_flags);
  LocalCellData(const LocalCellData     &data);
};
} // namespace ProjectionStepRightHandSideAssembly

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_ */