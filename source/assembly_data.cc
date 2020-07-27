/*
 * assembly_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/assembly_data.h>

namespace Step35
{
  using namespace dealii;
namespace AdvectionAssembly
{
template <int dim>
MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
    : local_matrix(dofs_per_cell, dofs_per_cell),
      local_dof_indices(dofs_per_cell)
  {}

template <int dim>  
LocalCellData<dim>::LocalCellData(
                            const FESystem<dim> &fe,
                            const QGauss<dim>   &quadrature_formula,
                            const UpdateFlags   flags)
    : fe_values(fe, quadrature_formula, flags),
      n_q_points(quadrature_formula.size()),
      dofs_per_cell(fe.dofs_per_cell),
      extrapolated_velocity_divergences(n_q_points),
      extrapolated_velocity_values(n_q_points),
      phi_velocity(dofs_per_cell),
      grad_phi_velocity(dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
    : fe_values(data.fe_values.get_fe(),
                data.fe_values.get_quadrature(),
                data.fe_values.get_update_flags()),
      n_q_points(data.n_q_points),
      dofs_per_cell(data.dofs_per_cell),
      extrapolated_velocity_divergences(n_q_points),
      extrapolated_velocity_values(n_q_points),
      phi_velocity(dofs_per_cell),
      grad_phi_velocity(dofs_per_cell)
  {}
}// namespace AdvectionTermAssembly

namespace PressureGradientAssembly
{
template <int dim>
MappingData<dim>::MappingData(
                          const unsigned int velocity_dofs_per_cell,
                          const unsigned int pressure_dofs_per_cell)
    : velocity_dofs_per_cell(velocity_dofs_per_cell),
      pressure_dofs_per_cell(pressure_dofs_per_cell),
      local_matrix(velocity_dofs_per_cell, pressure_dofs_per_cell),
      local_velocity_dof_indices(velocity_dofs_per_cell),
      local_pressure_dof_indices(pressure_dofs_per_cell)
  {}

template <int dim>  
LocalCellData<dim>::LocalCellData(
                          const FESystem<dim> &v_fe,
                          const FE_Q<dim>     &p_fe,
                          const QGauss<dim>   &quadrature_formula,
                          const UpdateFlags   v_flags,
                          const UpdateFlags   p_flags)
    : velocity_fe_values(v_fe, quadrature_formula, v_flags),
      pressure_fe_values(p_fe, quadrature_formula, p_flags),
      n_q_points(quadrature_formula.size()),
      velocity_dofs_per_cell(v_fe.dofs_per_cell),
      pressure_dofs_per_cell(p_fe.dofs_per_cell),
      div_phi_velocity(velocity_dofs_per_cell),
      phi_pressure(pressure_dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
    : velocity_fe_values(data.velocity_fe_values.get_fe(),
                data.velocity_fe_values.get_quadrature(),
                data.velocity_fe_values.get_update_flags()),
      pressure_fe_values(data.pressure_fe_values.get_fe(),
                  data.pressure_fe_values.get_quadrature(),
                  data.pressure_fe_values.get_update_flags()),
      n_q_points(data.n_q_points),
      velocity_dofs_per_cell(data.velocity_dofs_per_cell),
      pressure_dofs_per_cell(data.velocity_dofs_per_cell),
      div_phi_velocity(velocity_dofs_per_cell),
      phi_pressure(pressure_dofs_per_cell)
  {}
}// namespace PressureGradientTermAssembly

}

// explicit instantiations
template struct Step35::AdvectionAssembly::MappingData<2>;
template struct Step35::AdvectionAssembly::MappingData<3>;
template struct Step35::AdvectionAssembly::LocalCellData<2>;
template struct Step35::AdvectionAssembly::LocalCellData<3>;
template struct Step35::PressureGradientAssembly::MappingData<2>;
template struct Step35::PressureGradientAssembly::MappingData<3>;
template struct Step35::PressureGradientAssembly::LocalCellData<2>;
template struct Step35::PressureGradientAssembly::LocalCellData<3>;