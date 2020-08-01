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

namespace VelocityMassLaplaceAssembly
{
template <int dim>
MappingData<dim>::MappingData(const unsigned int velocity_dofs_per_cell)
  : velocity_dofs_per_cell(velocity_dofs_per_cell),
    local_velocity_mass_matrix(velocity_dofs_per_cell, 
                               velocity_dofs_per_cell),
    local_velocity_laplace_matrix(velocity_dofs_per_cell, 
                                  velocity_dofs_per_cell),
    local_velocity_dof_indices(velocity_dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(
                      const FESystem<dim> &velocity_fe,
                      const QGauss<dim>   &velocity_quadrature_formula,
                      const UpdateFlags   velocity_update_flags)
  : velocity_fe_values(velocity_fe, 
                       velocity_quadrature_formula,
                       velocity_update_flags),
    n_q_points(velocity_quadrature_formula.size()),
    velocity_dofs_per_cell(velocity_fe.dofs_per_cell),
    phi_velocity(velocity_dofs_per_cell),
    grad_phi_velocity(velocity_dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
  : velocity_fe_values(data.velocity_fe_values.get_fe(), 
                       data.velocity_fe_values.get_quadrature(),
                       data.velocity_fe_values.get_update_flags()),
    n_q_points(data.n_q_points),
    velocity_dofs_per_cell(data.velocity_dofs_per_cell),
    phi_velocity(velocity_dofs_per_cell),
    grad_phi_velocity(velocity_dofs_per_cell) 
  {}
} // namespace VelocityMassLaplaceAssembly

namespace PressureMassLaplaceAssembly
{
template <int dim>
MappingData<dim>::MappingData(const unsigned int pressure_dofs_per_cell)
  : pressure_dofs_per_cell(pressure_dofs_per_cell),
    local_pressure_mass_matrix(pressure_dofs_per_cell, 
                               pressure_dofs_per_cell),
    local_pressure_laplace_matrix(pressure_dofs_per_cell, 
                                  pressure_dofs_per_cell),
    local_pressure_dof_indices(pressure_dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(
                      const FE_Q<dim>     &pressure_fe,
                      const QGauss<dim>   &pressure_quadrature_formula,
                      const UpdateFlags   pressure_update_flags)
  : pressure_fe_values(pressure_fe, 
                       pressure_quadrature_formula,
                       pressure_update_flags),
    n_q_points(pressure_quadrature_formula.size()),
    pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
    phi_pressure(pressure_dofs_per_cell),
    grad_phi_pressure(pressure_dofs_per_cell)
  {}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
  : pressure_fe_values(data.pressure_fe_values.get_fe(), 
                       data.pressure_fe_values.get_quadrature(),
                       data.pressure_fe_values.get_update_flags()),
    n_q_points(data.n_q_points),
    pressure_dofs_per_cell(data.pressure_dofs_per_cell),
    phi_pressure(pressure_dofs_per_cell),
    grad_phi_pressure(pressure_dofs_per_cell) 
  {}
} // namespace PressureMassLaplaceAssembly

namespace DiffusionStepRightHandSideAssembly
{
template <int dim>
MappingData<dim>::MappingData(const unsigned int velocity_dofs_per_cell)
  : velocity_dofs_per_cell(velocity_dofs_per_cell),
    local_diffusion_step_rhs(velocity_dofs_per_cell),
    local_velocity_dof_indices(velocity_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(
                      const FESystem<dim> &velocity_fe,
                      const FE_Q<dim>     &pressure_fe,
                      const QGauss<dim>   &velocity_quadrature_formula,
                      const UpdateFlags   velocity_update_flags,
                      const UpdateFlags   pressure_update_flags)
  : velocity_fe_values(velocity_fe,
                       velocity_quadrature_formula,
                       velocity_update_flags),
    pressure_fe_values(pressure_fe,
                       velocity_quadrature_formula,
                       pressure_update_flags),
    n_q_points(velocity_quadrature_formula.size()),
    velocity_dofs_per_cell(velocity_fe.dofs_per_cell),
    pressure_tmp_values(velocity_dofs_per_cell),
    velocity_tmp_values(velocity_dofs_per_cell),
    phi_velocity(velocity_dofs_per_cell),
    div_phi_velocity(velocity_dofs_per_cell)
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
    pressure_tmp_values(velocity_dofs_per_cell),
    velocity_tmp_values(velocity_dofs_per_cell),
    phi_velocity(velocity_dofs_per_cell),
    div_phi_velocity(velocity_dofs_per_cell)
{}
} // namespace DiffusionStepRightHandSideAssembly

namespace ProjectionStepRightHandSideAssembly
{
template <int dim>
MappingData<dim>::MappingData(const unsigned int pressure_dofs_per_cell)
  : pressure_dofs_per_cell(pressure_dofs_per_cell),
    local_projection_step_rhs(pressure_dofs_per_cell),
    local_pressure_dof_indices(pressure_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(
                      const FESystem<dim> &velocity_fe,
                      const FE_Q<dim>     &pressure_fe,
                      const QGauss<dim>   &pressure_quadrature_formula,
                      const UpdateFlags   velocity_update_flags,
                      const UpdateFlags   pressure_update_flags)
  : velocity_fe_values(velocity_fe,
                       pressure_quadrature_formula,
                       velocity_update_flags),
    pressure_fe_values(pressure_fe,
                       pressure_quadrature_formula,
                       pressure_update_flags),
    n_q_points(pressure_quadrature_formula.size()),
    pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
    velocity_n_divergence_values(pressure_dofs_per_cell)
    //phi_pressure(pressure_dofs_per_cell)
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
    pressure_dofs_per_cell(data.pressure_dofs_per_cell), 
    velocity_n_divergence_values(pressure_dofs_per_cell)
    //phi_pressure(pressure_dofs_per_cell)
{}
} // namespace ProjectionStepRightHandSideAssembly

} // namespace Step35
// explicit instantiations
template struct Step35::AdvectionAssembly::MappingData<2>;
template struct Step35::AdvectionAssembly::MappingData<3>;
template struct Step35::AdvectionAssembly::LocalCellData<2>;
template struct Step35::AdvectionAssembly::LocalCellData<3>;
template struct Step35::PressureGradientAssembly::MappingData<2>;
template struct Step35::PressureGradientAssembly::MappingData<3>;
template struct Step35::PressureGradientAssembly::LocalCellData<2>;
template struct Step35::PressureGradientAssembly::LocalCellData<3>;
template struct Step35::VelocityMassLaplaceAssembly::MappingData<2>;
template struct Step35::VelocityMassLaplaceAssembly::MappingData<3>;
template struct Step35::VelocityMassLaplaceAssembly::LocalCellData<2>;
template struct Step35::VelocityMassLaplaceAssembly::LocalCellData<3>;
template struct Step35::PressureMassLaplaceAssembly::MappingData<2>;
template struct Step35::PressureMassLaplaceAssembly::MappingData<3>;
template struct Step35::PressureMassLaplaceAssembly::LocalCellData<2>;
template struct Step35::PressureMassLaplaceAssembly::LocalCellData<3>;
template struct Step35::DiffusionStepRightHandSideAssembly::MappingData<2>;
template struct Step35::DiffusionStepRightHandSideAssembly::MappingData<3>;
template struct Step35::DiffusionStepRightHandSideAssembly::LocalCellData<2>;
template struct Step35::DiffusionStepRightHandSideAssembly::LocalCellData<3>;
template struct Step35::ProjectionStepRightHandSideAssembly::MappingData<2>;
template struct Step35::ProjectionStepRightHandSideAssembly::MappingData<3>;
template struct Step35::ProjectionStepRightHandSideAssembly::LocalCellData<2>;
template struct Step35::ProjectionStepRightHandSideAssembly::LocalCellData<3>;