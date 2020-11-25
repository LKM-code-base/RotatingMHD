/*
 * assembly_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/assembly_data.h>

namespace RMHD
{

using namespace dealii;

namespace AdvectionAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
:
local_matrix(dofs_per_cell, dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
local_matrix(data.local_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>  
LocalCellData<dim>::LocalCellData
(const FESystem<dim>   &fe,
 const Quadrature<dim> &quadrature_formula,
 const UpdateFlags      flags)
:
fe_values(fe, quadrature_formula, flags),
n_q_points(quadrature_formula.size()),
dofs_per_cell(fe.dofs_per_cell),
extrapolated_velocity_divergences(n_q_points),
extrapolated_velocity_values(n_q_points),
extrapolated_velocity_curls(n_q_points),
phi_velocity(dofs_per_cell),
grad_phi_velocity(dofs_per_cell),
curl_phi_velocity(dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
fe_values(data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell),
extrapolated_velocity_divergences(n_q_points),
extrapolated_velocity_values(n_q_points),
extrapolated_velocity_curls(n_q_points),
phi_velocity(dofs_per_cell),
grad_phi_velocity(dofs_per_cell),
curl_phi_velocity(dofs_per_cell)
{}

}// namespace AdvectionTermAssembly

namespace VelocityMatricesAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int velocity_dofs_per_cell)
:
velocity_dofs_per_cell(velocity_dofs_per_cell),
local_velocity_mass_matrix(velocity_dofs_per_cell,
                           velocity_dofs_per_cell),
local_velocity_laplace_matrix(velocity_dofs_per_cell,
                              velocity_dofs_per_cell),
local_velocity_dof_indices(velocity_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
velocity_dofs_per_cell(data.velocity_dofs_per_cell),
local_velocity_mass_matrix(data.local_velocity_mass_matrix),
local_velocity_laplace_matrix(data.local_velocity_laplace_matrix),
local_velocity_dof_indices(data.local_velocity_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const FESystem<dim> &velocity_fe,
 const Quadrature<dim>   &velocity_quadrature_formula,
 const UpdateFlags    velocity_update_flags)
:
velocity_fe_values(velocity_fe,
                   velocity_quadrature_formula,
                   velocity_update_flags),
n_q_points(velocity_quadrature_formula.size()),
velocity_dofs_per_cell(velocity_fe.dofs_per_cell),
phi_velocity(velocity_dofs_per_cell),
grad_phi_velocity(velocity_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
velocity_fe_values(data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
n_q_points(data.n_q_points),
velocity_dofs_per_cell(data.velocity_dofs_per_cell),
phi_velocity(velocity_dofs_per_cell),
grad_phi_velocity(velocity_dofs_per_cell)
{}

} // namespace VelocityMatricesAssembly

namespace PressureMatricesAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int pressure_dofs_per_cell)
:
pressure_dofs_per_cell(pressure_dofs_per_cell),
local_pressure_mass_matrix(pressure_dofs_per_cell,
                           pressure_dofs_per_cell),
local_phi_laplace_matrix(pressure_dofs_per_cell,
                              pressure_dofs_per_cell),
local_pressure_dof_indices(pressure_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
local_pressure_mass_matrix(data.local_pressure_mass_matrix),
local_phi_laplace_matrix(data.local_phi_laplace_matrix),
local_pressure_dof_indices(data.local_pressure_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const FE_Q<dim>     &pressure_fe,
 const Quadrature<dim>   &pressure_quadrature_formula,
 const UpdateFlags    pressure_update_flags)
:
pressure_fe_values(pressure_fe,
                   pressure_quadrature_formula,
                   pressure_update_flags),
n_q_points(pressure_quadrature_formula.size()),
pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
pressure_fe_values(data.pressure_fe_values.get_fe(),
                   data.pressure_fe_values.get_quadrature(),
                   data.pressure_fe_values.get_update_flags()),
n_q_points(data.n_q_points),
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

} // namespace PressureMatricesAssembly

namespace VelocityRightHandSideAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int velocity_dofs_per_cell)
:
velocity_dofs_per_cell(velocity_dofs_per_cell),
local_diffusion_step_rhs(velocity_dofs_per_cell),
local_matrix_for_inhomogeneous_bc(velocity_dofs_per_cell,
                                  velocity_dofs_per_cell),
local_velocity_dof_indices(velocity_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
velocity_dofs_per_cell(data.velocity_dofs_per_cell),
local_diffusion_step_rhs(data.local_diffusion_step_rhs),
local_matrix_for_inhomogeneous_bc(data.local_matrix_for_inhomogeneous_bc),
local_velocity_dof_indices(data.local_velocity_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const FESystem<dim>    &velocity_fe,
 const FE_Q<dim>        &pressure_fe,
 const Quadrature<dim>  &velocity_quadrature_formula,
 const Quadrature<dim-1>&velocity_face_quadrature_formula,
 const UpdateFlags      velocity_update_flags,
 const UpdateFlags      pressure_update_flags,
 const UpdateFlags      velocity_face_update_flags)
:
velocity_fe_values(velocity_fe,
                   velocity_quadrature_formula,
                   velocity_update_flags),
pressure_fe_values(pressure_fe,
                   velocity_quadrature_formula,
                   pressure_update_flags),
velocity_fe_face_values(velocity_fe,
                        velocity_face_quadrature_formula,
                        velocity_face_update_flags),
n_q_points(velocity_quadrature_formula.size()),
n_face_q_points(velocity_face_quadrature_formula.size()),
velocity_dofs_per_cell(velocity_fe.dofs_per_cell),
pressure_tmp_values(n_q_points),
velocity_tmp_values(n_q_points),
extrapolated_velocity_divergences(n_q_points),
extrapolated_velocity_values(n_q_points),
extrapolated_velocity_curls(n_q_points),
old_velocity_divergences(n_q_points),
old_velocity_values(n_q_points),
old_velocity_curls(n_q_points),
old_velocity_gradients(n_q_points),
old_old_velocity_divergences(n_q_points),
old_old_velocity_values(n_q_points),
old_old_velocity_curls(n_q_points),
old_old_velocity_gradients(n_q_points),
body_force_values(n_q_points),
neumann_function_values(n_face_q_points),
phi_velocity(velocity_dofs_per_cell),
div_phi_velocity(velocity_dofs_per_cell),
grad_phi_velocity(velocity_dofs_per_cell),
curl_phi_velocity(velocity_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
velocity_fe_values(data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
pressure_fe_values(data.pressure_fe_values.get_fe(),
                   data.pressure_fe_values.get_quadrature(),
                   data.pressure_fe_values.get_update_flags()),
velocity_fe_face_values(data.velocity_fe_face_values.get_fe(),
                        data.velocity_fe_face_values.get_quadrature(),
                        data.velocity_fe_face_values.get_update_flags()),
n_q_points(data.n_q_points),
n_face_q_points(data.n_face_q_points),
velocity_dofs_per_cell(data.velocity_dofs_per_cell),
pressure_tmp_values(n_q_points),
velocity_tmp_values(n_q_points),
extrapolated_velocity_divergences(n_q_points),
extrapolated_velocity_values(n_q_points),
extrapolated_velocity_curls(n_q_points),
old_velocity_divergences(n_q_points),
old_velocity_values(n_q_points),
old_velocity_curls(n_q_points),
old_velocity_gradients(n_q_points),
old_old_velocity_divergences(n_q_points),
old_old_velocity_values(n_q_points),
old_old_velocity_curls(n_q_points),
old_old_velocity_gradients(n_q_points),
body_force_values(n_q_points),
neumann_function_values(n_face_q_points),
phi_velocity(velocity_dofs_per_cell),
face_phi_velocity(velocity_dofs_per_cell),
div_phi_velocity(velocity_dofs_per_cell),
grad_phi_velocity(velocity_dofs_per_cell),
curl_phi_velocity(velocity_dofs_per_cell)
{}

} // namespace VelocityRightHandSideAssembly

namespace PressureRightHandSideAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int pressure_dofs_per_cell)
:
pressure_dofs_per_cell(pressure_dofs_per_cell),
local_projection_step_rhs(pressure_dofs_per_cell),
local_pressure_space_projection_rhs(pressure_dofs_per_cell),
local_matrix_for_inhomogeneous_bc(pressure_dofs_per_cell,
                                  pressure_dofs_per_cell),
local_pressure_dof_indices(pressure_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
local_projection_step_rhs(data.local_projection_step_rhs),
local_pressure_space_projection_rhs(data.local_pressure_space_projection_rhs),
local_matrix_for_inhomogeneous_bc(data.local_matrix_for_inhomogeneous_bc),
local_pressure_dof_indices(data.local_pressure_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const FESystem<dim> &velocity_fe,
 const FE_Q<dim>     &pressure_fe,
 const Quadrature<dim>   &pressure_quadrature_formula,
 const UpdateFlags    velocity_update_flags,
 const UpdateFlags    pressure_update_flags)
:
velocity_fe_values(velocity_fe,
                   pressure_quadrature_formula,
                   velocity_update_flags),
pressure_fe_values(pressure_fe,
                   pressure_quadrature_formula,
                   pressure_update_flags),
n_q_points(pressure_quadrature_formula.size()),
pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
velocity_divergence_values(n_q_points),
phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
velocity_fe_values(data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
pressure_fe_values(data.pressure_fe_values.get_fe(),
                   data.pressure_fe_values.get_quadrature(),
                   data.pressure_fe_values.get_update_flags()),
n_q_points(data.n_q_points),
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
velocity_divergence_values(n_q_points),
phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

} // namespace PressureRightHandSideAssembly

namespace PoissonPrestepRightHandSideAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int pressure_dofs_per_cell)
:
pressure_dofs_per_cell(pressure_dofs_per_cell),
local_poisson_prestep_rhs(pressure_dofs_per_cell),
local_matrix_for_inhomogeneous_bc(pressure_dofs_per_cell,
                                  pressure_dofs_per_cell),
local_pressure_dof_indices(pressure_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
local_poisson_prestep_rhs(data.local_poisson_prestep_rhs),
local_matrix_for_inhomogeneous_bc(data.local_matrix_for_inhomogeneous_bc),
local_pressure_dof_indices(data.local_pressure_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const FESystem<dim>     &velocity_fe,
 const FE_Q<dim>         &pressure_fe,
 const Quadrature<dim>   &pressure_quadrature_formula,
 const Quadrature<dim-1> &pressure_face_quadrature_formula,
 const UpdateFlags        velocity_face_update_flags,
 const UpdateFlags        pressure_update_flags,
 const UpdateFlags        pressure_face_update_flags)
:
pressure_fe_values(pressure_fe,
                   pressure_quadrature_formula,
                   pressure_update_flags),
velocity_fe_face_values(velocity_fe,
                        pressure_face_quadrature_formula,
                        velocity_face_update_flags),
pressure_fe_face_values(pressure_fe,
                        pressure_face_quadrature_formula,
                        pressure_face_update_flags),
n_q_points(pressure_quadrature_formula.size()),
n_face_q_points(pressure_face_quadrature_formula.size()),
pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
body_force_divergence_values(n_q_points),
velocity_laplacian_values(n_face_q_points),
body_force_values(n_face_q_points),
normal_vectors(n_face_q_points),
phi_pressure(pressure_dofs_per_cell),
face_phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
pressure_fe_values(data.pressure_fe_values.get_fe(),
                   data.pressure_fe_values.get_quadrature(),
                   data.pressure_fe_values.get_update_flags()),
velocity_fe_face_values(data.velocity_fe_face_values.get_fe(),
                        data.velocity_fe_face_values.get_quadrature(),
                        data.velocity_fe_face_values.get_update_flags()),
pressure_fe_face_values(data.pressure_fe_face_values.get_fe(),
                        data.pressure_fe_face_values.get_quadrature(),
                        data.pressure_fe_face_values.get_update_flags()),
n_q_points(data.n_q_points),
n_face_q_points(data.n_face_q_points),
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
body_force_divergence_values(n_q_points),
velocity_laplacian_values(n_face_q_points),
body_force_values(n_face_q_points),
normal_vectors(n_face_q_points),
phi_pressure(pressure_dofs_per_cell),
face_phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

} // PoissonPrestepRightHandSideAssembly


} // namespace RMHD

// explicit instantiations
template struct RMHD::AdvectionAssembly::MappingData<2>;
template struct RMHD::AdvectionAssembly::MappingData<3>;

template struct RMHD::AdvectionAssembly::LocalCellData<2>;
template struct RMHD::AdvectionAssembly::LocalCellData<3>;

template struct RMHD::VelocityMatricesAssembly::MappingData<2>;
template struct RMHD::VelocityMatricesAssembly::MappingData<3>;

template struct RMHD::VelocityMatricesAssembly::LocalCellData<2>;
template struct RMHD::VelocityMatricesAssembly::LocalCellData<3>;

template struct RMHD::PressureMatricesAssembly::MappingData<2>;
template struct RMHD::PressureMatricesAssembly::MappingData<3>;

template struct RMHD::PressureMatricesAssembly::LocalCellData<2>;
template struct RMHD::PressureMatricesAssembly::LocalCellData<3>;

template struct RMHD::VelocityRightHandSideAssembly::MappingData<2>;
template struct RMHD::VelocityRightHandSideAssembly::MappingData<3>;

template struct RMHD::VelocityRightHandSideAssembly::LocalCellData<2>;
template struct RMHD::VelocityRightHandSideAssembly::LocalCellData<3>;

template struct RMHD::PressureRightHandSideAssembly::MappingData<2>;
template struct RMHD::PressureRightHandSideAssembly::MappingData<3>;

template struct RMHD::PressureRightHandSideAssembly::LocalCellData<2>;
template struct RMHD::PressureRightHandSideAssembly::LocalCellData<3>;

template struct RMHD::PoissonPrestepRightHandSideAssembly::MappingData<2>;
template struct RMHD::PoissonPrestepRightHandSideAssembly::MappingData<3>;

template struct RMHD::PoissonPrestepRightHandSideAssembly::LocalCellData<2>;
template struct RMHD::PoissonPrestepRightHandSideAssembly::LocalCellData<3>;
