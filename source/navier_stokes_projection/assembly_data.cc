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

namespace AssemblyData
{

namespace NavierStokesProjection
{

namespace VelocityConstantMatrices
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
Generic::Matrix::Scratch<dim>(data),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

} // namespace VelocityConstantMatrices

namespace PressureConstantMatrices
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
Generic::Matrix::Scratch<dim>(data),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

} // namespace PressureConstantMatrices

namespace AdvectionMatrix
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_divergences(this->n_q_points),
old_old_velocity_divergences(this->n_q_points),
old_velocity_curls(this->n_q_points),
old_old_velocity_curls(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
Generic::Matrix::Scratch<dim>(data),
old_velocity_values(data.n_q_points),
old_old_velocity_values(data.n_q_points),
old_velocity_divergences(data.n_q_points),
old_old_velocity_divergences(data.n_q_points),
old_velocity_curls(data.n_q_points),
old_old_velocity_curls(data.n_q_points),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell),
curl_phi(data.dofs_per_cell)
{}

} // namespace AdvectionMatrix

namespace DiffusionStepRHS
{

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 velocity_fe),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
velocity_fe_face_values(
  mapping,
  velocity_fe,
  face_quadrature_formula,
  velocity_face_update_flags),
pressure_fe_values(
  mapping,
  pressure_fe,
  quadrature_formula,
  pressure_update_flags),
temperature_fe_values(
  mapping,
  temperature_fe,
  quadrature_formula,
  temperature_update_flags),
n_face_q_points(face_quadrature_formula.size()),
old_pressure_values(this->n_q_points),
old_phi_values(this->n_q_points),
old_old_phi_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
old_velocity_divergences(this->n_q_points),
old_old_velocity_divergences(this->n_q_points),
old_velocity_curls(this->n_q_points),
old_old_velocity_curls(this->n_q_points),
old_rotation_values(this->n_q_points),
old_old_rotation_values(this->n_q_points),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
gravity_unit_vector_values(this->n_q_points),
body_force_values(this->n_q_points),
old_body_force_values(this->n_q_points),
old_old_body_force_values(this->n_q_points),
neuamnn_bc_values(n_face_q_points),
old_neuamnn_bc_values(n_face_q_points),
old_old_neuamnn_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
div_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
ScratchBase<dim>(data),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
velocity_fe_face_values(
  data.velocity_fe_face_values.get_mapping(),
  data.velocity_fe_face_values.get_fe(),
  data.velocity_fe_face_values.get_quadrature(),
  data.velocity_fe_face_values.get_update_flags()),
pressure_fe_values(
  data.pressure_fe_values.get_mapping(),
  data.pressure_fe_values.get_fe(),
  data.pressure_fe_values.get_quadrature(),
  data.pressure_fe_values.get_update_flags()),
temperature_fe_values(
  data.temperature_fe_values.get_mapping(),
  data.temperature_fe_values.get_fe(),
  data.temperature_fe_values.get_quadrature(),
  data.temperature_fe_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
old_pressure_values(this->n_q_points),
old_phi_values(this->n_q_points),
old_old_phi_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
old_velocity_divergences(this->n_q_points),
old_old_velocity_divergences(this->n_q_points),
old_velocity_curls(this->n_q_points),
old_old_velocity_curls(this->n_q_points),
old_rotation_values(this->n_q_points),
old_old_rotation_values(this->n_q_points),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
gravity_unit_vector_values(this->n_q_points),
body_force_values(this->n_q_points),
old_body_force_values(this->n_q_points),
old_old_body_force_values(this->n_q_points),
neuamnn_bc_values(n_face_q_points),
old_neuamnn_bc_values(n_face_q_points),
old_old_neuamnn_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
div_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}

} // namespace DiffusionStepRHS

namespace ProjectionStepRHS
{

template <int dim>
Copy<dim>::Copy(const unsigned int dofs_per_cell)
:
CopyBase<dim>(dofs_per_cell),
local_projection_step_rhs(dofs_per_cell),
local_correction_step_rhs(dofs_per_cell)
{}

template <int dim>
Copy<dim>::Copy(const Copy &data)
:
CopyBase<dim>(data),
local_projection_step_rhs(data.local_projection_step_rhs),
local_correction_step_rhs(data.local_correction_step_rhs)
{}

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 pressure_fe),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
pressure_fe_values(
  mapping,
  pressure_fe,
  quadrature_formula,
  pressure_update_flags),
velocity_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
ScratchBase<dim>(data),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
pressure_fe_values(
  data.pressure_fe_values.get_mapping(),
  data.pressure_fe_values.get_fe(),
  data.pressure_fe_values.get_quadrature(),
  data.pressure_fe_values.get_update_flags()),
velocity_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}

} // namespace ProjectionStepRHS

} // namespace NavierStokesProjection

} // namespace AssemblyData

namespace VelocityRightHandSideAssembly
{

template <int dim>
LocalCellData<dim>::LocalCellData
(const FESystem<dim>    &velocity_fe,
 const FE_Q<dim>        &pressure_fe,
 const FE_Q<dim>        &temperature_fe,
 const Quadrature<dim>  &velocity_quadrature_formula,
 const UpdateFlags      velocity_update_flags,
 const UpdateFlags      pressure_update_flags,
 const UpdateFlags      temperature_update_flags)
:
velocity_fe_values(velocity_fe,
                   velocity_quadrature_formula,
                   velocity_update_flags),
pressure_fe_values(pressure_fe,
                   velocity_quadrature_formula,
                   pressure_update_flags),
temperature_fe_values(temperature_fe,
                      velocity_quadrature_formula,
                      temperature_update_flags),
n_q_points(velocity_quadrature_formula.size()),
velocity_dofs_per_cell(velocity_fe.dofs_per_cell),
pressure_tmp_values(n_q_points),
velocity_tmp_values(n_q_points),
phi_velocity(velocity_dofs_per_cell),
div_phi_velocity(velocity_dofs_per_cell),
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
extrapolated_temperature_values(n_q_points),
body_force_values(n_q_points),
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
temperature_fe_values(data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
n_q_points(data.n_q_points),
velocity_dofs_per_cell(data.velocity_dofs_per_cell),
pressure_tmp_values(n_q_points),
velocity_tmp_values(n_q_points),
phi_velocity(velocity_dofs_per_cell),
div_phi_velocity(velocity_dofs_per_cell),
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
extrapolated_temperature_values(n_q_points),
body_force_values(n_q_points),
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
local_matrix_for_inhomogeneous_bc(pressure_dofs_per_cell,
                                  pressure_dofs_per_cell),
local_pressure_dof_indices(pressure_dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
local_projection_step_rhs(data.local_projection_step_rhs),
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
 const FE_Q<dim>         &temperature_fe,
 const Quadrature<dim>   &pressure_quadrature_formula,
 const Quadrature<dim-1> &pressure_face_quadrature_formula,
 const UpdateFlags        velocity_face_update_flags,
 const UpdateFlags        pressure_update_flags,
 const UpdateFlags        pressure_face_update_flags,
 const UpdateFlags        temperature_update_flags,
 const UpdateFlags        temperature_face_update_flags)
:
pressure_fe_values(pressure_fe,
                   pressure_quadrature_formula,
                   pressure_update_flags),
temperature_fe_values(temperature_fe,
                      pressure_quadrature_formula,
                      temperature_update_flags),
velocity_fe_face_values(velocity_fe,
                        pressure_face_quadrature_formula,
                        velocity_face_update_flags),
pressure_fe_face_values(pressure_fe,
                        pressure_face_quadrature_formula,
                        pressure_face_update_flags),
temperature_fe_face_values(temperature_fe,
                           pressure_face_quadrature_formula,
                           temperature_face_update_flags),
n_q_points(pressure_quadrature_formula.size()),
n_face_q_points(pressure_face_quadrature_formula.size()),
pressure_dofs_per_cell(pressure_fe.dofs_per_cell),
body_force_divergence_values(n_q_points),
temperature_gradient_values(n_q_points),
velocity_laplacian_values(n_face_q_points),
body_force_values(n_face_q_points),
temperature_values(n_face_q_points),
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
temperature_fe_values(data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
velocity_fe_face_values(data.velocity_fe_face_values.get_fe(),
                        data.velocity_fe_face_values.get_quadrature(),
                        data.velocity_fe_face_values.get_update_flags()),
pressure_fe_face_values(data.pressure_fe_face_values.get_fe(),
                        data.pressure_fe_face_values.get_quadrature(),
                        data.pressure_fe_face_values.get_update_flags()),
temperature_fe_face_values(data.temperature_fe_face_values.get_fe(),
                           data.temperature_fe_face_values.get_quadrature(),
                           data.temperature_fe_face_values.get_update_flags()),
n_q_points(data.n_q_points),
n_face_q_points(data.n_face_q_points),
pressure_dofs_per_cell(data.pressure_dofs_per_cell),
body_force_divergence_values(n_q_points),
temperature_gradient_values(n_q_points),
velocity_laplacian_values(n_face_q_points),
body_force_values(n_face_q_points),
temperature_values(n_face_q_points),
normal_vectors(n_face_q_points),
phi_pressure(pressure_dofs_per_cell),
face_phi_pressure(pressure_dofs_per_cell),
grad_phi_pressure(pressure_dofs_per_cell)
{}

} // PoissonPrestepRightHandSideAssembly


} // namespace RMHD

// explicit instantiations
template struct RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<3>;

template struct RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<3>;

template struct RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<3>;

template struct RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<3>;

template struct RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy<3>;

template struct RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<2>;
template struct RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<3>;


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
