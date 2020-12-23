#include <rotatingMHD/assembly_data.h>

namespace RMHD
{

namespace AssemblyData
{

namespace HeatEquation
{

namespace ConstantMatrices
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

} // namespace ConstantMatrices

namespace AdvectionMatrix
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &temperature_fe,
  const UpdateFlags         temperature_update_flags,
  const FiniteElement<dim>  &velocity_fe,
  const UpdateFlags         velocity_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 temperature_fe),
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
ScratchBase<dim>(data),
temperature_fe_values(data.temperature_fe_values.get_mapping(),
                      data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
velocity_fe_values(data.velocity_fe_values.get_mapping(),
                   data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
velocity_values(data.n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

} // namespace AdvectionMatrix

namespace RightHandSide
{

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags,
 const UpdateFlags         temperature_face_update_flags,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags)
:
ScratchBase<dim>(quadrature_formula,
            temperature_fe),
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
temperature_fe_face_values(mapping,
                           temperature_fe,
                           face_quadrature_formula,
                           temperature_face_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
n_face_q_points(face_quadrature_formula.size()),
temperature_values(this->n_q_points),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
temperature_gradients(this->n_q_points),
old_temperature_gradients(this->n_q_points),
old_old_temperature_gradients(this->n_q_points),
source_term_values(this->n_q_points),
old_source_term_values(this->n_q_points),
old_old_source_term_values(this->n_q_points),
neuamnn_bc_values(n_face_q_points),
old_neuamnn_bc_values(n_face_q_points),
old_old_neuamnn_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch &data)
:
ScratchBase<dim>(data),
temperature_fe_values(
  data.temperature_fe_values.get_mapping(),
  data.temperature_fe_values.get_fe(),
  data.temperature_fe_values.get_quadrature(),
  data.temperature_fe_values.get_update_flags()),
temperature_fe_face_values(
  data.temperature_fe_face_values.get_mapping(),
  data.temperature_fe_face_values.get_fe(),
  data.temperature_fe_face_values.get_quadrature(),
  data.temperature_fe_face_values.get_update_flags()),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
temperature_values(this->n_q_points),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
temperature_gradients(this->n_q_points),
old_temperature_gradients(this->n_q_points),
old_old_temperature_gradients(this->n_q_points),
source_term_values(this->n_q_points),
old_source_term_values(this->n_q_points),
old_old_source_term_values(this->n_q_points),
neuamnn_bc_values(n_face_q_points),
old_neuamnn_bc_values(n_face_q_points),
old_old_neuamnn_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}

} // namespace RightHandSide

} // namespace HeatEquation

} // namespace AssemblyData

namespace TemperatureConstantMatricesAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_mass_matrix(dofs_per_cell, dofs_per_cell),
local_stiffness_matrix(dofs_per_cell, dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
dofs_per_cell(data.dofs_per_cell),
local_mass_matrix(data.local_mass_matrix),
local_stiffness_matrix(data.local_stiffness_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const Mapping<dim>     &mapping,
 const FE_Q<dim>        &fe,
 const Quadrature<dim>  &quadrature_formula,
 const UpdateFlags      update_flags)
:
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags),
n_q_points(quadrature_formula.size()),
dofs_per_cell(fe.dofs_per_cell),
phi(dofs_per_cell),
grad_phi(dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags()),
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell),
phi(dofs_per_cell),
grad_phi(dofs_per_cell)
{}

} // namespace TemperatureConstantMatricesAssembly

namespace TemperatureAdvectionMatrixAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
dofs_per_cell(data.dofs_per_cell),
local_matrix(data.local_matrix),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const Mapping<dim>     &mapping,
 const FE_Q<dim>        &temperature_fe,
 const FESystem<dim>    &velocity_fe,
 const Quadrature<dim>  &quadrature_formula,
 const UpdateFlags      temperature_update_flags,
 const UpdateFlags      velocity_update_flags)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
n_q_points(quadrature_formula.size()),
dofs_per_cell(temperature_fe.dofs_per_cell),
velocity_values(n_q_points),
phi(dofs_per_cell),
grad_phi(dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
temperature_fe_values(data.temperature_fe_values.get_mapping(),
                      data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
velocity_fe_values(data.velocity_fe_values.get_mapping(),
                   data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell),
velocity_values(n_q_points),
phi(dofs_per_cell),
grad_phi(dofs_per_cell)
{}

} // namespace TemperatureAdvectionMatrixAssembly

namespace TemperatureRightHandSideAssembly
{

template <int dim>
MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bc(dofs_per_cell, dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
MappingData<dim>::MappingData(const MappingData &data)
:
dofs_per_cell(data.dofs_per_cell),
local_rhs(data.local_rhs),
local_matrix_for_inhomogeneous_bc(data.local_matrix_for_inhomogeneous_bc),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
LocalCellData<dim>::LocalCellData
(const Mapping<dim>       &mapping,
 const FE_Q<dim>          &temperature_fe,
 const FESystem<dim>      &velocity_fe,
 const Quadrature<dim>    &quadrature_formula,
 const Quadrature<dim-1>  &face_quadrature_formula,
 const UpdateFlags        temperature_update_flags,
 const UpdateFlags        velocity_update_flags,
 const UpdateFlags        temperature_face_update_flags)
:
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
temperature_fe_face_values(mapping,
                           temperature_fe,
                           face_quadrature_formula,
                           temperature_face_update_flags),
n_q_points(quadrature_formula.size()),
n_face_q_points(face_quadrature_formula.size()),
dofs_per_cell(temperature_fe.dofs_per_cell),
temperature_tmp_values(n_q_points),
source_term_values(n_q_points),
neumann_bc_values(n_face_q_points),
old_temperature_values(n_q_points),
old_old_temperature_values(n_q_points),
old_temperature_gradients(n_q_points),
old_old_temperature_gradients(n_q_points),
velocity_values(n_q_points),
phi(dofs_per_cell),
grad_phi(dofs_per_cell),
face_phi(dofs_per_cell)
{}

template <int dim>
LocalCellData<dim>::LocalCellData(const LocalCellData &data)
:
temperature_fe_values(
  data.temperature_fe_values.get_mapping(),
  data.temperature_fe_values.get_fe(),
  data.temperature_fe_values.get_quadrature(),
  data.temperature_fe_values.get_update_flags()),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
temperature_fe_face_values(
  data.temperature_fe_face_values.get_mapping(),
  data.temperature_fe_face_values.get_fe(),
  data.temperature_fe_face_values.get_quadrature(),
  data.temperature_fe_face_values.get_update_flags()),
n_q_points(data.n_q_points),
n_face_q_points(data.n_face_q_points),
dofs_per_cell(data.dofs_per_cell),
temperature_tmp_values(n_q_points),
source_term_values(n_q_points),
neumann_bc_values(n_face_q_points),
old_temperature_values(n_q_points),
old_old_temperature_values(n_q_points),
old_temperature_gradients(n_q_points),
old_old_temperature_gradients(n_q_points),
velocity_values(n_q_points),
phi(dofs_per_cell),
grad_phi(dofs_per_cell),
face_phi(dofs_per_cell)
{}

} // namespace TemperatureRightHandSideAssembly

} // namespace RMHD

template struct RMHD::AssemblyData::HeatEquation::ConstantMatrices::Scratch<2>;
template struct RMHD::AssemblyData::HeatEquation::ConstantMatrices::Scratch<3>;

template struct RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Scratch<2>;
template struct RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Scratch<3>;

template struct RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<2>;
template struct RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<3>;

template struct RMHD::TemperatureConstantMatricesAssembly::MappingData<2>;
template struct RMHD::TemperatureConstantMatricesAssembly::MappingData<3>;

template struct RMHD::TemperatureConstantMatricesAssembly::LocalCellData<2>;
template struct RMHD::TemperatureConstantMatricesAssembly::LocalCellData<3>;

template struct RMHD::TemperatureAdvectionMatrixAssembly::MappingData<2>;
template struct RMHD::TemperatureAdvectionMatrixAssembly::MappingData<3>;

template struct RMHD::TemperatureAdvectionMatrixAssembly::LocalCellData<2>;
template struct RMHD::TemperatureAdvectionMatrixAssembly::LocalCellData<3>;

template struct RMHD::TemperatureRightHandSideAssembly::MappingData<2>;
template struct RMHD::TemperatureRightHandSideAssembly::MappingData<3>;

template struct RMHD::TemperatureRightHandSideAssembly::LocalCellData<2>;
template struct RMHD::TemperatureRightHandSideAssembly::LocalCellData<3>;