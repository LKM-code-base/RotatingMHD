#include <rotatingMHD/assembly_data.h>

namespace RMHD
{

namespace AssemblyData
{

template <int dim>
CopyBase<dim>::CopyBase(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
CopyBase<dim>::CopyBase(const CopyBase &data)
:
dofs_per_cell(data.dofs_per_cell),
local_dof_indices(data.local_dof_indices)
{}

template <int dim>
ScratchBase<dim>::ScratchBase
(const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &fe)
:
n_q_points(quadrature_formula.size()),
dofs_per_cell(fe.dofs_per_cell)
{}

template <int dim>
ScratchBase<dim>::ScratchBase(const ScratchBase &data)
:
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell)
{}

namespace Generic
{

namespace Matrix
{

template <int dim>
Copy<dim>::Copy(const unsigned int dofs_per_cell)
:
CopyBase<dim>(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell)
{}

template <int dim>
Copy<dim>::Copy(const Copy &data)
:
CopyBase<dim>(data),
local_matrix(data.local_matrix)
{}

template <int dim>
MassStiffnessCopy<dim>::MassStiffnessCopy(const unsigned int dofs_per_cell)
:
CopyBase<dim>(dofs_per_cell),
local_mass_matrix(dofs_per_cell, dofs_per_cell),
local_stiffness_matrix(dofs_per_cell, dofs_per_cell)
{}

template <int dim>
MassStiffnessCopy<dim>::MassStiffnessCopy(const MassStiffnessCopy &data)
:
CopyBase<dim>(data),
local_mass_matrix(data.local_mass_matrix),
local_stiffness_matrix(data.local_stiffness_matrix)
{}


template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
ScratchBase<dim>(quadrature_formula, fe),
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags)
{}

template <int dim>
Scratch<dim>::Scratch(
  const Scratch &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags())
{}

} // namespace Matrix

namespace RightHandSide
{

template <int dim>
Copy<dim>::Copy(const unsigned int dofs_per_cell)
:
CopyBase<dim>(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bc(dofs_per_cell, dofs_per_cell)
{}

template <int dim>
Copy<dim>::Copy(const Copy &data)
:
CopyBase<dim>(data),
local_rhs(data.local_rhs),
local_matrix_for_inhomogeneous_bc(data.local_matrix_for_inhomogeneous_bc)
{}

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
ScratchBase<dim>(quadrature_formula, fe),
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags)
{}

template <int dim>
Scratch<dim>::Scratch(
  const Scratch &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags())
{}

} // namespace RightHandSide

} // namespace Generic

} // namespace AssemblyData

} // namespace RMHD

template struct RMHD::AssemblyData::CopyBase<2>;
template struct RMHD::AssemblyData::CopyBase<3>;

template struct RMHD::AssemblyData::ScratchBase<2>;
template struct RMHD::AssemblyData::ScratchBase<3>;

template struct RMHD::AssemblyData::Generic::Matrix::Copy<2>;
template struct RMHD::AssemblyData::Generic::Matrix::Copy<3>;

template struct RMHD::AssemblyData::Generic::Matrix::MassStiffnessCopy<2>;
template struct RMHD::AssemblyData::Generic::Matrix::MassStiffnessCopy<3>;

template struct RMHD::AssemblyData::Generic::Matrix::Scratch<2>;
template struct RMHD::AssemblyData::Generic::Matrix::Scratch<3>;

template struct RMHD::AssemblyData::Generic::RightHandSide::Copy<2>;
template struct RMHD::AssemblyData::Generic::RightHandSide::Copy<3>;

template struct RMHD::AssemblyData::Generic::RightHandSide::Scratch<2>;
template struct RMHD::AssemblyData::Generic::RightHandSide::Scratch<3>;
