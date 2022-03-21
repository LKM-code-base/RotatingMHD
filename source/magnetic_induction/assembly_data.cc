#include <rotatingMHD/assembly_data.h>

namespace RMHD
{



namespace AssemblyData
{



namespace MagneticInduction
{



namespace AdvectionMatrix
{



template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &magnetic_field_fe,
  const UpdateFlags         magnetic_field_update_flags,
  const FiniteElement<dim>  &velocity_fe,
  const UpdateFlags         velocity_update_flags)
:
ScratchBase<dim>(
  quadrature_formula,
  magnetic_field_fe),
magnetic_field_fe_values(
  mapping,
  magnetic_field_fe,
  quadrature_formula,
  magnetic_field_update_flags),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
velocity_gradients(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
magnetic_field_fe_values(
  data.magnetic_field_fe_values.get_mapping(),
  data.magnetic_field_fe_values.get_fe(),
  data.magnetic_field_fe_values.get_quadrature(),
  data.magnetic_field_fe_values.get_update_flags()),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
velocity_values(data.n_q_points),
old_velocity_values(data.n_q_points),
old_old_velocity_values(data.n_q_points),
velocity_gradients(data.n_q_points),
old_velocity_gradients(data.n_q_points),
old_old_velocity_gradients(data.n_q_points),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}



} // namespace AdvectionMatrix



namespace InitializationStepRHS
{



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &magnetic_field_fe,
 const UpdateFlags         magnetic_field_face_update_flags,
 const FiniteElement<dim>  &pseudo_pressure_fe,
 const UpdateFlags         pseudo_pressure_update_flags,
 const UpdateFlags         pseudo_pressure_face_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 pseudo_pressure_fe),
magnetic_field_fe_face_values(
  mapping,
  magnetic_field_fe,
  face_quadrature_formula,
  magnetic_field_face_update_flags),
pseudo_pressure_fe_values(
  mapping,
  pseudo_pressure_fe,
  quadrature_formula,
  pseudo_pressure_update_flags),
pseudo_pressure_fe_face_values(
  mapping,
  pseudo_pressure_fe,
  face_quadrature_formula,
  pseudo_pressure_face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
magnetic_field_face_laplacians(n_face_q_points),
normal_vectors(n_face_q_points),
supply_term_values(this->n_q_points),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
magnetic_field_fe_face_values(
  data.magnetic_field_fe_face_values.get_mapping(),
  data.magnetic_field_fe_face_values.get_fe(),
  data.magnetic_field_fe_face_values.get_quadrature(),
  data.magnetic_field_fe_face_values.get_update_flags()),
pseudo_pressure_fe_values(
  data.pseudo_pressure_fe_values.get_mapping(),
  data.pseudo_pressure_fe_values.get_fe(),
  data.pseudo_pressure_fe_values.get_quadrature(),
  data.pseudo_pressure_fe_values.get_update_flags()),
pseudo_pressure_fe_face_values(
  data.pseudo_pressure_fe_face_values.get_mapping(),
  data.pseudo_pressure_fe_face_values.get_fe(),
  data.pseudo_pressure_fe_face_values.get_quadrature(),
  data.pseudo_pressure_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
magnetic_field_face_laplacians(n_face_q_points),
supply_term_values(this->n_q_points),
face_phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



} // namespace InitializationStepRHS


namespace DiffusionStepRHS
{



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>         &mapping,
 const Quadrature<dim>      &quadrature_formula,
 const FiniteElement<dim>   &magnetic_field_fe,
 const UpdateFlags          magnetic_field_update_flags,
 const FiniteElement<dim>   &pseudo_pressure_fe,
 const UpdateFlags          pseudo_pressure_update_flags,
 const FiniteElement<dim>   &velocity_fe,
 const UpdateFlags          velocity_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 magnetic_field_fe),
magnetic_field_fe_values(
  mapping,
  magnetic_field_fe,
  quadrature_formula,
  magnetic_field_update_flags),
pseudo_pressure_fe_values(
  mapping,
  pseudo_pressure_fe,
  quadrature_formula,
  pseudo_pressure_update_flags),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_gradients(this->n_q_points),
old_old_magnetic_field_gradients(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
velocity_gradients(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
old_pseudo_pressure_gradients(this->n_q_points),
old_auxiliary_scalar_gradients(this->n_q_points),
old_old_auxiliary_scalar_gradients(this->n_q_points),
supply_term_values(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
magnetic_field_fe_values(
  data.magnetic_field_fe_values.get_mapping(),
  data.magnetic_field_fe_values.get_fe(),
  data.magnetic_field_fe_values.get_quadrature(),
  data.magnetic_field_fe_values.get_update_flags()),
pseudo_pressure_fe_values(
  data.pseudo_pressure_fe_values.get_mapping(),
  data.pseudo_pressure_fe_values.get_fe(),
  data.pseudo_pressure_fe_values.get_quadrature(),
  data.pseudo_pressure_fe_values.get_update_flags()),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_gradients(this->n_q_points),
old_old_magnetic_field_gradients(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
old_pseudo_pressure_gradients(this->n_q_points),
old_auxiliary_scalar_gradients(this->n_q_points),
old_old_auxiliary_scalar_gradients(this->n_q_points),
supply_term_values(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell)
{}



} // namespace DiffusionStepRHS



namespace ProjectionStepRHS
{



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &magnetic_field_fe,
 const UpdateFlags         magnetic_field_update_flags,
 const FiniteElement<dim>  &auxiliary_scalar_fe,
 const UpdateFlags         auxiliary_scalar_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 auxiliary_scalar_fe),
magnetic_field_fe_values(
  mapping,
  magnetic_field_fe,
  quadrature_formula,
  magnetic_field_update_flags),
auxiliary_scalar_fe_values(
  mapping,
  auxiliary_scalar_fe,
  quadrature_formula,
  auxiliary_scalar_update_flags),
magnetic_field_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
magnetic_field_fe_values(
  data.magnetic_field_fe_values.get_mapping(),
  data.magnetic_field_fe_values.get_fe(),
  data.magnetic_field_fe_values.get_quadrature(),
  data.magnetic_field_fe_values.get_update_flags()),
auxiliary_scalar_fe_values(
  data.auxiliary_scalar_fe_values.get_mapping(),
  data.auxiliary_scalar_fe_values.get_fe(),
  data.auxiliary_scalar_fe_values.get_quadrature(),
  data.auxiliary_scalar_fe_values.get_update_flags()),
magnetic_field_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}



} // namespace ProjectionStepRHS



} // namespace MagneticInduction



} // namespace AssemblyData



} // namespace RMHD


// Explicit
template struct RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<2>;
template struct RMHD::AssemblyData::MagneticInduction::AdvectionMatrix::Scratch<3>;

template struct RMHD::AssemblyData::MagneticInduction::InitializationStepRHS::Scratch<2>;
template struct RMHD::AssemblyData::MagneticInduction::InitializationStepRHS::Scratch<3>;

template struct RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<2>;
template struct RMHD::AssemblyData::MagneticInduction::DiffusionStepRHS::Scratch<3>;

template struct RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<2>;
template struct RMHD::AssemblyData::MagneticInduction::ProjectionStepRHS::Scratch<3>;

