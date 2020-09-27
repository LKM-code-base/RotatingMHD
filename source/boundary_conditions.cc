#include <rotatingMHD/boundary_conditions.h>

namespace RMHD
{
  using namespace dealii;

namespace Entities
{

template <int dim>
void ScalarBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->dirichlet_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>());
  else
  {
    AssertThrow(
      function->n_components == 1,
      ExcMessage("Scalar boundary function need to have a single component."));

    this->dirichlet_bcs[boundary_id] = function;
  }
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->neumann_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
  else
  {
    AssertThrow(
      function->n_components == 1,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    this->neumann_bcs[boundary_id] = function;
  }
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_time(const double &time)
{
  for (const auto &dirichlet_bc : this->dirichlet_bcs)
    dirichlet_bc.second->set_time(time);
  for (const auto &neumann_bc : this->neumann_bcs)
    neumann_bc.second->set_time(time);
}

template <int dim>
void ScalarBoundaryConditions<dim>::clear()
{
  this->dirichlet_bcs.clear();
  this->neumann_bcs.clear();
  this->periodic_bcs.clear();
}

template <int dim>
void ScalarBoundaryConditions<dim>::check_boundary_id(
  types::boundary_id boundary_id) const
{
  AssertThrow(this->dirichlet_bcs.find(boundary_id) == this->dirichlet_bcs.end(),
                ExcMessage("A Dirichlet boundary condition was already set on "
                           "the given boundary."));

  AssertThrow(this->neumann_bcs.find(boundary_id) == this->neumann_bcs.end(),
                ExcMessage("A Neumann boundary condition was already set on "
                           "the given boundary."))
  
  for (auto &periodic_bc : this->periodic_bcs)
  {
    AssertThrow( boundary_id != periodic_bc.boundary_pair.first &&
                 boundary_id != periodic_bc.boundary_pair.second,
                ExcMessage("A periodic boundary condition was already set on "
                           "the given boundary."));
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->dirichlet_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Scalar boundary function need to have a single component."));

    this->dirichlet_bcs[boundary_id] = function;
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->neumann_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    this->neumann_bcs[boundary_id] = function;
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_normal_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    normal_flux_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    normal_flux_bcs[boundary_id] = function;
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_tangential_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    tangential_flux_bcs[boundary_id] = std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim));
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    tangential_flux_bcs[boundary_id] = function;
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::clear()
{
  this->dirichlet_bcs.clear();
  this->neumann_bcs.clear();
  this->periodic_bcs.clear();
  normal_flux_bcs.clear();
  tangential_flux_bcs.clear();
}

template <int dim>
void VectorBoundaryConditions<dim>::check_boundary_id(
  types::boundary_id boundary_id) const
{
  AssertThrow(this->dirichlet_bcs.find(boundary_id) == this->dirichlet_bcs.end(),
                ExcMessage("A Dirichlet boundary condition was already set on "
                           "the given boundary."));

  AssertThrow(this->neumann_bcs.find(boundary_id) == this->neumann_bcs.end(),
                ExcMessage("A Neumann boundary condition was already set on "
                           "the given boundary."))

  AssertThrow(normal_flux_bcs.find(boundary_id) == normal_flux_bcs.end(),
                ExcMessage("A normal flux boundary condition was already set on "
                           "the given boundary."))

  AssertThrow(tangential_flux_bcs.find(boundary_id) == tangential_flux_bcs.end(),
                ExcMessage("A tangential flux boundary condition was already set on "
                           "the given boundary."))
  for (auto &periodic_bc : this->periodic_bcs)
  {
    AssertThrow( boundary_id != periodic_bc.boundary_pair.first &&
                 boundary_id != periodic_bc.boundary_pair.second,
                ExcMessage("A periodic boundary condition was already set on "
                           "the given boundary."));
  }
}

} // namespace Entities

} // namespace RMHD

template struct RMHD::Entities::PeriodicBoundaryData<2>;
template struct RMHD::Entities::PeriodicBoundaryData<3>;
template struct RMHD::Entities::BoundaryConditionsBase<2>;
template struct RMHD::Entities::BoundaryConditionsBase<3>;
template struct RMHD::Entities::ScalarBoundaryConditions<2>;
template struct RMHD::Entities::ScalarBoundaryConditions<3>;
template struct RMHD::Entities::VectorBoundaryConditions<2>;
template struct RMHD::Entities::VectorBoundaryConditions<3>;