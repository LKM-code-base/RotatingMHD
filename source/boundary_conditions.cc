#include <rotatingMHD/boundary_conditions.h>

namespace RMHD
{
  using namespace dealii;

namespace Entities
{

template <int dim>
void ScalarBoundaryConditions<dim>::set_periodic_bcs(
  const types::boundary_id  first_boundary,
  const types::boundary_id  second_boundary,
  const unsigned int        direction,
  const FullMatrix<double>  rotation_matrix,
  const Tensor<1,dim>       offset)
{
  check_boundary_id(first_boundary);
  check_boundary_id(second_boundary);

  PeriodicBoundaryData  periodic_bc(first_boundary,
                                    second_boundary,
                                    direction,
                                    rotation_matrix,
                                    offset);
  
  this->periodic_bcs.push_back(periodic_bc);
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->dirichlet_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == 1,
      ExcMessage("Scalar boundary function need to have a single component."));

    this->dirichlet_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::dirichlet, 
        boundary_id));
  }
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->neumann_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == 1,
      ExcMessage("Scalar boundary function need to have a single component."));
    
    this->neumann_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::neumann, 
        boundary_id));
  }
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_time(const double time)
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
void ScalarBoundaryConditions<dim>::copy(
  const ScalarBoundaryConditions<dim> &other)
{
  this->dirichlet_bcs           = other.dirichlet_bcs;
  this->neumann_bcs             = other.neumann_bcs;
  this->periodic_bcs            = other.periodic_bcs;
  this->time_dependent_bcs_map  = other.time_dependent_bcs_map;
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
void VectorBoundaryConditions<dim>::set_periodic_bcs(
  const types::boundary_id  first_boundary,
  const types::boundary_id  second_boundary,
  const unsigned int        direction,
  const FullMatrix<double>  rotation_matrix,
  const Tensor<1,dim>       offset)
{
  check_boundary_id(first_boundary);
  check_boundary_id(second_boundary);

  PeriodicBoundaryData  periodic_bc(first_boundary,
                                    second_boundary,
                                    direction,
                                    rotation_matrix,
                                    offset);
  
  this->periodic_bcs.push_back(periodic_bc);
}

template <int dim>
void VectorBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->dirichlet_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Scalar boundary function need to have a single component."));

    this->dirichlet_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::dirichlet, 
        boundary_id));
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    this->neumann_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    this->neumann_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::neumann, 
        boundary_id));
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_normal_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    normal_flux_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    normal_flux_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::normal_flux, 
        boundary_id));
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_tangential_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  if (function.get() == 0)
    tangential_flux_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == dim,
      ExcMessage("Neumann boundary function needs to have dim components."));
    
    tangential_flux_bcs[boundary_id] = function;
  }

  if (time_dependent)
  {
    typedef std::multimap<BCType, types::boundary_id> MapType;

    this->time_dependent_bcs_map.insert(
      MapType::value_type(
        BCType::tangential_flux, 
        boundary_id));
  }
}

template <int dim>
void VectorBoundaryConditions<dim>::set_time(const double time)
{
  for (const auto &dirichlet_bc : this->dirichlet_bcs)
    dirichlet_bc.second->set_time(time);
  for (const auto &neumann_bc : this->neumann_bcs)
    neumann_bc.second->set_time(time);
  for (const auto &normal_flux_bc : this->normal_flux_bcs)
    normal_flux_bc.second->set_time(time);
  for (const auto &tangential_flux_bc : this->tangential_flux_bcs)
    tangential_flux_bc.second->set_time(time);
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
void VectorBoundaryConditions<dim>::copy(
  const VectorBoundaryConditions<dim> &other)
{
  this->dirichlet_bcs           = other.dirichlet_bcs;
  this->neumann_bcs             = other.neumann_bcs;
  this->periodic_bcs            = other.periodic_bcs;
  this->time_dependent_bcs_map  = other.time_dependent_bcs_map;
  normal_flux_bcs               = other.normal_flux_bcs;
  tangential_flux_bcs           = other.tangential_flux_bcs;
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