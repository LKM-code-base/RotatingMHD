#include <rotatingMHD/boundary_conditions.h>

namespace RMHD
{

using namespace dealii;

namespace Entities
{

template <int dim>
PeriodicBoundaryData<dim>::PeriodicBoundaryData
(const types::boundary_id first_boundary,
 const types::boundary_id second_boundary,
 const unsigned int       direction,
 const FullMatrix<double> rotation_matrix,
 const Tensor<1,dim>      offset)
:
boundary_pair(std::make_pair(first_boundary, second_boundary)),
direction(direction),
rotation_matrix(rotation_matrix),
offset(offset)
{}

template <int dim>
BoundaryConditionsBase<dim>::BoundaryConditionsBase(
  const parallel::distributed::Triangulation<dim> &triangulation)
:
triangulation(triangulation),
flag_extract_boundary_ids(true),
flag_datum_at_boundary(false),
flag_regularity_guaranteed(false)
{}

template <int dim>
std::vector<types::boundary_id> BoundaryConditionsBase<dim>::get_unconstrained_boundary_ids()
{
  // Extracts the boundary indicators from the triangulation
  if (this->flag_extract_boundary_ids)
  {
    this->boundary_ids              = this->triangulation.get_boundary_ids();
    this->flag_extract_boundary_ids = false;
  }

  // Initiates the returns vector
  std::vector<types::boundary_id> unconstrained_boundaries;

  // Loops through the triangulation's boundary indicators and adds them
  // to the return vector if they are not constrained.
  for (const auto &boundary_id : this->boundary_ids)
    if (std::find(this->constrained_boundaries.begin(),
                  this->constrained_boundaries.end(),
                  boundary_id) == this->constrained_boundaries.end())
      unconstrained_boundaries.push_back(boundary_id);

  return unconstrained_boundaries;
}


template <int dim>
ScalarBoundaryConditions<dim>::ScalarBoundaryConditions(
  const parallel::distributed::Triangulation<dim> &triangulation)
:
BoundaryConditionsBase<dim>(triangulation)
{}


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

  this->constrained_boundaries.push_back(first_boundary);
  this->constrained_boundaries.push_back(second_boundary);

  this->periodic_bcs.emplace_back(first_boundary,
                                  second_boundary,
                                  direction,
                                  rotation_matrix,
                                  offset);

  this->flag_regularity_guaranteed = true;
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    this->dirichlet_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(function->n_components == 1,
                ExcMessage("Function of a Dirichlet boundary condition needs to have a single component."));

    this->dirichlet_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::dirichlet, boundary_id);

  this->flag_regularity_guaranteed = true;
}

template <int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    this->neumann_bcs[boundary_id] = zero_function_ptr;
  else
  {
    AssertThrow(
      function->n_components == 1,
      ExcMessage("Function of a Neumann boundary condition needs to have a single component."));

    this->neumann_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::neumann, boundary_id);

  this->flag_regularity_guaranteed = true;
}



template <int dim>
void ScalarBoundaryConditions<dim>::set_datum_at_boundary()
{
  this->flag_datum_at_boundary      = true;
  this->flag_regularity_guaranteed  = true;
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

  this->time_dependent_bcs_map.clear();
}

template <int dim>
void ScalarBoundaryConditions<dim>::copy
(const ScalarBoundaryConditions<dim> &other)
{
  this->dirichlet_bcs               = other.dirichlet_bcs;
  this->neumann_bcs                 = other.neumann_bcs;
  this->periodic_bcs                = other.periodic_bcs;
  this->time_dependent_bcs_map      = other.time_dependent_bcs_map;
  this->flag_regularity_guaranteed  = other.flag_regularity_guaranteed;
}

template <int dim>
void ScalarBoundaryConditions<dim>::check_boundary_id
(const types::boundary_id boundary_id)
{
  if (this->flag_extract_boundary_ids)
  {
    this->boundary_ids              = this->triangulation.get_boundary_ids();
    this->flag_extract_boundary_ids = false;
  }

  AssertThrow(std::find(this->boundary_ids.begin(),
                        this->boundary_ids.end(),
                        boundary_id) != this->boundary_ids.end(),
              ExcMessage("The triangulation does not have a boundary"
                         " marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  AssertThrow(this->dirichlet_bcs.find(boundary_id) == this->dirichlet_bcs.end(),
              ExcMessage("A Dirichlet boundary condition was already set on "
                         "the boundary marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  AssertThrow(this->neumann_bcs.find(boundary_id) == this->neumann_bcs.end(),
              ExcMessage("A Neumann boundary condition was already set on "
                         "the boundary marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  for (const auto &periodic_bc : this->periodic_bcs)
  {
    AssertThrow(boundary_id != periodic_bc.boundary_pair.first &&
                boundary_id != periodic_bc.boundary_pair.second,
                ExcMessage("A periodic boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));
  }
}


template <int dim>
VectorBoundaryConditions<dim>::VectorBoundaryConditions(
  const parallel::distributed::Triangulation<dim> &triangulation)
:
BoundaryConditionsBase<dim>(triangulation)
{}

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

  this->constrained_boundaries.push_back(first_boundary);
  this->constrained_boundaries.push_back(second_boundary);

  this->periodic_bcs.emplace_back(first_boundary,
                                  second_boundary,
                                  direction,
                                  rotation_matrix,
                                  offset);

  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_dirichlet_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    this->dirichlet_bcs[boundary_id] = zero_function_ptr;
  else
  {
    std::stringstream message;
    message << "Function of a Dirichlet boundary condition needs to have "
            << dim << " components.";

    AssertThrow(function->n_components == dim,
                ExcMessage(message.str()));

    this->dirichlet_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::dirichlet, boundary_id);

  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_neumann_bcs(
  const types::boundary_id                      boundary_id,
  const std::shared_ptr<TensorFunction<1, dim>> &function,
  const bool                                    time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    this->neumann_bcs[boundary_id] = zero_vector;
  else
  {
    std::stringstream message;
    message << "Function of a Neumann boundary condition needs to have "
            << dim << " components.";

    this->neumann_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::neumann, boundary_id);

  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_normal_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    normal_flux_bcs[boundary_id] = zero_function_ptr;
  else
  {
    std::stringstream message;
    message << "Function of a normal flux boundary condition needs to have "
            << dim << " components.";

    AssertThrow(function->n_components == dim,
                ExcMessage(message.str()));

    normal_flux_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::normal_flux, boundary_id);

  /*! @attention I am not sure if this passes a fully constrained */
  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_tangential_flux_bcs(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == 0)
    tangential_flux_bcs[boundary_id] = zero_function_ptr;
  else
  {
    std::stringstream message;
    message << "Function of a tangential flux boundary condition needs to have "
            << dim << " components.";

    AssertThrow(function->n_components == dim,
                ExcMessage(message.str()));

    tangential_flux_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::tangential_flux, boundary_id);

  /*! @attention I am not sure if this passes a fully constrained */
  this->flag_regularity_guaranteed = true;
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
  this->dirichlet_bcs               = other.dirichlet_bcs;
  this->neumann_bcs                 = other.neumann_bcs;
  this->periodic_bcs                = other.periodic_bcs;
  this->time_dependent_bcs_map      = other.time_dependent_bcs_map;
  normal_flux_bcs                   = other.normal_flux_bcs;
  tangential_flux_bcs               = other.tangential_flux_bcs;
  this->flag_regularity_guaranteed  = other.flag_regularity_guaranteed;
}

template <int dim>
void VectorBoundaryConditions<dim>::check_boundary_id(
  const types::boundary_id boundary_id)
{
  if (this->flag_extract_boundary_ids)
  {
    this->boundary_ids              = this->triangulation.get_boundary_ids();
    this->flag_extract_boundary_ids = false;
  }

  AssertThrow(std::find(this->boundary_ids.begin(),
                        this->boundary_ids.end(),
                        boundary_id) != this->boundary_ids.end(),
              ExcMessage("The triangulation does not have a boundary"
                         " marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  AssertThrow(this->dirichlet_bcs.find(boundary_id) == this->dirichlet_bcs.end(),
              ExcMessage("A Dirichlet boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));

  AssertThrow(this->neumann_bcs.find(boundary_id) == this->neumann_bcs.end(),
              ExcMessage("A Neumann boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));

  AssertThrow(normal_flux_bcs.find(boundary_id) == normal_flux_bcs.end(),
              ExcMessage("A normal flux boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));

  AssertThrow(tangential_flux_bcs.find(boundary_id) == tangential_flux_bcs.end(),
              ExcMessage("A tangential flux boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));

  for (const auto &periodic_bc : this->periodic_bcs)
    AssertThrow(boundary_id != periodic_bc.boundary_pair.first &&
                boundary_id != periodic_bc.boundary_pair.second,
                ExcMessage("A periodic boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));
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
