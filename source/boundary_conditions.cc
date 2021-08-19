#include <rotatingMHD/boundary_conditions.h>

#include <deal.II/base/conditional_ostream.h>

#include <boost/core/demangle.hpp>

#include <typeinfo>
#include <iostream>
#include <string>

namespace RMHD
{

using namespace dealii;

namespace Entities
{

namespace internal
{
  constexpr char header[] = "+---------+"
                            "------------+"
                            "------------------------------------------+";

  constexpr size_t column_width[3] = { 7, 10, 40};

  constexpr size_t line_width = 63;

  template<typename Stream, typename A>
  void add_line(Stream  &stream,
                const A &line)
  {
    stream << "| "
           << std::setw(line_width)
           << line
           << " |"
           << std::endl;
  }

  template<typename Stream, typename A, typename B, typename C>
  void add_line(Stream  &stream,
                const A &first_column,
                const B &second_column,
                const C &third_column)
  {
    stream << "| "
           << std::setw(column_width[0]) << first_column
           << " | "
           << std::setw(column_width[1]) << second_column
           << " | "
           << std::setw(column_width[2]) << third_column
          << " |"
           << std::endl;
  }

  template<typename A>
  std::string get_type_string(const A &object)
  {
    std::string typestr = boost::core::demangle(typeid(object).name());
    //                              123456789*123456789*
    std::size_t pos = typestr.find("RMHD::EquationData::");
    if (pos!=std::string::npos)
      typestr.erase(pos, 20);
    //                    123456789*123456789*
    pos = typestr.find("dealii::Functions::");
          if (pos!=std::string::npos)
            typestr.erase(pos, 19);
    //                    12345678
    pos = typestr.find("dealii::");
    if (pos!=std::string::npos)
      typestr.erase(pos, 8);
    //                    123456
    pos = typestr.find("RMHD::");
    if (pos!=std::string::npos)
      typestr.erase(pos, 6);

    return (typestr);
  }

  template<typename Stream>
  void add_header(Stream  &stream)
  {
    stream << std::left << header << std::endl;
  }

} // internal



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
BoundaryConditionsBase<dim>::BoundaryConditionsBase
(const Triangulation<dim> &tria,
 const unsigned int n_components)
:
triangulation(tria),
n_components(n_components),
flag_extract_boundary_ids(true),
flag_regularity_guaranteed(false),
flag_boundary_conditions_closed(false),
zero_function_ptr(std::make_shared<Functions::ZeroFunction<dim>>(n_components))
{}



template <int dim>
void BoundaryConditionsBase<dim>::add_function_to_bc_mapping
(typename BoundaryConditionsBase<dim>::BCMapping &bcs,
 const BCType                bc_type,
 const types::boundary_id    boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const unsigned int          n_function_components,
 const bool                  time_dependent_bc)
{
  AssertThrow(!closed(),
              ExcMessage("The boundary conditions have already been closed"));

  check_boundary_id(boundary_id);

  constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    bcs[boundary_id] = zero_function_ptr;
  else
  {
    std::stringstream message;
    message << "A boundary condition of the type "
            << internal::get_type_string(bc_type)
            << " needs to have a function with "
            << n_function_components
            << (n_function_components > 1? "components.": "component.");

    AssertThrow(function->n_components == n_function_components,
                ExcMessage(message.str().c_str()));
    bcs[boundary_id] = function;
  }

  if (time_dependent_bc)
    time_dependent_bcs_map.emplace(bc_type, boundary_id);
}



template <int dim>
void BoundaryConditionsBase<dim>::check_boundary_id
(const types::boundary_id boundary_id)
{
  extract_boundary_ids();

  AssertThrow(std::find(boundary_ids.begin(), boundary_ids.end(), boundary_id)
              != boundary_ids.end(),
              ExcMessage("The triangulation does not have a boundary"
                         " marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  AssertThrow(dirichlet_bcs.find(boundary_id) == dirichlet_bcs.end(),
              ExcMessage("A Dirichlet boundary condition was already set on "
                         "the boundary marked with the indicator " +
                         std::to_string(boundary_id) + "."));

  for (const auto &periodic_bc: periodic_bcs)
  {
    AssertThrow(boundary_id != periodic_bc.boundary_pair.first &&
                boundary_id != periodic_bc.boundary_pair.second,
                ExcMessage("A periodic boundary condition was already set on "
                           "the boundary marked with the indicator " +
                           std::to_string(boundary_id) + "."));
  }
}



template <int dim>
void BoundaryConditionsBase<dim>::clear()
{
  dirichlet_bcs.clear();
  periodic_bcs.clear();
  time_dependent_bcs_map.clear();

  boundary_ids.clear();
  constrained_boundaries.clear();

  flag_extract_boundary_ids = true;
  flag_regularity_guaranteed = false;
  flag_boundary_conditions_closed = false;
}



template <int dim>
void BoundaryConditionsBase<dim>::close()
{
  flag_boundary_conditions_closed = true;
}



template <int dim>
void BoundaryConditionsBase<dim>::copy
(const BoundaryConditionsBase<dim> &other)
{
  AssertDimension(n_components, other.n_components);

  dirichlet_bcs               = other.dirichlet_bcs;
  periodic_bcs                = other.periodic_bcs;
  time_dependent_bcs_map      = other.time_dependent_bcs_map;
  boundary_ids                = other.boundary_ids;
  constrained_boundaries      = other.constrained_boundaries;
  flag_regularity_guaranteed  = other.flag_regularity_guaranteed;
  flag_extract_boundary_ids   = other.flag_extract_boundary_ids;
  flag_boundary_conditions_closed = false;
}



template <int dim>
std::vector<types::boundary_id>
BoundaryConditionsBase<dim>::get_unconstrained_boundary_ids() const
{
  Assert(flag_extract_boundary_ids == false,
         ExcInternalError());

  // Initiates the returns vector
  std::vector<types::boundary_id> unconstrained_boundaries;

  // Loops through the triangulation's boundary indicators and adds them
  // to the return vector if they are not constrained.
  for (const auto &boundary_id : boundary_ids)
    if (std::find(constrained_boundaries.begin(),
                  constrained_boundaries.end(),
                  boundary_id) == constrained_boundaries.end())
      unconstrained_boundaries.push_back(boundary_id);

  return unconstrained_boundaries;
}



template <int dim>
void BoundaryConditionsBase<dim>::print_summary
(std::ostream      &stream,
 const std::string &/* name */) const
{
  if (dirichlet_bcs.size() != 0)
    for(auto &[key, value]: this->dirichlet_bcs)
      internal::add_line(stream,
                         key,
                         "Dirichlet",
                         internal::get_type_string(*value));

  if (periodic_bcs.size() != 0)
    for(const PeriodicBoundaryData<dim> &periodic_bc: this->periodic_bcs)
      internal::add_line(stream,
                         std::to_string(periodic_bc.boundary_pair.first) + ", " +
                         std::to_string(periodic_bc.boundary_pair.second),
                         "Periodic",
                         "---");

  const std::vector<types::boundary_id> unconstrained_boundary_ids
    = get_unconstrained_boundary_ids();

  if (unconstrained_boundary_ids.size() != 0)
  {
    std::stringstream strstream;
    strstream << "Unconstrained boundary ids: ";
    for(const auto &boundary_id: unconstrained_boundary_ids)
      strstream << boundary_id << ", ";

    strstream.seekp(-2, strstream.cur);
    strstream << " ";

    internal::add_line(stream, strstream.str().c_str());
  }
}



template <int dim>
void BoundaryConditionsBase<dim>::set_dirichlet_bc
(const types::boundary_id             boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool                           time_dependent)
{
  add_function_to_bc_mapping(dirichlet_bcs,
                             BCType::dirichlet,
                             boundary_id,
                             function,
                             n_components,
                             time_dependent);

  flag_regularity_guaranteed = true;
}



template <int dim>
void BoundaryConditionsBase<dim>::set_periodic_bc(
  const types::boundary_id  first_boundary,
  const types::boundary_id  second_boundary,
  const unsigned int        direction,
  const FullMatrix<double>  rotation_matrix,
  const Tensor<1,dim>       offset)
{
  AssertThrow(!this->closed(),
              ExcMessage("The boundary conditions have already been closed"));

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
void BoundaryConditionsBase<dim>::set_time(const double time)
{
  for (const auto &bc : dirichlet_bcs)
    bc.second->set_time(time);
}



template <int dim>
ScalarBoundaryConditions<dim>::ScalarBoundaryConditions
(const Triangulation<dim> &triangulation)
:
BoundaryConditionsBase<dim>(triangulation),
flag_datum_at_boundary(false)
{}



template <int dim>
void ScalarBoundaryConditions<dim>::check_boundary_id
(const types::boundary_id boundary_id)
{
  BoundaryConditionsBase<dim>::check_boundary_id(boundary_id);

  AssertThrow(neumann_bcs.find(boundary_id) == neumann_bcs.end(),
              ExcMessage("A Neumann boundary condition was already set on "
                         "the boundary marked with the indicator " +
                         std::to_string(boundary_id) + "."));

}


template <int dim>
void ScalarBoundaryConditions<dim>::clear()
{
  BoundaryConditionsBase<dim>::clear();

  neumann_bcs.clear();
  flag_datum_at_boundary = false;
}



template <int dim>
void ScalarBoundaryConditions<dim>::copy
(const BoundaryConditionsBase<dim> &other)
{
  BoundaryConditionsBase<dim>::copy(other);

  const ScalarBoundaryConditions<dim> & other_scalar_bc =
      dynamic_cast<const ScalarBoundaryConditions<dim> &>(other);

  neumann_bcs = other_scalar_bc.neumann_bcs;
  flag_datum_at_boundary = other_scalar_bc.flag_datum_at_boundary;
}



template <int dim>
void ScalarBoundaryConditions<dim>::print_summary
(std::ostream       &stream,
 const std::string  &name) const
{
  internal::add_header(stream);

  {
    std::stringstream strstream;
    strstream << "Boundary conditions of the "
              << name
              << " entity";

    internal::add_line(stream, strstream.str().c_str());
  }

  internal::add_header(stream);

  internal::add_line(stream,
                     "Bdy. id",
                     "   Type",
                     "             Function");

  if (neumann_bcs.size() != 0)
    for(const auto &[key, value]: neumann_bcs)
      internal::add_line(stream,
                         key,
                         "Neumann",
                         internal::get_type_string(*value));

  if (this->flag_datum_at_boundary)
    internal::add_line(stream, "A datum has been set at the boundary");

  BoundaryConditionsBase<dim>::print_summary(stream);

  internal::add_header(stream);
}



template <int dim>
void ScalarBoundaryConditions<dim>::set_datum_at_boundary()
{
  AssertThrow(!this->closed(),
              ExcMessage("The boundary conditions have already been closed"));

  AssertThrow(this->dirichlet_bcs.empty(),
              ExcMessage("Dirichlet boundary conditions were set. A datum is not needed."));

  this->flag_datum_at_boundary      = true;
  this->flag_regularity_guaranteed  = true;
}



template <int dim>
void ScalarBoundaryConditions<dim>::set_neumann_bc
(const types::boundary_id             boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool                           time_dependent)
{
  this->add_function_to_bc_mapping(neumann_bcs,
                                   BCType::neumann,
                                   boundary_id,
                                   function,
                                   1,
                                   time_dependent);
}



template <int dim>
void ScalarBoundaryConditions<dim>::set_time(const double time)
{
  BoundaryConditionsBase<dim>::set_time(time);
  for (const auto &bc : neumann_bcs)
    bc.second->set_time(time);
}



template <int dim>
VectorBoundaryConditions<dim>::VectorBoundaryConditions
(const Triangulation<dim> &triangulation)
:
BoundaryConditionsBase<dim>(triangulation, dim)
{}



template <int dim>
void VectorBoundaryConditions<dim>::check_boundary_id
(const types::boundary_id boundary_id)
{
  BoundaryConditionsBase<dim>::check_boundary_id(boundary_id);

  AssertThrow(neumann_bcs.find(boundary_id) == neumann_bcs.end(),
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
}



template <int dim>
void VectorBoundaryConditions<dim>::clear()
{
  BoundaryConditionsBase<dim>::clear();
  neumann_bcs.clear();
  normal_flux_bcs.clear();
  tangential_flux_bcs.clear();
}



template <int dim>
void VectorBoundaryConditions<dim>::copy
(const BoundaryConditionsBase<dim> &other)
{
  BoundaryConditionsBase<dim>::copy(other);

  const VectorBoundaryConditions<dim> & other_vector =
      dynamic_cast<const VectorBoundaryConditions<dim> &>(other);

  neumann_bcs         = other_vector.neumann_bcs;
  normal_flux_bcs     = other_vector.normal_flux_bcs;
  tangential_flux_bcs = other_vector.tangential_flux_bcs;
}



template <int dim>
void VectorBoundaryConditions<dim>::print_summary
(std::ostream      &stream,
 const std::string &name) const
{
  internal::add_header(stream);

  {
    std::stringstream strstream;
    strstream << "Boundary conditions of the "
              << name
              << " entity";

    internal::add_line(stream, strstream.str().c_str());
  }

  internal::add_header(stream);

  internal::add_line(stream,
                     "Bdy. id",
                     "   Type",
                     "             Function");

  if (neumann_bcs.size() != 0)
    for(const auto &[key, value]: this->neumann_bcs)
      internal::add_line(stream,
                         key,
                         "Neumann",
                         internal::get_type_string(*value));

  if (normal_flux_bcs.size() != 0)
    for(const auto &[key, value]: normal_flux_bcs)
      internal::add_line(stream,
                         key,
                         "Norm. flux",
                         internal::get_type_string(*value));

  if (tangential_flux_bcs.size() != 0)
    for(const auto &[key, value]: tangential_flux_bcs)
      internal::add_line(stream,
                         key,
                         "Tang. flux",
                         internal::get_type_string(*value)  );

  BoundaryConditionsBase<dim>::print_summary(stream);

  internal::add_header(stream);
}


template <int dim>
void VectorBoundaryConditions<dim>::set_neumann_bc(
  const types::boundary_id                      boundary_id,
  const std::shared_ptr<TensorFunction<1, dim>> &function,
  const bool                                    time_dependent)
{
  AssertThrow(!this->closed(),
              ExcMessage("The boundary conditions have already been closed"));

  check_boundary_id(boundary_id);

  this->constrained_boundaries.push_back(boundary_id);

  if (function.get() == nullptr)
    this->neumann_bcs[boundary_id] = zero_tensor_function_ptr;
  else
  {
    std::stringstream message;
    message << "Function of a Neumann boundary condition needs to have "
            << dim << " components.";

    this->neumann_bcs[boundary_id] = function;
  }

  if (time_dependent)
    this->time_dependent_bcs_map.emplace(BCType::neumann, boundary_id);
}

template <int dim>
void VectorBoundaryConditions<dim>::set_normal_flux_bc(
  const types::boundary_id             boundary_id,
  const std::shared_ptr<Function<dim>> &function,
  const bool                           time_dependent)
{
  this->add_function_to_bc_mapping(normal_flux_bcs,
                                   BCType::normal_flux,
                                   boundary_id,
                                   function,
                                   dim,
                                   time_dependent);
  // @attention I am not sure if this passes a fully constrained
  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_tangential_flux_bc
(const types::boundary_id             boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool                           time_dependent)
{
  this->add_function_to_bc_mapping(tangential_flux_bcs,
                                   BCType::tangential_flux,
                                   boundary_id,
                                   function,
                                   dim,
                                   time_dependent);
  // @attention I am not sure if this passes a fully constrained
  this->flag_regularity_guaranteed = true;
}

template <int dim>
void VectorBoundaryConditions<dim>::set_time(const double time)
{
  BoundaryConditionsBase<dim>::set_time(time);
  for (const auto &bc: neumann_bcs)
    bc.second->set_time(time);
  for (const auto &bc: normal_flux_bcs)
    bc.second->set_time(time);
  for (const auto &bc: tangential_flux_bcs)
    bc.second->set_time(time);
}

} // namespace Entities

} // namespace RMHD

// explicit instantiations
template class RMHD::Entities::PeriodicBoundaryData<2>;
template class RMHD::Entities::PeriodicBoundaryData<3>;

template class RMHD::Entities::BoundaryConditionsBase<2>;
template class RMHD::Entities::BoundaryConditionsBase<3>;

template class RMHD::Entities::ScalarBoundaryConditions<2>;
template class RMHD::Entities::ScalarBoundaryConditions<3>;

template class RMHD::Entities::VectorBoundaryConditions<2>;
template class RMHD::Entities::VectorBoundaryConditions<3>;
