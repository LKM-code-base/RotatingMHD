#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/finite_element_field.h>

#include <algorithm>
#include <type_traits>

namespace RMHD
{

using namespace dealii;

namespace Entities
{

template <int dim, typename VectorType>
FE_FieldBase<dim, VectorType>::FE_FieldBase
(const Triangulation<dim>  &triangulation,
 const std::string         &name)
:
name(name),
flag_child_entity(false),
flag_setup_dofs(true),
triangulation(triangulation),
dof_handler(std::make_shared<DoFHandler<dim>>())
{}



template <int dim, typename VectorType>
FE_FieldBase<dim, VectorType>::FE_FieldBase
(const FE_FieldBase<dim, VectorType>  &entity,
 const std::string      &new_name)
:
name(new_name),
flag_child_entity(true),
flag_setup_dofs(entity.flag_setup_dofs),
triangulation(entity.get_triangulation()),
dof_handler(entity.dof_handler),
finite_element(entity.finite_element)
{}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::apply_boundary_conditions(const bool check_regularity)
{
  if (check_regularity)
    AssertThrow(boundary_conditions->regularity_guaranteed(),
                ExcMessage("No boundary conditions were set for the \""
                            + this->name + "\" entity"));

  AssertThrow(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  AssertThrow(boundary_conditions->closed(),
              ExcMessage("The boundary conditions have not been closed."));

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  constraints.merge(hanging_node_constraints);

  apply_dirichlet_constraints();

  apply_periodicity_constraints();
}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::apply_periodicity_constraints()
{
  if (!boundary_conditions->periodic_bcs.empty())
  {
     std::vector<GridTools::
                 PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
     periodicity_vector;

     for (auto const &periodic_bc : this->boundary_conditions->periodic_bcs)
       GridTools::collect_periodic_faces(*dof_handler,
                                         periodic_bc.boundary_pair.first,
                                         periodic_bc.boundary_pair.second,
                                         periodic_bc.direction,
                                         periodicity_vector);

     DoFTools::make_periodicity_constraints<DoFHandler<dim>>(periodicity_vector,
                                                             constraints);
  }
}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::apply_dirichlet_constraints()
{
  if (!boundary_conditions->dirichlet_bcs.empty())
  {
    std::map<types::boundary_id, const Function<dim> *> function_map;

    for (const auto &[boundary_id, function]: this->boundary_conditions->dirichlet_bcs)
      function_map[boundary_id] = function.get();

    VectorTools::interpolate_boundary_values(*dof_handler,
                                             function_map,
                                             constraints);
  }
}


template <>
void FE_FieldBase<2, Vector<double>>::clear()
{
  solution.reinit(0);
  old_solution.reinit(0);
  old_old_solution.reinit(0);
  distributed_vector.reinit(0);

  hanging_node_constraints.clear();
  constraints.clear();

  locally_owned_dofs.clear();
  locally_relevant_dofs.clear();

  if (!flag_child_entity)
    dof_handler->clear();

  boundary_conditions->clear();

  flag_setup_dofs = true;
}



template <>
void FE_FieldBase<3, Vector<double>>::clear()
{
  solution.reinit(0);
  old_solution.reinit(0);
  old_old_solution.reinit(0);
  distributed_vector.reinit(0);

  hanging_node_constraints.clear();
  constraints.clear();

  locally_owned_dofs.clear();
  locally_relevant_dofs.clear();

  if (!flag_child_entity)
    dof_handler->clear();

  boundary_conditions->clear();

  flag_setup_dofs = true;
}


template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::clear()
{
  solution.clear();
  old_solution.clear();
  old_old_solution.clear();
  distributed_vector.clear();

  hanging_node_constraints.clear();
  constraints.clear();

  locally_owned_dofs.clear();
  locally_relevant_dofs.clear();

  if (!flag_child_entity)
    dof_handler->clear();

  boundary_conditions->clear();

  flag_setup_dofs = true;
}




template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::clear_boundary_conditions()
{
  boundary_conditions->clear();

  this->constraints.clear();
}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::close_boundary_conditions(const bool print_summary)
{
  boundary_conditions->close();
  if (!print_summary)
    return;

  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);

  if (tria_ptr != nullptr)
  {
    if (Utilities::MPI::this_mpi_process(tria_ptr->get_communicator()) == 0)
      boundary_conditions->print_summary(std::cout, this->name);
  }
  else
    boundary_conditions->print_summary(std::cout, this->name);
}



template <int dim, typename VectorType>
template <typename T>
void FE_FieldBase<dim, VectorType>::send_point_data
(T                 &point_value,
 const Point<dim>  &point,
 const bool         point_found) const
{
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);

  if (tria_ptr == nullptr)
    return;

  // ensure that at least one processor found things
  const MPI_Comm &mpi_communicator(tria_ptr->get_communicator());
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, mpi_communicator);
  AssertThrow(n_procs > 0,
              ExcMessage("While trying to evaluate the solution at point " +
                         Utilities::to_string(point[0]) + ", " +
                         Utilities::to_string(point[1]) +
                         (dim == 3 ?
                             ", " + Utilities::to_string(point[2]) :
                             "") +
                         "), " +
                         "no processors reported that the point lies inside the " +
                         "set of cells they own. Are you trying to evaluate the " +
                         "solution at a point that lies outside of the domain?"));

  // Reduce all collected values into local Vector
  Utilities::MPI::sum(point_value,
                      mpi_communicator,
                      point_value);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    point_value /= n_procs;
}



template <int dim, typename VectorType>
template <typename T>
void FE_FieldBase<dim, VectorType>::send_point_data_vector
(std::vector<T>    &point_data_vector,
 const Point<dim>  &point,
 const bool         point_found) const
{
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);

  if (tria_ptr == nullptr)
    return;

  // ensure that at least one processor found things
  const MPI_Comm &mpi_communicator(tria_ptr->get_communicator());
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, mpi_communicator);
  AssertThrow(n_procs > 0,
              ExcMessage("While trying to evaluate the solution at point " +
                         Utilities::to_string(point[0]) + ", " +
                         Utilities::to_string(point[1]) +
                         (dim == 3 ?
                             ", " + Utilities::to_string(point[2]) :
                             "") +
                         "), " +
                         "no processors reported that the point lies inside the " +
                         "set of cells they own. Are you trying to evaluate the " +
                         "solution at a point that lies outside of the domain?"));

  // Reduce all collected values into local Vector
  for (auto &data : point_data_vector)
    data = Utilities::MPI::sum(data, mpi_communicator);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    for (auto &data: point_data_vector)
      data /= n_procs;
}



template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::setup_dofs()
{
  if (flag_child_entity)
  {
    AssertThrow(finite_element != nullptr,
                ExcMessage("The shared pointer to the FiniteElement of the base "
                           "entity is not setup correctly."));
    AssertThrow(dof_handler != nullptr,
                ExcMessage("The shared pointer to the DoFHandler of the base "
                           "entity is not setup correctly."));
    AssertThrow(dof_handler->has_active_dofs(),
                ExcMessage("The DoFHandler of the base entity does not have any "
                           "active degrees of freedom."));
  }
  else
  {
    dof_handler->initialize(triangulation, *finite_element);
    DoFRenumbering::Cuthill_McKee(*dof_handler);
  }

  locally_owned_dofs = dof_handler->locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                          locally_relevant_dofs);
  // Fill hanging node constraints
  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(*dof_handler,
                                            hanging_node_constraints);
  }
  hanging_node_constraints.close();

  // Modify flag because the dofs are setup
  flag_setup_dofs = false;
}



template <>
void FE_FieldBase<2, Vector<double>>::setup_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  const typename types::global_cell_index n_dofs{dof_handler->n_dofs()};
  solution.reinit(n_dofs);
  old_solution.reinit(n_dofs);
  old_old_solution.reinit(n_dofs);
  distributed_vector.reinit(n_dofs);
}



template <>
void FE_FieldBase<3, Vector<double>>::setup_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  const typename types::global_cell_index n_dofs{dof_handler->n_dofs()};
  solution.reinit(n_dofs);
  old_solution.reinit(n_dofs);
  old_old_solution.reinit(n_dofs);
  distributed_vector.reinit(n_dofs);
}



template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::setup_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);
  AssertThrow(tria_ptr == nullptr, ExcInternalError());

  const MPI_Comm &mpi_communicator(tria_ptr->get_communicator());
  #ifdef USE_PETSC_LA
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
  #else
    solution.reinit(locally_relevant_dofs,
                    mpi_communicator);
  #endif
  old_solution.reinit(solution);
  old_old_solution.reinit(solution);

  #ifdef USE_PETSC_LA
    distributed_vector.reinit(locally_owned_dofs,
                              mpi_communicator);
  #else
    distributed_vector.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator,
                              true);
  #endif
}



template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::set_dirichlet_boundary_condition
(const types::boundary_id boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool time_dependent_bc)
{
  boundary_conditions->set_dirichlet_bc(boundary_id, function, time_dependent_bc);
}



template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::set_periodic_boundary_condition
(const types::boundary_id first_boundary_id,
 const types::boundary_id second_boundary_id,
 const unsigned int       direction)
{
  boundary_conditions->set_periodic_bc(first_boundary_id,
                                      second_boundary_id,
                                      direction);
}


template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::set_solution_vectors_to_zero()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  solution          = 0.;
  old_solution      = 0.;
  old_old_solution  = 0.;
}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::update_boundary_conditions()
{
  if (boundary_conditions->time_dependent_bcs_map.empty())
    return;

  using const_iterator = typename std::multimap<BCType, types::boundary_id>::const_iterator;

  AffineConstraints<value_type>   tmp_constraints;

  tmp_constraints.clear();
  tmp_constraints.reinit(this->locally_relevant_dofs);

  if (boundary_conditions->time_dependent_bcs_map.find(BCType::dirichlet)
      != boundary_conditions->time_dependent_bcs_map.end())
  {
    typename std::map<types::boundary_id, const Function<dim> *> function_map;

    // Extract a pair of iterators representing the upper and
    // lower limit of the iterator range of all boundary ids on
    // which a time dependent boundary condition of the type BCType::::dirichlet
    // was specified.
    const std::pair<const_iterator, const_iterator>
    iterator_pair =
        this->boundary_conditions->time_dependent_bcs_map.equal_range(BCType::dirichlet);

    // Iterate over all necessary boundary conditions to populate the function
    // map to constrain the AffineConstraints instance.
    const_iterator it = iterator_pair.first, endit = iterator_pair.second;
    for (; it != endit; ++it)
      function_map[it->second] = this->boundary_conditions->dirichlet_bcs[it->second].get();

    VectorTools::interpolate_boundary_values(*dof_handler,
                                             function_map,
                                             tmp_constraints);
  }

  tmp_constraints.close();

  this->constraints.merge(tmp_constraints,
                          AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
}

template <int dim, typename VectorType>
void FE_FieldBase<dim, VectorType>::update_solution_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  old_old_solution  = old_solution;
  old_solution      = solution;
}



template <int dim, typename VectorType>
FE_VectorField<dim, VectorType>::FE_VectorField
(const unsigned int         fe_degree,
 const Triangulation<dim>  &triangulation,
 const std::string         &name)
:
FE_FieldBase<dim, VectorType>(triangulation, name)
{
  this->finite_element = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);
  this->boundary_conditions = new VectorBoundaryConditions<dim>(triangulation);
}



template <int dim, typename VectorType>
FE_VectorField<dim, VectorType>::FE_VectorField
(const FE_VectorField<dim, VectorType>  &entity,
 const std::string        &new_name)
:
FE_FieldBase<dim, VectorType>(entity, new_name)
{
  this->boundary_conditions = new VectorBoundaryConditions<dim>(this->triangulation);
}


template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::apply_boundary_conditions(const bool check_regularity)
{
  FE_FieldBase<dim, VectorType>::apply_boundary_conditions(check_regularity);

  const VectorBoundaryConditions<dim>* const vector_boundary_conditions =
      static_cast<const VectorBoundaryConditions<dim> *>(this->boundary_conditions);

  if (!vector_boundary_conditions->normal_flux_bcs.empty())
  {
    std::map<types::boundary_id, const Function<dim> *> function_map;

    std::set<types::boundary_id>  boundary_id_set;

    for (auto const &[boundary_id, function] : vector_boundary_conditions->normal_flux_bcs)
    {
      function_map[boundary_id] = function.get();
      boundary_id_set.insert(boundary_id);
    }

    VectorTools::compute_nonzero_normal_flux_constraints(*(this->dof_handler),
                                                         0,
                                                         boundary_id_set,
                                                         function_map,
                                                         this->constraints);
  }

  if (!vector_boundary_conditions->tangential_flux_bcs.empty())
  {
    std::map<types::boundary_id, const Function<dim> *> function_map;

    std::set<types::boundary_id>  boundary_id_set;

    for (auto const &[boundary_id, function] : vector_boundary_conditions->tangential_flux_bcs)
    {
      function_map[boundary_id] = function.get();
      boundary_id_set.insert(boundary_id);
    }

    VectorTools::compute_nonzero_tangential_flux_constraints(*(this->dof_handler),
                                                             0,
                                                             boundary_id_set,
                                                             function_map,
                                                             this->constraints);
  }

  /*! @todo Implement a datum for vector valued problem*/

  this->constraints.close();
}


template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::setup_dofs()
{
  if (this->flag_child_entity)
  {
    AssertThrow(this->dof_handler != nullptr,
                ExcMessage("The shared pointer to the DoFHandler of the base "
                           "entity is not setup correctly."));
    AssertThrow(this->dof_handler->has_active_dofs(),
                ExcMessage("The DoFHandler of the base entity does not have any "
                           "active degrees of freedom."));
  }
  else
  {
    this->dof_handler->initialize(this->triangulation,
                                  this->get_finite_element());
//    DoFRenumbering::Cuthill_McKee(*(this->dof_handler));
//    DoFRenumbering::block_wise(*(this->dof_handler));
  }

  this->locally_owned_dofs = this->dof_handler->locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(*(this->dof_handler),
                                          this->locally_relevant_dofs);
  // Fill hanging node constraints
  this->hanging_node_constraints.clear();
  {
    this->hanging_node_constraints.reinit(this->locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(*(this->dof_handler),
                                            this->hanging_node_constraints);
  }
  this->hanging_node_constraints.close();

  // Modify flag because the dofs are setup
  this->flag_setup_dofs = false;
}



template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::set_neumann_boundary_condition
(const types::boundary_id boundary_id,
 const std::shared_ptr<TensorFunction<1, dim>> &function,
 const bool time_dependent_bc)
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));
  static_cast<VectorBoundaryConditions<dim>*>(this->boundary_conditions)
      ->set_neumann_bc(boundary_id, function, time_dependent_bc);
}



template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::set_tangential_component_boundary_condition
(const types::boundary_id boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool time_dependent_bc)
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));
  static_cast<VectorBoundaryConditions<dim>*>(this->boundary_conditions)
      ->set_tangential_flux_bc(boundary_id, function, time_dependent_bc);
}


template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::set_normal_component_boundary_condition
(const types::boundary_id boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool time_dependent_bc)
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));
  static_cast<VectorBoundaryConditions<dim>*>(this->boundary_conditions)
      ->set_normal_flux_bc(boundary_id, function, time_dependent_bc);
}



template <int dim, typename VectorType>
Tensor<1, dim> FE_VectorField<dim, VectorType>::point_value
(const Point<dim>   &point,
 const Mapping<dim> &external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  Vector<typename FE_FieldBase<dim, VectorType>::value_type>  point_value(dim);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    VectorTools::point_value(external_mapping,
                             *(this->dof_handler),
                             this->solution,
                             point,
                             point_value);
    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  this->send_point_data(point_value, point, point_found);

  Tensor<1, dim> point_value_tensor;
  for (unsigned i=0; i<dim; ++i)
    point_value_tensor[i] = point_value[i];

  return (point_value_tensor);
}



template <int dim, typename VectorType>
Tensor<2, dim> FE_VectorField<dim, VectorType>::point_gradient
(const Point<dim>   &point,
 const Mapping<dim> &external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  std::vector<Tensor<1, dim>> point_gradient_rows(dim);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    VectorTools::point_gradient(external_mapping,
                                *(this->dof_handler),
                                this->solution,
                                point);
    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  this->send_point_data_vector(point_gradient_rows, point, point_found);

  Tensor<2, dim> point_gradient;
  for (unsigned int d=0; d<dim; ++d)
    point_gradient[d] = point_gradient_rows[d];

  return (point_gradient);
}



template <int dim, typename VectorType>
void FE_VectorField<dim, VectorType>::update_boundary_conditions()
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));

  if (this->boundary_conditions->time_dependent_bcs_map.empty())
    return;

  using FunctionMap = typename std::map<types::boundary_id, const Function<dim> *>;
  using const_iterator = typename std::multimap<BCType, types::boundary_id>::const_iterator;

  AffineConstraints<LinearAlgebra::MPI::Vector::value_type>   tmp_constraints;

  VectorBoundaryConditions<dim>* const vector_boundary_conditions =
      static_cast<VectorBoundaryConditions<dim> *>(this->boundary_conditions);

  tmp_constraints.clear();
  tmp_constraints.reinit(this->locally_relevant_dofs);

  if (vector_boundary_conditions->time_dependent_bcs_map.find(BCType::dirichlet)
      != vector_boundary_conditions->time_dependent_bcs_map.end())
  {
    FunctionMap   function_map;

    // Extract a pair of iterators representing the upper and
    // lower limit of the iterator range of all boundary ids on
    // which a time dependent boundary condition of the type BCType::::dirichlet
    // was specified.
    const std::pair<const_iterator, const_iterator>
    iterator_pair =
        vector_boundary_conditions->time_dependent_bcs_map.equal_range(BCType::dirichlet);

    // Iterate over all necessary boundary conditions to populate the function
    // map to constrain the AffineConstraints instance.
    const_iterator it = iterator_pair.first, endit = iterator_pair.second;
    for (; it != endit; ++it)
      function_map[it->second] = vector_boundary_conditions->dirichlet_bcs[it->second].get();

    VectorTools::interpolate_boundary_values(*(this->dof_handler),
                                             function_map,
                                             tmp_constraints);
  }

  if (vector_boundary_conditions->time_dependent_bcs_map.find(BCType::normal_flux)
      != vector_boundary_conditions->time_dependent_bcs_map.end())
  {
    FunctionMap                   function_map;
    std::set<types::boundary_id>  boundary_id_set;

    // Extract a pair of iterators representing the upper and
    // lower limit of the iterator range of all the boundary ids on
    // which a time dependent boundary condition of the type BCType::normal_flux
    // was specified.
    const std::pair<const_iterator, const_iterator>
    iterator_pair =
        vector_boundary_conditions->time_dependent_bcs_map.equal_range(BCType::normal_flux);

    // Iterate over all necessary boundary conditions to populate the function
    // map to constrain the AffineConstraints instance.
    const_iterator it = iterator_pair.first, endit = iterator_pair.second;
    for (; it != endit; ++it)
    {
      function_map[it->second] = vector_boundary_conditions->normal_flux_bcs[it->second].get();
      boundary_id_set.insert(it->second);
    }

    VectorTools::compute_nonzero_normal_flux_constraints(*(this->dof_handler),
                                                         0,
                                                         boundary_id_set,
                                                         function_map,
                                                         tmp_constraints);
  }

  if (vector_boundary_conditions->time_dependent_bcs_map.find(BCType::tangential_flux)
      != vector_boundary_conditions->time_dependent_bcs_map.end())
  {
    FunctionMap                   function_map;
    std::set<types::boundary_id>  boundary_id_set;

    // Extract a pair of iterators representing the upper and
    // lower limit of the iterator range of all the boundary ids on
    // which a time dependent boundary condition of the type BCType::tangential_flux
    // was specified.
    const std::pair<const_iterator, const_iterator>
    iterator_pair =
        vector_boundary_conditions->time_dependent_bcs_map.equal_range(BCType::tangential_flux);

    // Iterate over all necessary boundary conditions to populate the function
    // map to constrain the AffineConstraints instance.
    const_iterator it = iterator_pair.first, endit = iterator_pair.second;
    for (; it != endit; ++it)
    {
      function_map[it->second] = vector_boundary_conditions->tangential_flux_bcs[it->second].get();
      boundary_id_set.insert(it->second);
    }

    VectorTools::compute_nonzero_tangential_flux_constraints(*(this->dof_handler),
                                                             0,
                                                             boundary_id_set,
                                                             function_map,
                                                             tmp_constraints);
  }

  tmp_constraints.close();

  this->constraints.merge(tmp_constraints,
                          AffineConstraints<LinearAlgebra::MPI::Vector::value_type>::MergeConflictBehavior::right_object_wins);
}



template <int dim, typename VectorType>
FE_ScalarField<dim, VectorType>::FE_ScalarField
(const unsigned int         fe_degree,
 const Triangulation<dim>  &triangulation,
 const std::string         &name)
:
FE_FieldBase<dim, VectorType>(triangulation, name)
{
  this->finite_element = std::make_shared<FE_Q<dim>>(fe_degree);
  this->boundary_conditions = new ScalarBoundaryConditions<dim>(triangulation);
}



template <int dim, typename VectorType>
FE_ScalarField<dim, VectorType>::FE_ScalarField
(const FE_ScalarField<dim, VectorType>  &entity,
 const std::string        &new_name)
:
FE_FieldBase<dim, VectorType>(entity, new_name)
{
  this->boundary_conditions = new ScalarBoundaryConditions<dim>(this->triangulation);
}


template <int dim, typename VectorType>
void FE_ScalarField<dim, VectorType>::apply_boundary_conditions(const bool check_regularity)
{
  FE_FieldBase<dim, VectorType>::apply_boundary_conditions(check_regularity);

  const ScalarBoundaryConditions<dim>* const scalar_boundary_conditions =
      static_cast<const ScalarBoundaryConditions<dim> *>(this->boundary_conditions);

  if (scalar_boundary_conditions->datum_at_boundary())
  {
    IndexSet    boundary_dofs;
    DoFTools::extract_boundary_dofs(*(this->dof_handler),
                                    ComponentMask(this->finite_element->n_components(),
                                                  true),
                                    boundary_dofs);

    // Look for an admissible local degree of freedom to constrain
    types::global_dof_index local_idx = numbers::invalid_dof_index;
    IndexSet::ElementIterator idx = boundary_dofs.begin();
    IndexSet::ElementIterator endidx = boundary_dofs.end();
    for(; idx != endidx; ++idx)
      if (this->constraints.can_store_line(*idx) &&
          !this->constraints.is_constrained(*idx))
      {
        local_idx = *idx;
        break;
      }

    // Chooses the degree of freedom with the smallest index. If no
    // admissible degree of freedom was found in a given processor, its
    // value is set the number of degree of freedom
    types::global_dof_index global_idx;

    // ensure that at least one processor found things
    const parallel::TriangulationBase<dim> *tria_ptr =
        dynamic_cast<const parallel::TriangulationBase<dim> *>(&this->triangulation);
    if (tria_ptr != nullptr)
    {
      global_idx =
          Utilities::MPI::min((local_idx != numbers::invalid_dof_index)? local_idx: this->dof_handler->n_dofs(),
                              tria_ptr->get_communicator());
    }
    else
      global_idx = local_idx;

    // Checks that an admissable degree of freedom was found
    AssertThrow(global_idx < this->dof_handler->n_dofs(),
                ExcMessage("Error, couldn't find a DoF to constrain."));

    // Sets the degree of freedom to zero
    if (this->constraints.can_store_line(global_idx))
    {
        AssertThrow(!this->constraints.is_constrained(global_idx),
                    ExcInternalError());
        this->constraints.add_line(global_idx);
    }
  }

  this->constraints.close();
}



template <int dim, typename VectorType>
void FE_ScalarField<dim, VectorType>::set_neumann_boundary_condition
(const types::boundary_id boundary_id,
 const std::shared_ptr<Function<dim>> &function,
 const bool time_dependent_bc)
{
  static_cast<ScalarBoundaryConditions<dim>*>(this->boundary_conditions)
    ->set_neumann_bc(boundary_id, function, time_dependent_bc);
}



template <int dim, typename VectorType>
void FE_ScalarField<dim, VectorType>::set_datum_boundary_condition()
{
  static_cast<ScalarBoundaryConditions<dim>*>(this->boundary_conditions)
    ->set_datum_at_boundary();
}



template <int dim, typename VectorType>
typename FE_FieldBase<dim, VectorType>::value_type
FE_ScalarField<dim, VectorType>::point_value
(const Point<dim>   &point,
 const Mapping<dim> &external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  double  point_value;

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    point_value = VectorTools::point_value(external_mapping,
                                           *(this->dof_handler),
                                           this->solution,
                                           point);
    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  // ensure that at least one processor found things
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&this->triangulation);
  if (tria_ptr != nullptr)
  {
    const MPI_Comm  &mpi_communicator(tria_ptr->get_communicator());
    const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, mpi_communicator);
    AssertThrow(n_procs > 0,
                ExcMessage("While trying to evaluate the solution at point " +
                           Utilities::to_string(point[0]) + ", " +
                           Utilities::to_string(point[1]) +
                           (dim == 3 ?
                               ", " + Utilities::to_string(point[2]) :
                               "") +
                           "), " +
                           "no processors reported that the point lies inside the " +
                           "set of cells they own. Are you trying to evaluate the " +
                           "solution at a point that lies outside of the domain?"));

    // Reduce all collected values into local Vector
    point_value = Utilities::MPI::sum(point_value,
                                      mpi_communicator);

    // Normalize in cases where points are claimed by multiple processors
    if (n_procs > 1)
      point_value /= n_procs;
  }

  return (point_value);
}


template <int dim, typename VectorType>
Tensor<1, dim> FE_ScalarField<dim, VectorType>::point_gradient
(const Point<dim>   &point,
 const Mapping<dim> &external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  std::vector<Tensor<1, dim>> point_gradient(1);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    point_gradient[0] = VectorTools::point_gradient(external_mapping,
                                                    *(this->dof_handler),
                                                    this->solution,
                                                    point);
    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  this->send_point_data_vector(point_gradient, point, point_found);

  return (point_gradient[0]);
}



} // namespace Entities

} // namespace RMHD

// explicit instantiations
template class RMHD::Entities::FE_FieldBase<2>;
template class RMHD::Entities::FE_FieldBase<3>;

template class RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>;
template class RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>;

template class RMHD::Entities::FE_ScalarField<2>;
template class RMHD::Entities::FE_ScalarField<3>;

template class RMHD::Entities::FE_ScalarField<2, dealii::Vector<double>>;
template class RMHD::Entities::FE_ScalarField<3, dealii::Vector<double>>;

template class RMHD::Entities::FE_VectorField<2>;
template class RMHD::Entities::FE_VectorField<3>;

template class RMHD::Entities::FE_VectorField<2, dealii::Vector<double>>;
template class RMHD::Entities::FE_VectorField<3, dealii::Vector<double>>;

