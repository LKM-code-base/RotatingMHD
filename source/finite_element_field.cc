#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/finite_element_field.h>

#include <algorithm>

namespace RMHD
{

using namespace dealii;

namespace Entities
{



template <int dim>
FE_FieldBase<dim>::FE_FieldBase
(const parallel::distributed::Triangulation<dim> &triangulation,
 const std::string                               &name)
:
name(name),
flag_child_entity(false),
flag_setup_dofs(true),
mpi_communicator(triangulation.get_communicator()),
triangulation(triangulation),
dof_handler(std::make_shared<DoFHandler<dim>>())
{}



template <int dim>
FE_FieldBase<dim>::FE_FieldBase
(const FE_FieldBase<dim>  &entity,
 const std::string      &new_name)
:
name(new_name),
flag_child_entity(true),
flag_setup_dofs(entity.flag_setup_dofs),
mpi_communicator(entity.get_triangulation().get_communicator()),
triangulation(entity.get_triangulation()),
dof_handler(entity.dof_handler),
finite_element(entity.finite_element)
{}

template <int dim>
void FE_FieldBase<dim>::clear()
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

  flag_setup_dofs = true;
}

template <int dim>
void FE_FieldBase<dim>::setup_dofs()
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

template <int dim>
void FE_FieldBase<dim>::setup_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

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



template <int dim>
void FE_FieldBase<dim>::update_solution_vectors()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  old_old_solution  = old_solution;
  old_solution      = solution;
}



template <int dim>
void FE_FieldBase<dim>::set_solution_vectors_to_zero()
{
  Assert(!flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  solution          = 0.;
  old_solution      = 0.;
  old_old_solution  = 0.;
}

template<int dim>
std::map<typename VectorTools::NormType, double>
FE_FieldBase<dim>::compute_error
(const Function<dim>	&exact_solution,
 const std::shared_ptr<Mapping<dim>> external_mapping) const
{
  const Triangulation<dim> &tria{this->triangulation};

  Vector<double>  cellwise_error(tria.n_active_cells());

  auto compute_error
  = [external_mapping, &tria, &cellwise_error, this]
     (const Quadrature<dim>          &quadrature,
      const Function<dim>            &exact_solution,
      const VectorTools::NormType     norm_type)
  ->
  double
  {
    VectorTools::integrate_difference(*external_mapping,
                                      *this->dof_handler,
                                      this->solution,
                                      exact_solution,
                                      cellwise_error,
                                      quadrature,
                                      norm_type);
    return (VectorTools::compute_global_error(tria,
                                              cellwise_error,
                                              norm_type));
  };

  typename VectorTools::NormType norm_type;

  std::map<typename VectorTools::NormType, double> error_map;

  const unsigned int fe_degree{finite_element->degree};
	const QGauss<dim> quadrature_formula(fe_degree + 2);

	norm_type = VectorTools::NormType::L2_norm;
	double error = compute_error(quadrature_formula,
                               exact_solution,
                               norm_type);
  error_map[norm_type] = error;

	norm_type = VectorTools::NormType::H1_norm;
	error = compute_error(quadrature_formula,
                        exact_solution,
                        norm_type);
	error_map[norm_type] = error;

	const QTrapez<1>     trapezoidal_rule;
	const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                 fe_degree);

	norm_type = VectorTools::NormType::Linfty_norm;
	error = compute_error(linfty_quadrature_formula,
                        exact_solution,
                        norm_type);
	error_map[norm_type] = error;

	return (error_map);
}

template <int dim>
FE_VectorField<dim>::FE_VectorField
(const unsigned int                               fe_degree,
 const parallel::distributed::Triangulation<dim> &triangulation,
 const std::string                               &name)
:
FE_FieldBase<dim>(triangulation, name),
boundary_conditions(triangulation)
{
  this->finite_element = std::make_shared<FESystem<dim>>(FE_Q<dim>(fe_degree), dim);
}



template <int dim>
FE_VectorField<dim>::FE_VectorField
(const FE_VectorField<dim>  &entity,
 const std::string        &new_name)
:
FE_FieldBase<dim>(entity, new_name),
boundary_conditions(entity.get_triangulation())
{}


template <int dim>
void FE_VectorField<dim>::clear()
{
  boundary_conditions.clear();
  FE_FieldBase<dim>::clear();
}

template <int dim>
void FE_VectorField<dim>::setup_dofs()
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



template <int dim>
void FE_VectorField<dim>::apply_boundary_conditions(const bool check_regularity)
{
  if (check_regularity == true)
    AssertThrow(boundary_conditions.regularity_guaranteed(),
                ExcMessage("No boundary conditions were set for the \""
                            + this->name + "\" entity"));

  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  Assert(boundary_conditions.closed(),
         ExcMessage("The boundary conditions have not been closed."));

  using FunctionMap = std::map<types::boundary_id,
                              const Function<dim> *>;
  this->constraints.clear();
  this->constraints.reinit(this->locally_relevant_dofs);
  this->constraints.merge(this->hanging_node_constraints);

  if (!boundary_conditions.periodic_bcs.empty())
  {
    FEValuesExtractors::Vector extractor(0);

    std::vector<unsigned int> first_vector_components;
    first_vector_components.push_back(0);

    std::vector<
    GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

    for (auto const &periodic_bc : boundary_conditions.periodic_bcs)
      GridTools::collect_periodic_faces(
        *(this->dof_handler),
        periodic_bc.boundary_pair.first,
        periodic_bc.boundary_pair.second,
        periodic_bc.direction,
        periodicity_vector,
        periodic_bc.offset,
        periodic_bc.rotation_matrix);

    DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
      periodicity_vector,
      this->constraints,
      this->finite_element->component_mask(extractor),
      first_vector_components);
  }

  if (!boundary_conditions.dirichlet_bcs.empty())
  {
    FunctionMap function_map;

    for (auto const &[boundary_id, function] : boundary_conditions.dirichlet_bcs)
      function_map[boundary_id] = function.get();

    VectorTools::interpolate_boundary_values(
      *(this->dof_handler),
      function_map,
      this->constraints);
  }

  if (!boundary_conditions.normal_flux_bcs.empty())
  {
    FunctionMap                   function_map;

    std::set<types::boundary_id>  boundary_id_set;

    for (auto const &[boundary_id, function] : boundary_conditions.normal_flux_bcs)
    {
      function_map[boundary_id] = function.get();
      boundary_id_set.insert(boundary_id);
    }

    VectorTools::compute_nonzero_normal_flux_constraints(
      *(this->dof_handler),
      0,
      boundary_id_set,
      function_map,
      this->constraints);
  }

  if (!boundary_conditions.tangential_flux_bcs.empty())
  {
    FunctionMap                   function_map;

    std::set<types::boundary_id>  boundary_id_set;

    for (auto const &[boundary_id, function] : boundary_conditions.tangential_flux_bcs)
    {
      function_map[boundary_id] = function.get();
      boundary_id_set.insert(boundary_id);
    }

    VectorTools::compute_nonzero_tangential_flux_constraints(
      *(this->dof_handler),
      0,
      boundary_id_set,
      function_map,
      this->constraints);
  }

  /*! @todo Implement a datum for vector valued problem*/

  this->constraints.close();
}



template <int dim>
void FE_VectorField<dim>::close_boundary_conditions(const bool print_summary)
{
  boundary_conditions.close();

  if (print_summary)
  {
      ConditionalOStream  pcout(std::cout,
                                Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
      boundary_conditions.print_summary(pcout, this->name);
  }
}



template <int dim>
void FE_VectorField<dim>::update_boundary_conditions()
{
  if (boundary_conditions.time_dependent_bcs_map.empty())
    return;

  using FunctionMap = std::map<types::boundary_id,
                              const Function<dim> *>;

  AffineConstraints<double>   tmp_constraints;

  tmp_constraints.clear();

  tmp_constraints.reinit(this->locally_relevant_dofs);

  if (boundary_conditions.time_dependent_bcs_map.find(BCType::dirichlet)
      != boundary_conditions.time_dependent_bcs_map.end())
  {
    FunctionMap   function_map;

    /*!
     * Extract an std::pair containing the upper and
     * lower limit of the iteration range of all the boundary ids on
     * which a time dependent given BCType boundary condition was set.
     */

    auto iterator_range =
      boundary_conditions.time_dependent_bcs_map.equal_range(
        BCType::dirichlet);

    /*!
     * The variable multimap_pair is a std::pair<BCType, boundary_id>,
     * from which the the boundary_id is extracted to populate the
     * std::map<boundary_id, const Function<dim> *> needed to
     * constraint the AffineConstraints instance.
     */

    for (auto multimap_pair = iterator_range.first;
         multimap_pair != iterator_range.second;
         ++multimap_pair)
      function_map[multimap_pair->second] =
        boundary_conditions.dirichlet_bcs[multimap_pair->second].get();

    VectorTools::interpolate_boundary_values(
      *(this->dof_handler),
      function_map,
      tmp_constraints);
  }

  if (boundary_conditions.time_dependent_bcs_map.find(BCType::normal_flux)
      != boundary_conditions.time_dependent_bcs_map.end())
  {
    FunctionMap                   function_map;

    std::set<types::boundary_id>  boundary_id_set;

    /*!
     * Extract an std::pair containing the upper and
     * lower limit of the iteration range of all the boundary ids on
     * which a time dependent given BCType boundary condition was set.
     */

    auto iterator_range =
      boundary_conditions.time_dependent_bcs_map.equal_range(
        BCType::normal_flux);

    /*!
     * The variable multimap_pair is a std::pair<BCType, boundary_id>,
     * from which the the boundary_id is extracted to populate the
     * std::map<boundary_id, const Function<dim> *> and
     * std::set<boundary_id> instances needed to
     * constraint the AffineConstraints instance.
     */

    for (auto multimap_pair = iterator_range.first;
         multimap_pair != iterator_range.second;
         ++multimap_pair)
    {
      function_map[multimap_pair->second] =
        boundary_conditions.normal_flux_bcs[multimap_pair->second].get();
      boundary_id_set.insert(multimap_pair->second);
    }

    VectorTools::compute_nonzero_normal_flux_constraints(
      *(this->dof_handler),
      0,
      boundary_id_set,
      function_map,
      tmp_constraints);
  }

  if (boundary_conditions.time_dependent_bcs_map.find(BCType::tangential_flux)
      != boundary_conditions.time_dependent_bcs_map.end())
  {
    FunctionMap                   function_map;

    std::set<types::boundary_id>  boundary_id_set;

    /*!
     * Extract an std::pair containing the upper and
     * lower limit of the iteration range of all the boundary ids on
     * which a time dependent given BCType boundary condition was set.
     */

    auto iterator_range =
      boundary_conditions.time_dependent_bcs_map.equal_range(
        BCType::tangential_flux);

    /*!
     * The variable multimap_pair is a std::pair<BCType, boundary_id>,
     * from which the the boundary_id is extracted to populate the
     * std::map<boundary_id, const Function<dim> *> and
     * std::set<boundary_id> instances needed to
     * constraint the AffineConstraints instance.
     */

    for (auto multimap_pair = iterator_range.first;
         multimap_pair != iterator_range.second;
         ++multimap_pair)
    {
      function_map[multimap_pair->second] =
        boundary_conditions.tangential_flux_bcs[multimap_pair->second].get();
      boundary_id_set.insert(multimap_pair->second);
    }

    VectorTools::compute_nonzero_tangential_flux_constraints(
      *(this->dof_handler),
      0,
      boundary_id_set,
      function_map,
      tmp_constraints);
  }

  tmp_constraints.close();

  this->constraints.merge(
    tmp_constraints,
    AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
}



template <int dim>
void FE_VectorField<dim>::clear_boundary_conditions()
{
  boundary_conditions.clear();

  this->constraints.clear();
}



template<int dim>
Tensor<1,dim> FE_VectorField<dim>::point_value(
  const Point<dim>                    &point,
  const std::shared_ptr<Mapping<dim>> external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  Vector<double>  point_value(dim);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    if (external_mapping != nullptr)
      VectorTools::point_value(*external_mapping,
                               *(this->dof_handler),
                               this->solution,
                               point,
                               point_value);
    else
      VectorTools::point_value(*(this->dof_handler),
                               this->solution,
                               point,
                               point_value);

    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  // ensure that at least one processor found things
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, this->mpi_communicator);
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
                      this->mpi_communicator,
                      point_value);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    point_value /= n_procs;

  Tensor<1,dim> point_value_tensor;
  for (unsigned d=0; d<dim; ++d)
    point_value_tensor[d] = point_value[d];

  return (point_value_tensor);
}



template<int dim>
Tensor<2,dim> FE_VectorField<dim>::point_gradient(
  const Point<dim>                    &point,
  const std::shared_ptr<Mapping<dim>> external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  std::vector<Tensor<1,dim>>  point_gradient_rows(dim);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    if (external_mapping != nullptr)
      VectorTools::point_gradient(*external_mapping,
                                  *(this->dof_handler),
                                  this->solution,
                                  point,
                                  point_gradient_rows);
    else
      VectorTools::point_gradient(*(this->dof_handler),
                                  this->solution,
                                  point,
                                  point_gradient_rows);

    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  // ensure that at least one processor found things
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, this->mpi_communicator);
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
  for (auto &row : point_gradient_rows)
    row = Utilities::MPI::sum(row,
                              this->mpi_communicator);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    for (auto &row : point_gradient_rows)
      row /= n_procs;

  Tensor<2,dim> point_gradient_tensor;

  for (unsigned i=0; i<dim; ++i)
    for (unsigned j=0; j<dim; ++j)
      point_gradient_tensor[i][j] = point_gradient_rows[i][j];

  return (point_gradient_tensor);
}



template <int dim>
FE_ScalarField<dim>::FE_ScalarField
(const unsigned int                               fe_degree,
 const parallel::distributed::Triangulation<dim> &triangulation,
 const std::string                               &name)
:
FE_FieldBase<dim>(triangulation, name),
boundary_conditions(triangulation)
{
  this->finite_element = std::make_shared<FE_Q<dim>>(fe_degree);
}



template <int dim>
FE_ScalarField<dim>::FE_ScalarField
(const FE_ScalarField<dim>  &entity,
 const std::string        &new_name)
:
FE_FieldBase<dim>(entity, new_name),
boundary_conditions(entity.get_triangulation())
{}

template <int dim>
void FE_ScalarField<dim>::clear()
{
  boundary_conditions.clear();

  FE_FieldBase<dim>::clear();
}

template <int dim>
void FE_ScalarField<dim>::apply_boundary_conditions(const bool check_regularity)
{
  if (check_regularity == true)
    AssertThrow(boundary_conditions.regularity_guaranteed(),
                ExcMessage("No boundary conditions were set for the \""
                            + this->name + "\" entity"));

  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  Assert(boundary_conditions.closed(),
         ExcMessage("The boundary conditions have not been closed."));

  using FunctionMap = std::map<types::boundary_id,
                              const Function<dim> *>;
  this->constraints.clear();
  this->constraints.reinit(this->locally_relevant_dofs);
  this->constraints.merge(this->hanging_node_constraints);
  if (!boundary_conditions.periodic_bcs.empty())
  {
    std::vector<unsigned int> first_vector_components;
    first_vector_components.push_back(0);

    std::vector<
    GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

    for (auto const &periodic_bc : boundary_conditions.periodic_bcs)
      GridTools::collect_periodic_faces(
        *(this->dof_handler),
        periodic_bc.boundary_pair.first,
        periodic_bc.boundary_pair.second,
        periodic_bc.direction,
        periodicity_vector,
        periodic_bc.offset);

    DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
      periodicity_vector,
      this->constraints);
  }
  if (!boundary_conditions.dirichlet_bcs.empty())
  {
    FunctionMap function_map;

    for (auto const &[boundary_id, function] : boundary_conditions.dirichlet_bcs)
      function_map[boundary_id] = function.get();

    VectorTools::interpolate_boundary_values(
      *(this->dof_handler),
      function_map,
      this->constraints);
  }

  if (boundary_conditions.datum_set_at_boundary())
  {
    IndexSet    boundary_dofs;
    DoFTools::extract_boundary_dofs(*(this->dof_handler),
                                    ComponentMask(this->finite_element->n_components(),
                                                  true),
                                    boundary_dofs);

    // Looks for an admissible local degree of freedom to constrain
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
    const types::global_dof_index global_idx
      = Utilities::MPI::min((local_idx != numbers::invalid_dof_index)
                              ? local_idx
                              : this->dof_handler->n_dofs(),
                             this->mpi_communicator);

    // Checks that an admissable degree of freedom was found
    Assert(global_idx < this->dof_handler->n_dofs(),
           ExcMessage("Error, couldn't find a DoF to constrain."));

    // Sets the degree of freedom to zero
    if (this->constraints.can_store_line(global_idx))
    {
        Assert(!this->constraints.is_constrained(global_idx),
               ExcInternalError());
        this->constraints.add_line(global_idx);
    }
  }

  this->constraints.close();
}

template <int dim>
void FE_ScalarField<dim>::close_boundary_conditions(const bool print_summary)
{
  boundary_conditions.close();

  if (print_summary)
  {
      ConditionalOStream  pcout(std::cout,
                                Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
      boundary_conditions.print_summary(pcout, this->name);
  }
}

template <int dim>
void FE_ScalarField<dim>::update_boundary_conditions()
{
  if (boundary_conditions.time_dependent_bcs_map.empty())
    return;

  using FunctionMap = std::map<types::boundary_id,
                              const Function<dim> *>;

  AffineConstraints<double>   tmp_constraints;

  tmp_constraints.clear();

  tmp_constraints.reinit(this->locally_relevant_dofs);

  if (boundary_conditions.time_dependent_bcs_map.find(BCType::dirichlet)
      != boundary_conditions.time_dependent_bcs_map.end())
  {
    FunctionMap   function_map;

    /*!
     * Extract an std::pair containing the upper and
     * lower limit of the iteration range of all the boundary ids on
     * which a time dependent given BCType boundary condition was set.
     */

    auto iterator_range =
      boundary_conditions.time_dependent_bcs_map.equal_range(
        BCType::dirichlet);

    /*!
     * The variable multimap_pair is a std::pair<BCType, boundary_id>,
     * from which the the boundary_id is extracted to populate the
     * std::map<boundary_id, const Function<dim> *> instance needed to
     * constraint the AffineConstraints instance.
     */

    for (auto multimap_pair = iterator_range.first;
         multimap_pair != iterator_range.second;
         ++multimap_pair)
      function_map[multimap_pair->second] =
        boundary_conditions.dirichlet_bcs[multimap_pair->second].get();

    VectorTools::interpolate_boundary_values(
      *(this->dof_handler),
      function_map,
      tmp_constraints);
  }

  tmp_constraints.close();

  this->constraints.merge(
    tmp_constraints,
    AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
}



template <int dim>
void FE_ScalarField<dim>::clear_boundary_conditions()
{
  boundary_conditions.clear();

  this->constraints.clear();
}



template<int dim>
double FE_ScalarField<dim>::point_value(
  const Point<dim>                    &point,
  const std::shared_ptr<Mapping<dim>> external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  double  point_value = 0.0;

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    if (external_mapping != nullptr)
      point_value = VectorTools::point_value(*external_mapping,
                                             *(this->dof_handler),
                                             this->solution,
                                             point);
    else
      point_value = VectorTools::point_value(*(this->dof_handler),
                                             this->solution,
                                             point);
    point_found = true;
  }
  catch (const VectorTools::ExcPointNotAvailableHere &)
  {
    // ignore
  }

  // ensure that at least one processor found things
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, this->mpi_communicator);
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
                                    this->mpi_communicator);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    point_value /= n_procs;

  return (point_value);
}



template<int dim>
Tensor<1,dim> FE_ScalarField<dim>::point_gradient(
  const Point<dim> &point,
  const std::shared_ptr<Mapping<dim>> external_mapping) const
{
  Assert(!this->flag_setup_dofs, ExcMessage("Setup dofs was not called."));

  Tensor<1,dim>  point_gradient;

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    if (external_mapping != nullptr)
      point_gradient = VectorTools::point_gradient(
                              *external_mapping,
                              *(this->dof_handler),
                              this->solution,
                              point);
    else
      point_gradient = VectorTools::point_gradient(
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
  const int n_procs = Utilities::MPI::sum(point_found ? 1 : 0, this->mpi_communicator);
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
  point_gradient = Utilities::MPI::sum(point_gradient,
                                       this->mpi_communicator);

  // Normalize in cases where points are claimed by multiple processors
  if (n_procs > 1)
    point_gradient /= n_procs;

  return (point_gradient);
}


} // namespace Entities

} // namespace RMHD

template struct RMHD::Entities::FE_FieldBase<2>;
template struct RMHD::Entities::FE_FieldBase<3>;

template struct RMHD::Entities::FE_VectorField<2>;
template struct RMHD::Entities::FE_VectorField<3>;

template struct RMHD::Entities::FE_ScalarField<2>;
template struct RMHD::Entities::FE_ScalarField<3>;
