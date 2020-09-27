#include <rotatingMHD/entities_structs.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
namespace RMHD
{
  using namespace dealii;

namespace Entities
{

template <int dim>
EntityBase<dim>::EntityBase(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : fe_degree(fe_degree),
    dof_handler(triangulation),
    quadrature_formula(fe_degree + 1)
  {}

template <int dim>
void EntityBase<dim>::reinit()
{
  solution.reinit(locally_relevant_dofs, MPI_COMM_WORLD);
  old_solution.reinit(solution);
  old_old_solution.reinit(solution);
}

template <int dim>
void EntityBase<dim>::update_solution_vectors()
{
  old_old_solution  = old_solution;
  old_solution      = solution;
}

template <int dim>
VectorEntity<dim>::VectorEntity(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : EntityBase<dim>(fe_degree, triangulation),
    fe(FE_Q<dim>(fe_degree), dim)
{}

template <int dim>
void VectorEntity<dim>::setup_dofs()
{
  this->dof_handler.distribute_dofs(this->fe);
  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);
  this->hanging_nodes.clear();
  this->hanging_nodes.reinit(this->locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(this->dof_handler,
                                          this->hanging_nodes);
  this->hanging_nodes.close();
}

template <int dim>
void VectorEntity<dim>::apply_boundary_conditions()
{
  using FunctionMap = std::map<types::boundary_id, 
                              const Function<dim> *>;
  this->constraints.clear();
  this->constraints.reinit(this->locally_relevant_dofs);
  this->constraints.merge(this->hanging_nodes);
  if (!boundary_conditions.periodic_bcs.empty())
  {

  }
  if (!boundary_conditions.dirichlet_bcs.empty())
  {
    FunctionMap function_map;

    for (auto const &[boundary_id, function] : boundary_conditions.dirichlet_bcs)
      function_map[boundary_id] = function.get();
    
    VectorTools::interpolate_boundary_values(
      this->dof_handler,
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
      this->dof_handler,
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
      this->dof_handler,
      0,
      boundary_id_set,
      function_map,
      this->constraints);
  }
  this->constraints.close();
}

template <int dim>
ScalarEntity<dim>::ScalarEntity(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : EntityBase<dim>(fe_degree, triangulation),
    fe(fe_degree)
{}

template <int dim>
void ScalarEntity<dim>::setup_dofs()
{
  this->dof_handler.distribute_dofs(this->fe);
  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);
  
  this->hanging_nodes.clear();
  this->hanging_nodes.reinit(this->locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(this->dof_handler,
                                          this->hanging_nodes);
  this->hanging_nodes.close();
}

template <int dim>
void ScalarEntity<dim>::apply_boundary_conditions()
{
  using FunctionMap = std::map<types::boundary_id, 
                              const Function<dim> *>;
  this->constraints.clear();
  this->constraints.reinit(this->locally_relevant_dofs);
  this->constraints.merge(this->hanging_nodes);
  if (!boundary_conditions.periodic_bcs.empty())
  {

  }
  if (!boundary_conditions.dirichlet_bcs.empty())
  {
    FunctionMap function_map;

    for (auto const &[boundary_id, function] : boundary_conditions.dirichlet_bcs)
      function_map[boundary_id] = function.get();
    
    VectorTools::interpolate_boundary_values(
      this->dof_handler,
      function_map,
      this->constraints);
  }
  this->constraints.close();
}

} // namespace Entities
} // namespace RMHD

template struct RMHD::Entities::EntityBase<2>;
template struct RMHD::Entities::EntityBase<3>;
template struct RMHD::Entities::VectorEntity<2>;
template struct RMHD::Entities::VectorEntity<3>;
template struct RMHD::Entities::ScalarEntity<2>;
template struct RMHD::Entities::ScalarEntity<3>;