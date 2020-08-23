#include <rotatingMHD/entities_structs.h>

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
ScalarEntity<dim>::ScalarEntity(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : EntityBase<dim>(fe_degree, triangulation),
    fe(fe_degree)
{}

} // namespace Entities
} // namespace RMHD

template struct RMHD::Entities::EntityBase<2>;
template struct RMHD::Entities::EntityBase<3>;
template struct RMHD::Entities::VectorEntity<2>;
template struct RMHD::Entities::VectorEntity<3>;
template struct RMHD::Entities::ScalarEntity<2>;
template struct RMHD::Entities::ScalarEntity<3>;