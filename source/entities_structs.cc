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
  solution_n.reinit(locally_relevant_dofs, MPI_COMM_WORLD);
  solution_n_minus_1.reinit(solution_n);
}

template <int dim>
Velocity<dim>::Velocity(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : EntityBase<dim>(fe_degree, triangulation),
    fe(FE_Q<dim>(fe_degree), dim)
{}

template <int dim>
Pressure<dim>::Pressure(
  const unsigned int                              &fe_degree,
  const parallel::distributed::Triangulation<dim> &triangulation)
  : EntityBase<dim>(fe_degree, triangulation),
    fe(fe_degree)
{}

} // namespace Entities
} // namespace RMHD

template struct RMHD::Entities::EntityBase<2>;
template struct RMHD::Entities::EntityBase<3>;
template struct RMHD::Entities::Velocity<2>;
template struct RMHD::Entities::Velocity<3>;
template struct RMHD::Entities::Pressure<2>;
template struct RMHD::Entities::Pressure<3>;