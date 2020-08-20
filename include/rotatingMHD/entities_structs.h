
#ifndef INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_
#define INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_

#include <rotatingMHD/run_time_parameters.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

namespace RMHD
{
  using namespace dealii;

namespace Entities
{

template <int dim>
struct EntityBase
{
  const unsigned int                  fe_degree;
  DoFHandler<dim>                     dof_handler;
  AffineConstraints<double>           constraints;
  QGauss<dim>                         quadrature_formula;

  IndexSet                            locally_owned_dofs;
  IndexSet                            locally_relevant_dofs;

  TrilinosWrappers::MPI::Vector       solution_n;
  TrilinosWrappers::MPI::Vector       solution_n_minus_1;

  EntityBase(
    const unsigned int                              &fe_degree,
    const parallel::distributed::Triangulation<dim> &triangulation);
  void reinit();
};

template <int dim>
struct Velocity : EntityBase<dim>
{
  FESystem<dim>                       fe;

  Velocity(    
    const unsigned int                              &fe_degree,
    const parallel::distributed::Triangulation<dim> &triangulation);
};

template <int dim>
struct Pressure : EntityBase<dim>
{
  FE_Q<dim>                           fe;

  Pressure(
    const unsigned int                              &fe_degree,
    const parallel::distributed::Triangulation<dim> &triangulation);
};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_ */