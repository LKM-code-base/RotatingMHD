#include <rotatingMHD/entities_structs.h>

#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

using namespace dealii;

namespace Entities
{

template <int dim>
EntityBase<dim>::EntityBase
(const unsigned int                               fe_degree,
 const parallel::distributed::Triangulation<dim> &triangulation)
:
fe_degree(fe_degree),
mpi_communicator(MPI_COMM_WORLD),
dof_handler(triangulation),
quadrature_formula(fe_degree + 1)
{}

template <int dim>
void EntityBase<dim>::reinit()
{
  solution.reinit(locally_relevant_dofs,
                  mpi_communicator);
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
VectorEntity<dim>::VectorEntity
(const unsigned int                               fe_degree,
 const parallel::distributed::Triangulation<dim> &triangulation)
:
EntityBase<dim>(fe_degree, triangulation),
fe(FE_Q<dim>(fe_degree), dim)
{}

template <int dim>
void VectorEntity<dim>::setup_dofs()
{
  this->dof_handler.distribute_dofs(this->fe);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();

  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);
}

template<int dim>
Tensor<1,dim> VectorEntity<dim>::point_value(const Point<dim> &point) const
{
  Vector<double>  point_value(dim);

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
    VectorTools::point_value(this->dof_handler,
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

template <int dim>
ScalarEntity<dim>::ScalarEntity
(const unsigned int                               fe_degree,
 const parallel::distributed::Triangulation<dim> &triangulation)
:
EntityBase<dim>(fe_degree, triangulation),
fe(fe_degree)
{}

template <int dim>
void ScalarEntity<dim>::setup_dofs()
{
  this->dof_handler.distribute_dofs(this->fe);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();

  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);
}

template<int dim>
double ScalarEntity<dim>::point_value(const Point<dim> &point) const
{
  double  point_value = 0.0;

  // try to evaluate the solution at this point. in parallel, the point
  // will be on only one processor's owned cells, so the others are
  // going to throw an exception. make sure at least one processor
  // finds the given point
  bool point_found = false;

  try
  {
      point_value = VectorTools::point_value(this->dof_handler,
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


} // namespace Entities

} // namespace RMHD

template struct RMHD::Entities::EntityBase<2>;
template struct RMHD::Entities::EntityBase<3>;
template struct RMHD::Entities::VectorEntity<2>;
template struct RMHD::Entities::VectorEntity<3>;
template struct RMHD::Entities::ScalarEntity<2>;
template struct RMHD::Entities::ScalarEntity<3>;
