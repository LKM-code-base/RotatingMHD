#include <rotatingMHD/problem_class.h>

#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>

namespace RMHD
{
  using namespace dealii;

template <int dim>
void Problem<dim>::
set_initial_conditions(
                      Entities::EntityBase<dim>         &entity,
                      Function<dim>                     &function,
                      TimeDiscretization::VSIMEXMethod  &time_stepping)
{
  switch (time_stepping.get_order())
  {
    case 1 :
      {
      TrilinosWrappers::MPI::Vector tmp_old_solution(
                                            entity.locally_owned_dofs);
      function.set_time(time_stepping.get_start_time());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_old_solution);

      entity.old_solution          = tmp_old_solution;
      break;
      }
    case 2 :
      {
      TrilinosWrappers::MPI::Vector tmp_old_old_solution(
                                            entity.locally_owned_dofs);
      TrilinosWrappers::MPI::Vector tmp_old_solution(
                                            entity.locally_owned_dofs);
      function.set_time(time_stepping.get_start_time());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_old_old_solution);

      function.advance_time(time_stepping.get_next_step_size());
      VectorTools::project(entity.dof_handler,
                           entity.constraints,
                           QGauss<dim>(entity.fe_degree + 2),
                           function,
                           tmp_old_solution);

      entity.old_old_solution = tmp_old_old_solution;
      entity.old_solution     = tmp_old_solution;
      break;
      }
    default:
      Assert(false, ExcNotImplemented());
  };

}
/*
template <int dim>
void Problem<dim>::
mpi_point_value(const Entities::EntityBase<dim> &entity,
                const Point<dim>                &point,
                double                          &scalar_point_value)
{
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                    Point<dim>> cell_point =
      GridTools::find_active_cell_around_point(
                                      StaticMappingQ1<dim, dim>::mapping,
                                      entity.dof_handler, 
                                      point);
  scalar_point_value = 0.;
  
  if (cell_point.first->is_locally_owned())
    scalar_point_value = VectorTools::point_value(entity.dof_handler,
                                                  entity.solution,
                                                  point);
  scalar_point_value = 
              Utilities::MPI::sum(scalar_point_value, MPI_COMM_WORLD);
}

template <int dim>
void Problem<dim>::
mpi_point_value(const Entities::EntityBase<dim> &entity,
                const Point<dim>                &point,
                Vector<double>                  &vector_point_value)
{
  Assert((vector_point_value.size() == dim), ExcNotImplemented());
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                    Point<dim>> cell_point =
      GridTools::find_active_cell_around_point(
                                      StaticMappingQ1<dim, dim>::mapping,
                                      entity.dof_handler, 
                                      point);
  vector_point_value = 0.;
  if (cell_point.first->is_locally_owned())
    VectorTools::point_value(entity.dof_handler,
                             entity.solution,
                             point,
                             vector_point_value);
  Utilities::MPI::sum(vector_point_value, 
                      MPI_COMM_WORLD,
                      vector_point_value);
}
*/
} // namespace RMHD

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
