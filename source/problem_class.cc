#include <rotatingMHD/problem_class.h>

#include <deal.II/numerics/vector_tools.h>

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

template <int dim>
void Problem<dim>::
mpi_point_value()
{

}

} // namespace RMHD

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
