#include <rotatingMHD/global.h>
#include <rotatingMHD/problem_class.h>

//#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
//#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

using namespace dealii;

template<int dim>
Problem<dim>::Problem(const RunTimeParameters::ParameterSet &prm)
:
mpi_communicator(MPI_COMM_WORLD),
prm(prm),
triangulation(mpi_communicator,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
pcout(std::make_shared<ConditionalOStream>(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))),
computing_timer(
  std::make_shared<TimerOutput>(mpi_communicator,
                                *pcout,
                                TimerOutput::summary,
                                TimerOutput::wall_times))
{}

template <int dim>
void Problem<dim>::set_initial_conditions
(Entities::EntityBase<dim>              &entity,
 Function<dim>                          &function,
 const TimeDiscretization::VSIMEXMethod &time_stepping)
{
  switch (time_stepping.get_order())
  {
    case 1 :
      {
        #ifdef USE_PETSC_LA
          LinearAlgebra::MPI::Vector
          tmp_old_solution(entity.locally_owned_dofs, mpi_communicator);
        #else
          LinearAlgebra::MPI::Vector
          tmp_old_solution(entity.locally_owned_dofs);
        #endif

        function.set_time(time_stepping.get_start_time());

        VectorTools::project(entity.dof_handler,
                             entity.constraints,
                             QGauss<dim>(entity.fe_degree + 2),
                             function,
                             tmp_old_solution);

        entity.old_solution = tmp_old_solution;
        break;
      }
    case 2 :

      {
        #ifdef USE_PETSC_LA
          LinearAlgebra::MPI::Vector
          tmp_old_solution(entity.locally_owned_dofs, mpi_communicator);
          LinearAlgebra::MPI::Vector
          tmp_old_old_solution(entity.locally_owned_dofs, mpi_communicator);
        #else
          LinearAlgebra::MPI::Vector tmp_old_old_solution(entity.locally_owned_dofs);
          LinearAlgebra::MPI::Vector tmp_old_solution(entity.locally_owned_dofs);
        #endif

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
void Problem<dim>::compute_error(
  LinearAlgebra::MPI::Vector  &error_vector,
  Entities::EntityBase<dim>   &entity,
  Function<dim>               &exact_solution)
{
  #ifdef USE_PETSC_LA
    LinearAlgebra::MPI::Vector
    tmp_error_vector(entity.locally_owned_dofs, mpi_communicator);
  #else
    LinearAlgebra::MPI::Vector
    tmp_error_vector(entity.locally_owned_dofs);
  #endif
  VectorTools::project(entity.dof_handler,
                       entity.constraints,
                       QGauss<dim>(entity.fe_degree + 2),
                       exact_solution,
                       tmp_error_vector);

  error_vector = tmp_error_vector;

  LinearAlgebra::MPI::Vector distributed_error_vector;
  LinearAlgebra::MPI::Vector distributed_solution;

  #ifdef USE_PETSC_LA
    distributed_error_vector.reinit(entity.locally_owned_dofs,
                                    mpi_communicator);
  #else
    distributed_error_vector.reinit(entity.locally_owned_dofs,
                                    entity.locally_relevant_dofs,
                                    mpi_communicator,
                                    true);
  #endif
  distributed_solution.reinit(distributed_error_vector);

  distributed_error_vector  = error_vector;
  distributed_solution      = entity.solution;

  distributed_error_vector.add(-1.0, distributed_solution);
  
  for (unsigned int i = distributed_error_vector.local_range().first; 
       i < distributed_error_vector.local_range().second; ++i)
    if (distributed_error_vector(i) < 0)
      distributed_error_vector(i) *= -1.0;

  error_vector = distributed_error_vector;
}

template <int dim>
double Problem<dim>::compute_next_time_step
(const TimeDiscretization::VSIMEXMethod &time_stepping,
 const double                           cfl_number,
 const double                           max_cfl_number) const
{
  if (!prm.time_stepping_parameters.adaptive_time_stepping)
    return time_stepping.get_next_step_size();

  return max_cfl_number / cfl_number * 
         time_stepping.get_next_step_size();
}

} // namespace RMHD

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
