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
Problem<dim>::Problem()
:
mpi_communicator(MPI_COMM_WORLD),
triangulation(mpi_communicator,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
computing_timer(mpi_communicator,
                pcout,
                TimerOutput::summary,
                TimerOutput::wall_times)
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
          tmp_old_old_solution(entity.locally_owned_dofs, MPI_COMM_WORLD);
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

} // namespace RMHD

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
