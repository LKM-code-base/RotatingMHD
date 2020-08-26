#include <rotatingMHD/auxiliary_functions.h>


#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>

using namespace dealii;

template <int dim>
void mpi_point_value(const RMHD::Entities::EntityBase<dim> &entity,
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
void mpi_point_value(const RMHD::Entities::EntityBase<dim> &entity,
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

template void mpi_point_value(const RMHD::Entities::EntityBase<2> &,
                              const Point<2>                      &,
                              double                              &);
template void mpi_point_value(const RMHD::Entities::EntityBase<3> &,
                              const Point<3>                      &,
                              double                              &);
template void mpi_point_value(const RMHD::Entities::EntityBase<2> &,
                              const Point<2>                      &,
                              Vector<double>                      &);
template void mpi_point_value(const RMHD::Entities::EntityBase<3> &,
                              const Point<3>                      &,
                              Vector<double>                      &);