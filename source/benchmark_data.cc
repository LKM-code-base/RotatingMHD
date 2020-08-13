#include <rotatingMHD/benchmark_data.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>

#include <limits>
namespace Step35
{
  using namespace dealii;
namespace BenchmarkData
{

template <int dim>
DFG<dim>::DFG()
  : characteristic_length(0.1),
    mean_velocity(1.0),
    front_evaluation_point(0.15, 0.20),
    rear_evaluation_point(0.25, 0.20),
    pressure_difference(0),
    drag_force(0),
    drag_coefficient(0),
    max_drag_coefficient(-std::numeric_limits<double>::max()),
    min_drag_coefficient(std::numeric_limits<double>::max()),
    amp_drag_coefficient(0),
    mean_drag_coefficient(0),
    lift_force(0),
    lift_coefficient(0),
    max_lift_coefficient(-std::numeric_limits<double>::max()),
    min_lift_coefficient(std::numeric_limits<double>::max()),
    amp_lift_coefficient(0),
    mean_lift_coefficient(0),
    frequency(0),
    strouhal_number(0)
  {}

template <int dim>
void DFG<dim>::compute_pressure_difference(
              const DoFHandler<dim>               &pressure_dof_handler,
              const TrilinosWrappers::MPI::Vector &pressure_n)
{
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> front_point =
    GridTools::find_active_cell_around_point(
                                    StaticMappingQ1<dim, dim>::mapping,
                                    pressure_dof_handler, 
                                    front_evaluation_point);

  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> rear_point =
    GridTools::find_active_cell_around_point(
                                    StaticMappingQ1<dim, dim>::mapping,
                                    pressure_dof_handler, 
                                    rear_evaluation_point);

  ConditionalOStream    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  double front_point_pressure_value;
  double rear_point_pressure_value;
    pcout << pressure_difference << std::endl;
  pcout << front_point_pressure_value << std::endl;
  pcout << rear_point_pressure_value << std::endl;

  if (front_point.first->is_locally_owned())
  {
    front_point_pressure_value = VectorTools::point_value(
                                                pressure_dof_handler,
                                                pressure_n,
                                                front_evaluation_point);
    pcout << front_point_pressure_value << std::endl;
  }

  if (rear_point.first->is_locally_owned())
  {
    rear_point_pressure_value = VectorTools::point_value(
                                                pressure_dof_handler,
                                                pressure_n,
                                                rear_evaluation_point);
    pcout << rear_point_pressure_value << std::endl;
  }

  pressure_difference = front_point_pressure_value - 
                                              rear_point_pressure_value; 
}

template <int dim>
void DFG<dim>::compute_periodic_parameters()
{
  amp_drag_coefficient = max_drag_coefficient - min_drag_coefficient;
  mean_drag_coefficient = 0.5 * 
                          (max_drag_coefficient + min_drag_coefficient);

  amp_lift_coefficient = max_lift_coefficient - min_lift_coefficient;
  mean_lift_coefficient = 0.5 * 
                          (max_lift_coefficient + min_lift_coefficient);

  strouhal_number = characteristic_length * frequency / mean_velocity;
}

template <int dim>
void DFG<dim>::update_table()
{

}
} // namespace TimeDiscretiation
} // namespace Step35

// explicit instantiations
template struct Step35::BenchmarkData::DFG<2>;
template struct Step35::BenchmarkData::DFG<3>;