
#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_handler.h>

#include <iostream>
#include <vector>
namespace Step35
{
  using namespace dealii;

namespace BenchmarkData
{

template <int dim>
struct DFG
{
  double        characteristic_length;
  double        mean_velocity;
  Point<dim>    front_evaluation_point;
  Point<dim>    rear_evaluation_point;
  double        pressure_difference;
  double        drag_force;
  double        drag_coefficient;
  double        max_drag_coefficient;
  double        min_drag_coefficient;
  double        amp_drag_coefficient;
  double        mean_drag_coefficient;
  double        lift_force;
  double        lift_coefficient;
  double        max_lift_coefficient;
  double        min_lift_coefficient;
  double        amp_lift_coefficient;
  double        mean_lift_coefficient;
  double        frequency;
  double        strouhal_number;
  TableHandler  data_table;

  DFG();
  void compute_pressure_difference(
              const DoFHandler<dim>               &pressure_dof_handler,
              const TrilinosWrappers::MPI::Vector &pressure_n);
  void compute_periodic_parameters();
  void update_table();
};

} // namespace BenchmarkData
} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */