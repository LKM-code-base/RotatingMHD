
#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <deal.II/base/discrete_time.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/trilinos_vector.h>

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
  double        front_point_pressure_value;
  double        rear_point_pressure_value;
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
  bool          flag_min_reached;

  DFG();
  void compute_pressure_difference(
              const DoFHandler<dim>               &pressure_dof_handler,
              const TrilinosWrappers::MPI::Vector &pressure_n);
  void compute_drag_and_lift_forces_and_coefficients(
  const parallel::distributed::Triangulation<dim> &triangulation,
  const FESystem<dim>                             &velocity_fe,
  const unsigned int                              &velocity_fe_degree,
  const DoFHandler<dim>                           &velocity_dof_handler,
  const TrilinosWrappers::MPI::Vector             &velocity_n,
  const FE_Q<dim>                                 &pressure_fe,
  const DoFHandler<dim>                           &pressure_dof_handler,
  const TrilinosWrappers::MPI::Vector             &pressure_n,
  const double                                    &Re);
  void compute_periodic_parameters();
  void update_table(const unsigned int    &step,
                    DiscreteTime          &time);
  void print_step_data(const unsigned int &step,
                       DiscreteTime       &time);
  void print_table();
  void write_table_to_file(const std::string &file);
};

} // namespace BenchmarkData
} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */