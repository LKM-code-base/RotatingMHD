#include <rotatingMHD/benchmark_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <limits>
namespace RMHD
{
  using namespace dealii;
namespace BenchmarkData
{

template <int dim>
DFG<dim>::DFG()
  : characteristic_length(0.1),
    mean_velocity(1.0),
    kinematic_viscosity(0.001),
    front_evaluation_point(0.15 / characteristic_length, 
                           0.20 / characteristic_length),
    rear_evaluation_point(0.25 / characteristic_length, 
                          0.20 / characteristic_length),
    pressure_difference(0),
    front_point_pressure_value(0),
    rear_point_pressure_value(0),
    drag_force(0),
    drag_coefficient(0),
    max_drag_coefficient(-std::numeric_limits<double>::max()),
    min_drag_coefficient(std::numeric_limits<double>::max()),
    amp_drag_coefficient(0),
    mean_drag_coefficient(0),
    lift_force(0),
    lift_coefficient(0),
    old_lift_coefficient(0),
    max_lift_coefficient(-std::numeric_limits<double>::max()),
    min_lift_coefficient(std::numeric_limits<double>::max()),
    amp_lift_coefficient(0),
    mean_lift_coefficient(0),
    frequency(0),
    strouhal_number(0),
    flag_min_reached(false)
  {}

template <int dim>
void DFG<dim>::compute_pressure_difference(
                            const Entities::ScalarEntity<dim> &pressure)
{
  front_point_pressure_value = 0.; 
  rear_point_pressure_value = 0.;
  mpi_point_value(pressure,
                  front_evaluation_point,
                  front_point_pressure_value);
  mpi_point_value(pressure,
                  rear_evaluation_point,
                  rear_point_pressure_value); 
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
void DFG<dim>::compute_drag_and_lift_forces_and_coefficients(
                          const Entities::VectorEntity<dim> &velocity,
                          const Entities::ScalarEntity<dim> &pressure)
{
  QGauss<dim-1>     face_quadrature_formula(velocity.fe_degree + 1);
  FEFaceValues<dim> velocity_face_fe_values(velocity.fe,
                                            face_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values |
                                            update_normal_vectors);
  FEFaceValues<dim> pressure_face_fe_values(pressure.fe,
                                            face_quadrature_formula,
                                            update_values);

  const unsigned int n_face_q_points        = 
                                        face_quadrature_formula.size();
  const FEValuesExtractors::Vector          velocities(0);

  std::vector<double>         pressure_values(n_face_q_points);
  std::vector<Tensor<1, dim>> normal_vectors(n_face_q_points);
  std::vector<Tensor<2, dim>> velocity_gradients(n_face_q_points);

  Tensor<1, dim>              forces;

  for (const auto &cell : velocity.dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == 2)
          {
            velocity_face_fe_values.reinit(cell, face);

            typename DoFHandler<dim>::active_cell_iterator pressure_cell(
                              &velocity.dof_handler.get_triangulation(), 
                              cell->level(), 
                              cell->index(), 
                              &pressure.dof_handler);
            typename DoFHandler<dim>::active_face_iterator pressure_face(
                              &velocity.dof_handler.get_triangulation(), 
                              face->level(), 
                              face->index(), 
                              &pressure.dof_handler);

            pressure_face_fe_values.reinit(pressure_cell, pressure_face);

            velocity_face_fe_values[velocities].get_function_gradients(
                                                  velocity.solution,
                                                  velocity_gradients);
            pressure_face_fe_values.get_function_values(
                                                  pressure.solution,
                                                  pressure_values);
            normal_vectors = velocity_face_fe_values.get_normal_vectors();

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              /* The sign inversion here is due to how the normal
              vector is defined in the benchmark */
              forces += (-kinematic_viscosity *
                        mean_velocity *
                        normal_vectors[q] * 
                        velocity_gradients[q]
                        +
                        characteristic_length *
                        pressure_values[q] *
                        normal_vectors[q]) *
                        velocity_face_fe_values.JxW(q);
            }
          }

  forces = Utilities::MPI::sum(forces, MPI_COMM_WORLD);

  drag_force            = forces[0];
  drag_coefficient      = 2.0 / 
                          (mean_velocity * 
                          mean_velocity * 
                          characteristic_length) *
                          drag_force;
  lift_force            = forces[1];
  old_lift_coefficient  = lift_coefficient;
  lift_coefficient      = 2.0 / 
                          (mean_velocity * 
                          mean_velocity * 
                          characteristic_length) *
                          lift_force;

  if ((lift_coefficient - old_lift_coefficient) > 0.)
    flag_min_reached = true;
}

template <int dim>
void DFG<dim>::update_table(DiscreteTime  &time)
{
  data_table.add_value("n", time.get_step_number());
  data_table.add_value("t", time.get_current_time());
  data_table.add_value("dp", pressure_difference);
  data_table.add_value("C_d", drag_coefficient);
  data_table.add_value("C_l", lift_coefficient);
}

template <int dim>
void DFG<dim>::print_step_data(DiscreteTime &time)
{
  ConditionalOStream    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << "Step = " 
        << std::setw(2) 
        << time.get_step_number() 
        << " Time = " 
        << std::noshowpos << std::scientific
        << time.get_current_time()
        << " dp = " 
        << std::showpos << std::scientific
        << pressure_difference
        << " C_d = "
        << std::showpos << std::scientific
        << drag_coefficient
        << " C_l = " 
        << std::showpos << std::scientific
        << lift_coefficient << std::endl;
}

template <int dim>
void DFG<dim>::print_table()
{
  ConditionalOStream    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << std::endl;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    data_table.write_text(std::cout);
  pcout << std::endl;
}

template <int dim>
void DFG<dim>::write_table_to_file(const std::string  &file)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ofstream out_file(file);
    data_table.write_tex(out_file);
    out_file.close();
  }
}

} // namespace BenchmarkData
} // namespace RMHD

// explicit instantiations
template struct RMHD::BenchmarkData::DFG<2>;
template struct RMHD::BenchmarkData::DFG<3>;