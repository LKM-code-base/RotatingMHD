#include <rotatingMHD/benchmark_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
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
              const DoFHandler<dim>               &pressure_dof_handler,
              const TrilinosWrappers::MPI::Vector &pressure_n)
{  
  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> front_cell =
    GridTools::find_active_cell_around_point(
                                    StaticMappingQ1<dim, dim>::mapping,
                                    pressure_dof_handler, 
                                    front_evaluation_point);

  const std::pair<typename DoFHandler<dim>::active_cell_iterator,
                  Point<dim>> rear_cell =
    GridTools::find_active_cell_around_point(
                                    StaticMappingQ1<dim, dim>::mapping,
                                    pressure_dof_handler, 
                                    rear_evaluation_point);
  
  front_point_pressure_value  = 0.;
  rear_point_pressure_value   = 0.;

  if (front_cell.first->is_locally_owned())
  {
    front_point_pressure_value = VectorTools::point_value(
                                                pressure_dof_handler,
                                                pressure_n,
                                                front_evaluation_point);
  }
  if (rear_cell.first->is_locally_owned())
  {
    rear_point_pressure_value = VectorTools::point_value(
                                                pressure_dof_handler,
                                                pressure_n,
                                                rear_evaluation_point);
  }

  front_point_pressure_value = 
                        Utilities::MPI::sum(front_point_pressure_value,
                                            MPI_COMM_WORLD);
  rear_point_pressure_value = 
                        Utilities::MPI::sum(rear_point_pressure_value,
                                            MPI_COMM_WORLD);

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
  const parallel::distributed::Triangulation<dim> &triangulation,
  const FESystem<dim>                             &velocity_fe,
  const unsigned int                              &velocity_fe_degree,
  const DoFHandler<dim>                           &velocity_dof_handler,
  const TrilinosWrappers::MPI::Vector             &velocity_n,
  const FE_Q<dim>                                 &pressure_fe,
  const DoFHandler<dim>                           &pressure_dof_handler,
  const TrilinosWrappers::MPI::Vector             &pressure_n,
  const double                                    &Re)
{
  QGauss<dim-1>     face_quadrature_formula(velocity_fe_degree + 1);
  FEFaceValues<dim> velocity_face_fe_values(velocity_fe,
                                            face_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values |
                                            update_normal_vectors);
  FEFaceValues<dim> pressure_face_fe_values(pressure_fe,
                                            face_quadrature_formula,
                                            update_values);

  const unsigned int n_face_q_points        = 
                                        face_quadrature_formula.size();
  const FEValuesExtractors::Vector          velocity(0);

  std::vector<double>         pressure_n_values(n_face_q_points);
  std::vector<Tensor<1, dim>> normal_vectors(n_face_q_points);
  std::vector<Tensor<2, dim>> velocity_n_gradients(n_face_q_points);

  Tensor<1, dim>              forces;

  for (const auto &cell : velocity_dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == 2)
          {
            velocity_face_fe_values.reinit(cell, face);

            typename DoFHandler<dim>::active_cell_iterator pressure_cell(
                                                  &triangulation, 
                                                  cell->level(), 
                                                  cell->index(), 
                                                  &pressure_dof_handler);
            typename DoFHandler<dim>::active_face_iterator pressure_face(
                                                  &triangulation, 
                                                  face->level(), 
                                                  face->index(), 
                                                  &pressure_dof_handler);

            pressure_face_fe_values.reinit(pressure_cell, pressure_face);

            velocity_face_fe_values[velocity].get_function_gradients(
                                                  velocity_n,
                                                  velocity_n_gradients);
            pressure_face_fe_values.get_function_values(
                                                  pressure_n,
                                                  pressure_n_values);
            normal_vectors = velocity_face_fe_values.get_normal_vectors();

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                /* The sign inversion here is due to how the normal
                vector is defined in the benchmark */
                forces += (-0.001 * 
                          normal_vectors[q] * 
                          velocity_n_gradients[q]
                          +
                          pressure_n_values[q] *
                          normal_vectors[q]) *
                          velocity_face_fe_values.JxW(q);
              }
          }

  forces = Utilities::MPI::sum(forces, MPI_COMM_WORLD);

  drag_force        = forces[0];
  drag_coefficient  = 2.0 / 
                      (mean_velocity * 
                      mean_velocity * 
                      characteristic_length) *
                      drag_force;
  lift_force        = forces[1];
  lift_coefficient  = 2.0 / 
                      (mean_velocity * 
                      mean_velocity * 
                      characteristic_length) *
                      lift_force;
}

template <int dim>
void DFG<dim>::update_table(const unsigned int  &step,
                            DiscreteTime        &time)
{
  data_table.add_value("n", step);
  data_table.add_value("t", time.get_current_time());
  data_table.add_value("dp", pressure_difference);
  data_table.add_value("C_d", drag_coefficient);
  data_table.add_value("C_l", lift_coefficient);
}

template <int dim>
void DFG<dim>::print_step_data(const unsigned int &step,
                               DiscreteTime       &time)
{
  ConditionalOStream    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << "Step = " 
        << std::setw(2) 
        << step 
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
void DFG<dim>::write_table_to_file(const std::string &file)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ofstream out_file(file);
    data_table.write_tex(out_file);
    out_file.close();
  }
}

} // namespace BenchmarkData
} // namespace Step35

// explicit instantiations
template struct Step35::BenchmarkData::DFG<2>;
template struct Step35::BenchmarkData::DFG<3>;