#include <rotatingMHD/benchmark_data.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{
  using namespace dealii;
namespace BenchmarkData
{

template <int dim>
DFG<dim>::DFG()
:
characteristic_length(0.1),
kinematic_viscosity(0.001),
Re(characteristic_length * mean_velocity / kinematic_viscosity),
front_evaluation_point(0.15 / characteristic_length,
                       0.20 / characteristic_length),
rear_evaluation_point(0.25 / characteristic_length,
                      0.20 / characteristic_length),
pressure_difference(0),
drag_force(0),
drag_coefficient(0),
lift_force(0),
lift_coefficient(0)
{
  data_table.declare_column("n");
  data_table.declare_column("t");
  data_table.declare_column("dp");
  data_table.declare_column("C_d");
  data_table.declare_column("C_l");

  data_table.set_scientific("t", true);
  data_table.set_scientific("dp", true);
  data_table.set_scientific("C_d", true);
  data_table.set_scientific("C_l", true);

  data_table.set_precision("t", 6);
  data_table.set_precision("dp", 6);
  data_table.set_precision("C_d", 6);
  data_table.set_precision("C_l", 6);
}

template <int dim>
void DFG<dim>::compute_pressure_difference
(const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure)
{
  const double front_point_pressure_value
  = pressure->point_value(front_evaluation_point);

  const double rear_point_pressure_value
  = pressure->point_value(rear_evaluation_point);

  pressure_difference = front_point_pressure_value - 
                        rear_point_pressure_value;
}

template <int dim>
void DFG<dim>::compute_drag_and_lift_forces_and_coefficients
(const std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
 const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure,
 const types::boundary_id                            cylinder_boundary_id)
{

  AssertDimension(dim, 2);

  const MappingQ<dim> mapping(3);

  /*! @attention What would be the polynomial degree of the normal
      vector? */
  const int face_p_degree = 2 * velocity->fe_degree;

  const QGauss<dim-1>   face_quadrature_formula(
                            std::ceil(0.5 * double(face_p_degree + 1)));

  FEFaceValues<dim> velocity_face_fe_values(mapping,
                                            velocity->fe,
                                            face_quadrature_formula,
                                            update_values |
                                            update_gradients |
                                            update_JxW_values |
                                            update_normal_vectors);

  FEFaceValues<dim> pressure_face_fe_values(mapping,
                                            pressure->fe,
                                            face_quadrature_formula,
                                            update_values);

  const unsigned int n_face_q_points = face_quadrature_formula.size();

  const FEValuesExtractors::Vector  velocities(0);

  std::vector<double>         pressure_values(n_face_q_points);
  std::vector<Tensor<1, dim>> normal_vectors(n_face_q_points);
  std::vector<Tensor<2, dim>> velocity_gradients(n_face_q_points);

  Tensor<1, dim>              forces;

  for (const auto &cell : (velocity->dof_handler)->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
          {
            velocity_face_fe_values.reinit(cell, face);

            typename DoFHandler<dim>::active_cell_iterator pressure_cell(
                              &velocity->get_triangulation(), 
                              cell->level(), 
                              cell->index(), 
                              //Pointer to the pressure's DoFHandler
                              pressure->dof_handler.get());
            typename DoFHandler<dim>::active_face_iterator pressure_face(
                              &velocity->get_triangulation(), 
                              face->level(), 
                              face->index(), 
                              //Pointer to the pressure's DoFHandler
                              pressure->dof_handler.get());

            pressure_face_fe_values.reinit(pressure_cell, pressure_face);

            velocity_face_fe_values[velocities].get_function_gradients(
                                                  velocity->solution,
                                                  velocity_gradients);
            pressure_face_fe_values.get_function_values(
                                                  pressure->solution,
                                                  pressure_values);
            normal_vectors = velocity_face_fe_values.get_normal_vectors();

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              /* The sign inversion here is due to how the normal
              vector is defined in the benchmark */
              forces += (- 1.0 / Re *
                        (normal_vectors[q] * 
                        velocity_gradients[q]
                        +
                        velocity_gradients[q] *
                        normal_vectors[q])
                        +
                        pressure_values[q] *
                        normal_vectors[q]) *
                        velocity_face_fe_values.JxW(q);
            }
          }

  forces = Utilities::MPI::sum(forces, MPI_COMM_WORLD);

  drag_force            = forces[0];
  drag_coefficient      = 2.0 * drag_force;
  lift_force            = forces[1];
  lift_coefficient      = 2.0 * lift_force;
}

template <int dim>
void DFG<dim>::update_table(DiscreteTime  &time)
{
  data_table.add_value("n",   time.get_step_number());
  data_table.add_value("t",   time.get_current_time());
  data_table.add_value("dp",  pressure_difference);
  data_table.add_value("C_d", drag_coefficient);
  data_table.add_value("C_l", lift_coefficient);
}

template <int dim>
void DFG<dim>::print_step_data(DiscreteTime &time)
{
  ConditionalOStream    pcout(std::cout, 
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << "Step = " 
        << std::setw(4) 
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
void DFG<dim>::write_table_to_file(const std::string  &file)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ofstream out_file(file);
    data_table.write_text(
      out_file, 
      TableHandler::TextOutputFormat::org_mode_table);
    out_file.close();
  }
}

} // namespace BenchmarkData
} // namespace RMHD

// explicit instantiations
template struct RMHD::BenchmarkData::DFG<2>;
template struct RMHD::BenchmarkData::DFG<3>;
