#include <rotatingMHD/benchmark_data.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <ostream>
namespace RMHD
{
  using namespace dealii;
namespace BenchmarkData
{

template <int dim>
DFG<dim>::DFG()
:
density(1.0),
characteristic_length(0.1),
mean_velocity(1.0),
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
void DFG<dim>::compute_drag_and_lift_forces_and_coefficients(
  const std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
  const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure)
{
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
        if (face->at_boundary() && face->boundary_id() == 2)
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

template <int dim>
MIT<dim>::MIT(
  std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
  std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
  std::shared_ptr<Entities::ScalarEntity<dim>>  &temperature,
  TimeDiscretization::VSIMEXMethod              &time_stepping,
  const std::shared_ptr<Mapping<dim>>           external_mapping,
  const std::shared_ptr<ConditionalOStream>     external_pcout,
  const std::shared_ptr<TimerOutput>            external_timer)
:
mpi_communicator(velocity->mpi_communicator),
time_stepping(time_stepping),
velocity(velocity),
pressure(pressure),
temperature(temperature),
pressure_differences(3)
{
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != nullptr,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  
  // Initiating the probing sample_points.
  sample_points.emplace_back(0.1810, 7.3700);
  sample_points.emplace_back(0.8190, 0.6300);
  sample_points.emplace_back(0.1810, 0.6300);
  sample_points.emplace_back(0.8190, 7.3700);
  sample_points.emplace_back(0.1810, 4.0000);

  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));
}

template <int dim>
void MIT<dim>::compute_benchmark_data()
{
  compute_point_data();
  compute_wall_data();
  compute_global_data();
}

template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const MIT<dim> &mit)
{
  stream << std::noshowpos << std::scientific
         << "u_1 = "
         << mit.velocity_at_p1[0]
         << ", T_1 = "
         << mit.temperature_at_p1
         << ", p_14 = "
         << mit.pressure_differences[0];

  return stream;
}

template <int dim>
void MIT<dim>::compute_point_data()
{
  TimerOutput::Scope  t(*computing_timer, "MIT Benchmark: Point data");

  // Obtaining data at sample point 1
  velocity_at_p1        = velocity->point_value(sample_points[0]);
  temperature_at_p1     = temperature->point_value(sample_points[0]);
  stream_function_at_p1 = 0.0/*compute_stream_function()*/;
  vorticity_at_p1       = 0.0/*compute_vorticity()*/;

  // Computing skewness metric
  const double temperature_2 = temperature->point_value(sample_points[1]);

  skewness_metric = temperature_at_p1 - temperature_2;

  // Computing pressure differences
  const double pressure_1 = pressure->point_value(sample_points[0]);
  const double pressure_3 = pressure->point_value(sample_points[2]);
  const double pressure_4 = pressure->point_value(sample_points[3]);
  const double pressure_5 = pressure->point_value(sample_points[4]);

  pressure_differences[0] = pressure_1 - pressure_4;
  pressure_differences[1] = pressure_5 - pressure_1;
  pressure_differences[2] = pressure_3 - pressure_5;
}

template <int dim>
void MIT<dim>::compute_wall_data()
{
  TimerOutput::Scope  t(*computing_timer, "MIT Benchmark: Wall data");

  /*! @attention What would be the polynomial degree of the normal
      vector? */
  // Polynomial degree of the integrand    
  const int face_p_degree = temperature->fe_degree;

  // Quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(
                            std::ceil(0.5 * double(face_p_degree + 1)));

  // Finite element values
  FEFaceValues<dim> face_fe_values(*mapping,
                                   temperature->fe,
                                   face_quadrature_formula,
                                   update_gradients |
                                   update_JxW_values |
                                   update_normal_vectors);

  // Number of quadrature points
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<Tensor<1, dim>> temperature_gradients(n_face_q_points);
  std::vector<Tensor<1, dim>> normal_vectors(n_face_q_points);

  // Initiate the local integral value and at each wall.
  double local_boundary_intregral = 0.0;
  double left_boundary_intregral  = 0.0;
  double right_boundary_intregral = 0.0;

  for (const auto &cell : (temperature->dof_handler)->active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->boundary_id() == 0 || face->boundary_id() == 1)
          {
            // Initialize the finite element values
            face_fe_values.reinit(cell, face);

            // Get the temperature gradients at the quadrature points
            face_fe_values.get_function_gradients(temperature->solution,
                                                  temperature_gradients);
            
            // Get the normal vectors at the quadrature points
            normal_vectors = face_fe_values.get_normal_vectors();

            // Reset local face integral values
            local_boundary_intregral = 0.0;

            // Numerical integration
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              local_boundary_intregral += 
                face_fe_values.JxW(q) *     // JxW
                temperature_gradients[q] *  // grad T 
                normal_vectors[q];          // n
            
            // Add the local boundary integral to the respective
            // global boundary integral
            if (face->boundary_id() == 0)
              left_boundary_intregral   += local_boundary_intregral;
            else
              right_boundary_intregral  += local_boundary_intregral;
          }

  // Gather the values of each processor
  left_boundary_intregral   = Utilities::MPI::sum(left_boundary_intregral, 
                                                   mpi_communicator);
  right_boundary_intregral  = Utilities::MPI::sum(right_boundary_intregral, 
                                                  mpi_communicator);
  
  //Compute the Nusselt numbers
  Nusselt_numbers = std::make_pair(left_boundary_intregral/8.0,
                                   right_boundary_intregral/8.0);
}

template <int dim>
void MIT<dim>::compute_global_data()
{
  TimerOutput::Scope  t(*computing_timer, "MIT Benchmark: Global data");

  average_velocity_metric   = 0.0;
  average_vorticity_metric  = 0.0;
}

} // namespace BenchmarkData
} // namespace RMHD

// explicit instantiations
template struct RMHD::BenchmarkData::DFG<2>;
template struct RMHD::BenchmarkData::DFG<3>;

template class RMHD::BenchmarkData::MIT<2>;
template class RMHD::BenchmarkData::MIT<3>;

template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::MIT<2> &);
template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::MIT<3> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::MIT<2> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::MIT<3> &);