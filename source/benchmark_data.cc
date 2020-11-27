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
DFGBechmarkRequest<dim>::DFGBechmarkRequest
(const double reynolds_number,
 const double reference_length)
:
Re(reynolds_number),
front_evaluation_point(0.15 / reference_length,
                       0.20 / reference_length),
rear_evaluation_point(0.25 / reference_length,
                      0.20 / reference_length),
pressure_difference(0),
drag_coefficient(0),
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
void DFGBechmarkRequest<dim>::compute_pressure_difference
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
void DFGBechmarkRequest<dim>::compute_drag_and_lift_coefficients
(const std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
 const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure,
 const types::boundary_id                            cylinder_boundary_id)
{

  AssertDimension(dim, 2);

  const MappingQ<dim> mapping(3);

  const int face_p_degree = velocity->fe_degree;

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
    if (cell->is_locally_owned() && cell->at_boundary() )
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
          {
            velocity_face_fe_values.reinit(cell, face);

            typename DoFHandler<dim>::active_cell_iterator pressure_cell(
                              &velocity->get_triangulation(), 
                              cell->level(), 
                              cell->index(), 
                              // pointer to the pressure's DoFHandler
                              pressure->dof_handler.get());

            typename DoFHandler<dim>::active_face_iterator pressure_face(
                              &velocity->get_triangulation(), 
                              face->level(), 
                              face->index(), 
                              // pointer to the pressure's DoFHandler
                              pressure->dof_handler.get());

            pressure_face_fe_values.reinit(pressure_cell, pressure_face);

            velocity_face_fe_values[velocities].get_function_gradients
            (velocity->solution,
             velocity_gradients);

            pressure_face_fe_values.get_function_values
            (pressure->solution,
             pressure_values);

            normal_vectors = velocity_face_fe_values.get_normal_vectors();

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              /*
               * The reversed signs here are due to how the normal
               * vector is defined in the DFG benchmark.
               *
               */
              forces += (- 1.0 / Re *
                        (normal_vectors[q] * 
                        velocity_gradients[q]
                        +
                        velocity_gradients[q] *
                        normal_vectors[q])
                        +
                        pressure_values[q] *
                        normal_vectors[q] ) *
                        velocity_face_fe_values.JxW(q);
            }
          }

  forces = Utilities::MPI::sum(forces, MPI_COMM_WORLD);

  drag_coefficient = 2.0 * forces[0];
  lift_coefficient = 2.0 * forces[1];
}

template <int dim>
void DFGBechmarkRequest<dim>::update_table(DiscreteTime  &time)
{
  data_table.add_value("n",   time.get_step_number());
  data_table.add_value("t",   time.get_current_time());

  data_table.add_value("dp",  pressure_difference);

  data_table.add_value("C_d", drag_coefficient);
  data_table.add_value("C_l", lift_coefficient);
}

template <int dim>
void DFGBechmarkRequest<dim>::print_step_data(DiscreteTime &time)
{
  ConditionalOStream  pcout(std::cout,
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
void DFGBechmarkRequest<dim>::write_table_to_file(const std::string  &file)
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
  const std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
  const std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
  const std::shared_ptr<Entities::ScalarEntity<dim>>  &temperature,
  TimeDiscretization::VSIMEXMethod                    &time_stepping,
  const std::shared_ptr<Mapping<dim>>                 external_mapping,
  const std::shared_ptr<ConditionalOStream>           external_pcout,
  const std::shared_ptr<TimerOutput>                  external_timer)
:
mpi_communicator(velocity->mpi_communicator),
time_stepping(time_stepping),
velocity(velocity),
pressure(pressure),
temperature(temperature),
pressure_differences(3),
width(1.0),
height(8.0),
area(8.0)
{
  AssertDimension(dim, 2); 
  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(pressure.get() != nullptr,
         ExcMessage("The pressure's shared pointer has not be"
                    " initialized."));
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));
  
  // Initiating the sample points.
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

  // Setting up columns
  data.declare_column("time");
  data.declare_column("velocity_x_1");
  data.declare_column("velocity_y_1");
  data.declare_column("temperature_1");
  data.declare_column("streamfunction_1");
  data.declare_column("vorticity_1");
  data.declare_column("skewness");
  data.declare_column("dpressure_14");
  data.declare_column("dpressure_51");
  data.declare_column("dpressure_35");
  data.declare_column("Nu_left_wall");
  data.declare_column("Nu_right_wall");
  data.declare_column("average_velocity_metric");
  data.declare_column("average_vorticity_metric");

  // Setting all columns to scientific notation
  data.set_scientific("time", true);
  data.set_scientific("velocity_x_1", true);
  data.set_scientific("velocity_y_1", true);
  data.set_scientific("temperature_1", true);
  data.set_scientific("streamfunction_1", true);
  data.set_scientific("vorticity_1", true);
  data.set_scientific("skewness", true);
  data.set_scientific("dpressure_14", true);
  data.set_scientific("dpressure_51", true);
  data.set_scientific("dpressure_35", true);
  data.set_scientific("Nu_left_wall", true);
  data.set_scientific("Nu_right_wall", true);
  data.set_scientific("average_velocity_metric", true);
  data.set_scientific("average_vorticity_metric", true);

  // Setting columns' precision
  data.set_precision("time", 6);
  data.set_precision("velocity_x_1", 6);
  data.set_precision("velocity_y_1", 6);
  data.set_precision("temperature_1", 6);
  data.set_precision("streamfunction_1", 6);
  data.set_precision("vorticity_1", 6);
  data.set_precision("skewness", 6);
  data.set_precision("dpressure_14", 6);
  data.set_precision("dpressure_51", 6);
  data.set_precision("dpressure_35", 6);
  data.set_precision("Nu_left_wall", 6);
  data.set_precision("Nu_right_wall", 6);
  data.set_precision("average_velocity_metric", 6);
  data.set_precision("average_vorticity_metric", 6);
}

template <int dim>
void MIT<dim>::compute_benchmark_data()
{
  // Compute benchmark data
  compute_point_data();
  compute_wall_data();
  compute_global_data();

  // Update column's values
  data.add_value("time", time_stepping.get_current_time());
  data.add_value("velocity_x_1", velocity_at_p1[0]);
  data.add_value("velocity_y_1", velocity_at_p1[1]);
  data.add_value("temperature_1", temperature_at_p1);
  data.add_value("streamfunction_1", stream_function_at_p1);
  data.add_value("vorticity_1", vorticity_at_p1);
  data.add_value("skewness", skewness_metric);
  data.add_value("dpressure_14", pressure_differences[0]);
  data.add_value("dpressure_51", pressure_differences[1]);
  data.add_value("dpressure_35", pressure_differences[2]);
  data.add_value("Nu_left_wall", nusselt_numbers.first);
  data.add_value("Nu_right_wall", nusselt_numbers.second);
  data.add_value("average_velocity_metric", average_velocity_metric);
  data.add_value("average_vorticity_metric", average_vorticity_metric);
}

template <int dim>
void MIT<dim>::print_data_to_file(std::string file_name)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_name += ".txt";

    std::ofstream file(file_name);

    data.write_text(
      file,
      TableHandler::TextOutputFormat::org_mode_table);
  }
}


template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const MIT<dim> &mit)
{
  stream << std::noshowpos << std::scientific
         << "ux_1 = "
         << mit.velocity_at_p1[0]
         << ", T_1 = "
         << mit.temperature_at_p1
         << ", p_14 = "
         << mit.pressure_differences[0]
         << ", Nu = "
         << mit.nusselt_numbers.first
         << ", u = "
         << mit.average_velocity_metric
         << ", w = "
         << mit.average_vorticity_metric;

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
  const double temperature_at_p2 = temperature->point_value(sample_points[1]);

  skewness_metric = temperature_at_p1 + temperature_at_p2;

  // Computing pressure differences
  const double pressure_at_p1 = pressure->point_value(sample_points[0]);
  const double pressure_at_p3 = pressure->point_value(sample_points[2]);
  const double pressure_at_p4 = pressure->point_value(sample_points[3]);
  const double pressure_at_p5 = pressure->point_value(sample_points[4]);

  pressure_differences[0] = pressure_at_p1 - pressure_at_p4;
  pressure_differences[1] = pressure_at_p5 - pressure_at_p1;
  pressure_differences[2] = pressure_at_p3 - pressure_at_p5;
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
        if (face->boundary_id() == 1 || face->boundary_id() == 2)
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
                temperature_gradients[q] *  // grad T 
                normal_vectors[q] *         // n
                face_fe_values.JxW(q);      // da
          
            // Add the local boundary integral to the respective
            // global boundary integral
            if (face->boundary_id() == 1)
              left_boundary_intregral   += local_boundary_intregral;
            else
              right_boundary_intregral  += local_boundary_intregral;
          }

  // Gather the values of each processor
  left_boundary_intregral   = Utilities::MPI::sum(left_boundary_intregral, 
                                                   mpi_communicator);
  right_boundary_intregral  = Utilities::MPI::sum(right_boundary_intregral, 
                                                  mpi_communicator);
  
  //Compute and store the Nusselt numbers of the walls
  nusselt_numbers = std::make_pair(left_boundary_intregral/height,
                                   right_boundary_intregral/height);
}

template <int dim>
void MIT<dim>::compute_global_data()
{
  TimerOutput::Scope  t(*computing_timer, "MIT Benchmark: Global data");

  // Defining the type to contain the vorticity during assembly
  using CurlType = typename FEValuesViews::Vector< dim >::curl_type;

  // Initiate the average velocity and vorticity metrics
  average_velocity_metric   = 0.0;
  average_vorticity_metric  = 0.0;

  // Polynomial degree of the integrand    
  const int p_degree = 2 * velocity->fe_degree;

  // Quadrature formula
  const QGauss<dim>   quadrature_formula(
                            std::ceil(0.5 * double(p_degree + 1)));

  // Finite element values
  FEValues<dim> fe_values(*mapping,
                          velocity->fe,
                          quadrature_formula,
                          update_values |
                          update_gradients |
                          update_JxW_values);

  // Number of quadrature points
  const unsigned int n_q_points = quadrature_formula.size();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  std::vector<CurlType> vorticity_values(n_q_points);

  for (const auto &cell : (velocity->dof_handler)->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      // Initialize the finite element values
      fe_values.reinit(cell);

      // Define the vector extractor
      const FEValuesExtractors::Vector  velocities(0);

      // Get the velocity and vorticity values at each quadrature
      // point
      fe_values[velocities].get_function_values(velocity->solution,
                                                velocity_values);
      fe_values[velocities].get_function_curls(velocity->solution,
                                               vorticity_values);

      // Numerical integration (Loop over all quadrature points)
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        average_velocity_metric += 
          velocity_values[q] *  // v
          velocity_values[q] *  // v
          fe_values.JxW(q);     // dv
        average_vorticity_metric +=
          vorticity_values[q] * // curl v
          vorticity_values[q] * // curl v
          fe_values.JxW(q);     // dv
      }
    }

  // Gather the values of each processor
  average_velocity_metric   = Utilities::MPI::sum(average_velocity_metric, 
                                                  mpi_communicator);
  average_vorticity_metric  = Utilities::MPI::sum(average_vorticity_metric, 
                                                  mpi_communicator);

  // Compute the global averages
  average_velocity_metric   = std::sqrt(average_velocity_metric/(2.0 * area));
  average_vorticity_metric  = std::sqrt(average_vorticity_metric/(2.0 * area));
}

} // namespace BenchmarkData

} // namespace RMHD

// explicit instantiations
template struct RMHD::BenchmarkData::DFGBechmarkRequest<2>;
template struct RMHD::BenchmarkData::DFGBechmarkRequest<3>;

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
