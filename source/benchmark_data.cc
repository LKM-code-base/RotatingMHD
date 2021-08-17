
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/exceptions.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/vector_tools.h>

DeclException0(ExcBoostNoConvergence);

#include <boost/math/tools/roots.hpp>

#include <fstream>
#include <ostream>
namespace RMHD
{
  using namespace dealii;
namespace BenchmarkData
{

template <int dim>
DFGBechmarkRequest<dim>::DFGBechmarkRequest(const double reynolds_number)
:
Re(reynolds_number),
front_evaluation_point(1.5,
                       2.0),
rear_evaluation_point(2.5,
                      2.0),
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
(const std::shared_ptr<Entities::FE_ScalarField<dim>> &pressure)
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
(const std::shared_ptr<Entities::FE_VectorField<dim>> &velocity,
 const std::shared_ptr<Entities::FE_ScalarField<dim>> &pressure,
 const types::boundary_id                            cylinder_boundary_id)
{

  AssertDimension(dim, 2);

  const MappingQ<dim> mapping(3);

  const QGauss<dim-1>   face_quadrature_formula(velocity->fe_degree + 1);

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
void DFGBechmarkRequest<dim>::update_table(TimeDiscretization::DiscreteTime  &time)
{
  data_table.add_value("n",   time.get_step_number());
  data_table.add_value("t",   time.get_current_time());

  data_table.add_value("dp",  pressure_difference);

  data_table.add_value("C_d", drag_coefficient);
  data_table.add_value("C_l", lift_coefficient);
}

template <int dim>
void DFGBechmarkRequest<dim>::print_step_data()
{
  ConditionalOStream  pcout(std::cout,
                            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << "    dp = "
        << std::setprecision(6)
	      << std::showpos << std::scientific
        << pressure_difference
        << " C_d = "
        << std::showpos << std::scientific
        << drag_coefficient
        << " C_l = "
        << std::showpos << std::scientific
        << lift_coefficient
        << std::defaultfloat << std::endl;
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
  const std::shared_ptr<Entities::FE_VectorField<dim>>  &velocity,
  const std::shared_ptr<Entities::FE_ScalarField<dim>>  &pressure,
  const std::shared_ptr<Entities::FE_ScalarField<dim>>  &temperature,
  TimeDiscretization::VSIMEXMethod                    &time_stepping,
  const unsigned int                                  left_wall_boundary_id,
  const unsigned int                                  right_wall_boundary_id,
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
area(8.0),
left_wall_boundary_id(left_wall_boundary_id),
right_wall_boundary_id(right_wall_boundary_id)
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
  stream << std::noshowpos << std::scientific << std::setprecision(6)
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

  // Quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(temperature->fe_degree + 1);

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
  double local_boundary_integral = 0.0;
  double left_boundary_integral  = 0.0;
  double right_boundary_integral = 0.0;

  for (const auto &cell : (temperature->dof_handler)->active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            (face->boundary_id() == left_wall_boundary_id ||
             face->boundary_id() == right_wall_boundary_id))
          {
            // Initialize the finite element values
            face_fe_values.reinit(cell, face);

            // Get the temperature gradients at the quadrature points
            face_fe_values.get_function_gradients(temperature->solution,
                                                  temperature_gradients);

            // Get the normal vectors at the quadrature points
            normal_vectors = face_fe_values.get_normal_vectors();

            // Reset local face integral values
            local_boundary_integral = 0.0;

            // Numerical integration
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              local_boundary_integral +=
                temperature_gradients[q] *  // grad T
                normal_vectors[q] *         // n
                face_fe_values.JxW(q);      // da

            // Add the local boundary integral to the respective
            // global boundary integral
            if (face->boundary_id() == left_wall_boundary_id)
              left_boundary_integral   += local_boundary_integral;
            else
              right_boundary_integral  += local_boundary_integral;
          }

  // Gather the values of each processor
  left_boundary_integral   = Utilities::MPI::sum(left_boundary_integral,
                                                   mpi_communicator);
  right_boundary_integral  = Utilities::MPI::sum(right_boundary_integral,
                                                  mpi_communicator);

  //Compute and store the Nusselt numbers of the walls
  nusselt_numbers = std::make_pair(left_boundary_integral/height,
                                   right_boundary_integral/height);
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

  // Quadrature formula
  const QGauss<dim>   quadrature_formula(velocity->fe_degree + 1);

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



template <int dim>
ChristensenBenchmark<dim>::ChristensenBenchmark(
  const std::shared_ptr<Entities::FE_VectorField<dim>>  &velocity,
  const std::shared_ptr<Entities::FE_ScalarField<dim>>  &temperature,
  const std::shared_ptr<Entities::FE_VectorField<dim>>  &magnetic_field,
  const TimeDiscretization::VSIMEXMethod              &time_stepping,
  const RunTimeParameters::DimensionlessNumbers       &dimensionless_numbers,
  const double                                        inner_radius,
  const double                                        outer_radius,
  const unsigned int                                  case_number,
  const std::shared_ptr<Mapping<dim>>                 external_mapping,
  const std::shared_ptr<ConditionalOStream>           external_pcout,
  const std::shared_ptr<TimerOutput>                  external_timer)
:
mpi_communicator(velocity->mpi_communicator),
time_stepping(time_stepping),
velocity(velocity),
temperature(temperature),
magnetic_field(magnetic_field),
dimensionless_numbers(dimensionless_numbers),
case_number(case_number),
sample_point_radius(0.5*(outer_radius + inner_radius)),
sample_point_colatitude(numbers::PI_2),
sample_point_longitude(0.)
{
  Assert(case_number < 3,
         ExcMessage("Only case 0, 1 and 2  are defined."));

  // Temporary asserts
  AssertThrow(case_number == 1 ? false : true,
              ExcMessage("The lines concerning case 1 are currently commented out"));
  AssertThrow(case_number == 2 ? false : true,
              ExcMessage("Case 2 has not being implemented yet"));

  Assert(velocity.get() != nullptr,
         ExcMessage("The velocity's shared pointer has not be"
                    " initialized."));
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  if (case_number != 0)
    Assert(magnetic_field.get() != nullptr,
          ExcMessage("The magnetic field's shared pointer has not be"
                      " initialized."));

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
  data.declare_column("mean_kinetic_energy_density");
  data.declare_column("mean_magnetic_energy_density");
  data.declare_column("sample_point_longitude");
  data.declare_column("temperature_at_sample_point");
  data.declare_column("azimuthal_velocity_at_sample_point");
  data.declare_column("polar_magnetic_field_at_sample_point");
  data.declare_column("drift_frequency");

  // Setting all columns to scientific notation
  data.set_scientific("time", true);
  data.set_scientific("mean_kinetic_energy_density", true);
  data.set_scientific("mean_magnetic_energy_density", true);
  data.set_scientific("sample_point_longitude", true);
  data.set_scientific("temperature_at_sample_point", true);
  data.set_scientific("azimuthal_velocity_at_sample_point", true);
  data.set_scientific("polar_magnetic_field_at_sample_point", true);
  data.set_scientific("drift_frequency", true);

  // Setting columns' precision
  data.set_precision("time", 6);
  data.set_precision("mean_kinetic_energy_density", 6);
  data.set_precision("mean_magnetic_energy_density", 6);
  data.set_precision("sample_point_longitude", 6);
  data.set_precision("temperature_at_sample_point", 6);
  data.set_precision("azimuthal_velocity_at_sample_point", 6);
  data.set_precision("polar_magnetic_field_at_sample_point", 6);
  data.set_precision("drift_frequency", 6);
}



template <int dim>
void ChristensenBenchmark<dim>::compute_benchmark_data()
{
  // Compute benchmark data
  compute_global_data();
  find_sample_point();
  compute_point_data();

  // Update column's values
  data.add_value("time", time_stepping.get_current_time());
  data.add_value("mean_kinetic_energy_density", mean_kinetic_energy_density);
  data.add_value("mean_magnetic_energy_density", mean_magnetic_energy_density);
  data.add_value("sample_point_longitude", sample_point_longitude);
  data.add_value("temperature_at_sample_point", temperature_at_sample_point);
  data.add_value("azimuthal_velocity_at_sample_point", azimuthal_velocity_at_sample_point);
  data.add_value("polar_magnetic_field_at_sample_point", polar_magnetic_field_at_sample_point);
  data.add_value("drift_frequency", drift_frequency);

}



template <int dim>
void ChristensenBenchmark<dim>::print_data_to_file(std::string file_name)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_name += "_case_" + std::to_string(case_number) + ".txt";

    std::ofstream file(file_name);

    data.write_text(
      file,
      TableHandler::TextOutputFormat::org_mode_table);
  }
}



template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const ChristensenBenchmark<dim> &christensen)
{
  switch (christensen.case_number)
  {
    case 0:
      stream << std::noshowpos << std::scientific
             << "E_kin = " << std::setprecision(5)
             << christensen.mean_kinetic_energy_density
             << ", phi = " << std::fixed << std::setprecision(2)
             << (christensen.sample_point_longitude / 2. / numbers::PI * 360.)
             << ", T = " << std::scientific << std::setprecision(4)
             << christensen.temperature_at_sample_point
             << ", v_phi = " << std::setprecision(4)
             << christensen.azimuthal_velocity_at_sample_point
             << ", w = "
             << christensen.drift_frequency
             << std::defaultfloat;
      break;
    case 1:
      stream << std::noshowpos << std::scientific
             << "E_kin = " << std::setprecision(5)
             << christensen.mean_kinetic_energy_density
             << ", E_mag = "  << std::setprecision(5)
             << christensen.mean_magnetic_energy_density
             << ", phi = " << std::fixed << std::setprecision(2)
             << (christensen.sample_point_longitude / 2. / numbers::PI * 360.)
             << ", T = " << std::scientific << std::setprecision(4)
             << christensen.temperature_at_sample_point
             << ", v_phi = " << std::setprecision(4)
             << christensen.azimuthal_velocity_at_sample_point
             << ", B_theta = "
             << christensen.polar_magnetic_field_at_sample_point
             << ", w = "
             << christensen.drift_frequency
             << std::defaultfloat;
      break;
    case 2:
      AssertThrow(false, ExcNotImplemented());
      break;
    default:
      AssertThrow(false, ExcNotImplemented());
      break;
  }

  return stream;
}



template <int dim>
void ChristensenBenchmark<dim>::compute_global_data()
{
  TimerOutput::Scope  t(*computing_timer, "Christensen Benchmark: Global data");

  // Initiate the mean energy densities and the volume
  mean_kinetic_energy_density   = 0.0;
  mean_magnetic_energy_density  = 0.0;
  discrete_volume               = 0.0;
  drift_frequency               = 0.0;

  // Dummy finite element for when case 0 is being computed
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(2), dim);

  // Create pointer to the pertinent finite element
  const FESystem<dim>* const magnetic_field_fe =
              (case_number != 0) ? &magnetic_field->fe : &dummy_fe_system;

  // Polynomial degree of the integrand
  const int p_degree = 2 * (case_number != 0
                              ? std::max(velocity->fe_degree,
                                        magnetic_field->fe_degree)
                              : velocity->fe_degree);

  // Quadrature formula
  const QGauss<dim>   quadrature_formula(velocity->fe_degree + 1);

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::Benchmarks::Christensen::Scratch<dim>  &scratch,
           AssemblyData::Benchmarks::Christensen::Copy          &data)
    {
      this->compute_local_global_squared_norms(cell,
                                               scratch,
                                               data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::Benchmarks::Christensen::Copy  &data)
    {
      this->copy_local_to_global_squared_norms(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              (velocity->dof_handler)->begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              (velocity->dof_handler)->end()),
   worker,
   copier,
   AssemblyData::Benchmarks::Christensen::Scratch<dim>(
    *mapping,
    quadrature_formula,
    velocity->fe,
    update_JxW_values|
    update_values|
    update_quadrature_points,
    *magnetic_field_fe,
    update_values),
   AssemblyData::Benchmarks::Christensen::Copy(velocity->fe.dofs_per_cell));

  // Gather the values of each processor
  mean_kinetic_energy_density = Utilities::MPI::sum(
                                  mean_kinetic_energy_density,
                                  mpi_communicator);
  mean_magnetic_energy_density = Utilities::MPI::sum(
                                  mean_magnetic_energy_density,
                                  mpi_communicator);
  discrete_volume             = Utilities::MPI::sum(
                                  discrete_volume,
                                  mpi_communicator);

  // Compute the mean values
  mean_kinetic_energy_density *= 0.5 / discrete_volume;

  if (case_number != 0)
    mean_magnetic_energy_density *= 0.5 / discrete_volume /
                                    dimensionless_numbers.Ek /
                                    dimensionless_numbers.Pm;
}



template <int dim>
void ChristensenBenchmark<dim>::compute_local_global_squared_norms(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  AssemblyData::Benchmarks::Christensen::Scratch<dim>  &scratch,
  AssemblyData::Benchmarks::Christensen::Copy          &data)
{
  // Reset local data
  data.local_velocity_squared_norm        = 0.;
  data.local_magnetic_field_squared_norm  = 0.;
  data.local_discrete_volume              = 0.;

  // Initialize the finite element values
  scratch.velocity_fe_values.reinit(cell);

  // Define the vector extractor
  const FEValuesExtractors::Vector  vector_extractor(0);

  // Get the velocity values at each quadrature point
  scratch.velocity_fe_values[vector_extractor].get_function_values(
    velocity->solution,
    scratch.velocity_values);

  /*
  // Repeat the steps for the magnetic flux
  if (case_number != 0)
  {
    typename DoFHandler<dim>::active_cell_iterator
    magnetic_flux_cell(&velocity->get_triangulation(),
                        cell->level(),
                        cell->index(),
                        (magnetic_field->dof_handler).get());

    magnetic_flux_fe_values.reinit(magnetic_flux_cell);

    magnetic_flux_fe_values[vector_extractor].get_function_values(
      magnetic_field->solution,
      magnetic_flux_values);
  }
  */

  // Numerical integration (Loop over all quadrature points)
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    data.local_velocity_squared_norm +=
      scratch.velocity_values[q] *  // v
      scratch.velocity_values[q] *  // v
      scratch.velocity_fe_values.JxW(q);     // dv

    /*
    if (case_number != 0)
      data.local_magnetic_field_squared_norm +=
        scratc.hmagnetic_flux_values[q] *  // v
        scratc.hmagnetic_flux_values[q] *  // v
        scratc.hvelocity_fe_values.JxW(q);     // dv
    */

    data.local_discrete_volume +=
      scratch.velocity_fe_values.JxW(q);
  }
}




template <int dim>
void ChristensenBenchmark<dim>::copy_local_to_global_squared_norms(
  const AssemblyData::Benchmarks::Christensen::Copy  &data)
{
  mean_kinetic_energy_density   += data.local_velocity_squared_norm;
  mean_magnetic_energy_density  += data.local_magnetic_field_squared_norm;
  discrete_volume               += data.local_discrete_volume;
}



template <int dim>
void ChristensenBenchmark<dim>::find_sample_point()
{
  TimerOutput::Scope  t(*computing_timer, "Christensen Benchmark: Finding sample point");

  // Set the last known value of the sample point's longitude as initial
  // guess
  if (sample_point_longitude > 0.)
  {
    // Compute the derivative of the radial velocity w.r.t. the
    // longitude. This value is needed for the boost's
    // bracket_and_solve_root method.
    const double gradient_at_trial_point
    = compute_azimuthal_gradient_of_radial_velocity(sample_point_radius,
                                                    sample_point_longitude,
                                                    sample_point_colatitude);

    // Compute a root of the radial velocity w.r.t. the longitude
    const double trial_longitude
    = compute_zero_of_radial_velocity(sample_point_longitude,
                                      gradient_at_trial_point > 0.,
                                      1e-6,
                                      50);

    // Compute the derivative of the radial velocity w.r.t. the
    // longitude at the found root.
    const double gradient_at_zero
    = compute_azimuthal_gradient_of_radial_velocity(sample_point_radius,
                                                    trial_longitude,
                                                    sample_point_colatitude);

    if(gradient_at_zero > 0. &&
        trial_longitude >= 0. &&
        trial_longitude <= 2. * numbers::PI)
    {
        sample_point_longitude = trial_longitude;

        // Compute the position vector of the sample point in cartesian
        // coordinates.
        std::array<double, dim> spherical_coordinates;

        if constexpr(dim == 2)
          spherical_coordinates = {sample_point_radius,
                                  sample_point_longitude};
        else if constexpr(dim == 3)
          spherical_coordinates = {sample_point_radius,
                                  sample_point_longitude,
                                  sample_point_colatitude};

        sample_point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);

        return;
    }
  }

  // The initial guess was not computed or wasn't good enough. A set of
  // trial points well be used to find the sample point.
  const unsigned int n_trial_points = 16;

  std::vector<double> trial_longitudes;

  trial_longitudes.push_back(1e-3 * 2. * numbers::PI / static_cast<double>(n_trial_points));

  for (unsigned int i=1; i<n_trial_points; ++i)
      trial_longitudes.push_back(i * 2. * numbers::PI / static_cast<double>(n_trial_points));

  bool            point_found = false;

  unsigned int    cnt = 0;

  while(cnt < n_trial_points && point_found == false)
  {
      const double gradient_at_trial_point
      = compute_azimuthal_gradient_of_radial_velocity(
          sample_point_radius,
          trial_longitudes[cnt],
          sample_point_colatitude);

      try
      {
          const double tentative_longitude
          = compute_zero_of_radial_velocity(
            trial_longitudes[cnt],
            gradient_at_trial_point > 0.,
            1e-6,
            50);

          const double gradient_at_zero
          = compute_azimuthal_gradient_of_radial_velocity(
              sample_point_radius,
              tentative_longitude,
              sample_point_colatitude);

          if (gradient_at_zero > 0.)
          {
              point_found             = true;
              sample_point_longitude  = tentative_longitude;
          }
          ++cnt;
      }
      catch(ExcBoostNoConvergence &exc)
      {
          ++cnt;
          continue;
      }
      catch (std::exception &exc)
      {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Exception on processing: " << std::endl
                    << exc.what() << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::abort();
      }
      catch (...)
      {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Unknown exception!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::abort();
      }
  }

  if (!point_found)
  {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on find_sample_point!" << std::endl
                << "The algorithm did not find a benchmark point using "
                << n_trial_points << " trial points in [0,2*pi)."
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
  }
  if (sample_point_longitude < 0. || sample_point_longitude > 2. * numbers::PI)
  {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on find_sample_point!" << std::endl
                << "The algorithm did not find a benchmark point in [0,2*pi)."
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
  }

  // Compute the position vector of the sample point in cartesian
  // coordinates.
  std::array<double, dim> spherical_coordinates;

  if constexpr(dim == 2)
    spherical_coordinates = {sample_point_radius,
                             sample_point_longitude};
  else if constexpr(dim == 3)
    spherical_coordinates = {sample_point_radius,
                             sample_point_longitude,
                             sample_point_colatitude};

  sample_point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);
}



template <int dim>
double ChristensenBenchmark<dim>::compute_radial_velocity(
  const double radius,
  const double azimuthal_angle,
  const double polar_angle) const
{
  AssertThrow(radius > 0.,
              ExcLowerRangeType<double>(radius, 0));
  AssertThrow((polar_angle >= 0. && polar_angle <= numbers::PI),
               ExcMessage("The polar angle is not in the range [0,pi]."));

  // The boost's bracket_and_solve_root may produce test for values
  // outside the admissible range of the dealii's from_spherical method.
  // To this end, a temporary angle is introduced and shifted to its
  // admissible value.
  double tmp_azimuthal_angle = azimuthal_angle;

  if (azimuthal_angle < 0.)
      tmp_azimuthal_angle += 2. * numbers::PI;
  else if (azimuthal_angle > 2. * numbers::PI)
      tmp_azimuthal_angle -= 2. * numbers::PI;

  // Define the position vector in cartesian coordinates from the
  // spherical ones.
  std::array<double, dim> spherical_coordinates;

  if constexpr(dim == 2)
  {
    (void)polar_angle;
    spherical_coordinates = {radius, tmp_azimuthal_angle};
  }
  else if constexpr(dim == 3)
    spherical_coordinates = {radius, tmp_azimuthal_angle, polar_angle};

  const Point<dim> point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);

  // Compute the radial velocity at the given spherical coordinates.
  const Tensor<1,dim> local_velocity = velocity->point_value(point, mapping);

  return local_velocity * point / radius;
}



template<int dim>
double  ChristensenBenchmark<dim>::compute_zero_of_radial_velocity(
  const double        phi_guess,
  const bool          local_slope,
  const double        tol,
  const unsigned int  &max_iter) const
{
    using namespace boost::math::tools;
    using namespace GeometryExceptions;

    Assert(tol > 0.0, ExcLowerRangeType<double>(tol, 0));
    Assert(max_iter > 0, ExcLowerRange(max_iter, 0));

    Assert((phi_guess < 0.?
           (phi_guess >= -numbers::PI && phi_guess <= numbers::PI):
           (phi_guess >= 0. && phi_guess <= 2. * numbers::PI)),
           ExcAzimuthalAngleRange(phi_guess));

    auto boost_max_iter = boost::numeric_cast<boost::uintmax_t>(max_iter);

    const double radius = sample_point_radius;
    const double theta  = sample_point_colatitude;

    // Declare lambda functions for the boost's
    // root bracket_and_solve_root method
    auto function = [this,radius,theta](const double &x)
    {
        return compute_radial_velocity(radius, x, theta);
    };
    auto tolerance_criterion = [tol,function](const double &a, const double &b)
    {
        return std::abs(function(a)) <= tol && std::abs(function(b)) <= tol;
    };

    /*! @attention Why initialize phi to this value? */
    double phi = -2. * numbers::PI;

    // Find a root of the radial velocity w.r.t. the longitude
    try
    {
        auto phi_interval
        = bracket_and_solve_root(
                function,
                phi_guess,
                1.05,
                local_slope,
                tolerance_criterion,
                boost_max_iter);
        phi = 0.5 * (phi_interval.first + phi_interval.second);
    }
    catch (std::exception &exc)
    {
        throw ExcBoostNoConvergence();
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
    }

    // The found root may be outside the internal allowed range.
    // The following if scope corrects it by shifting the value inside
    // the range.
    if (phi < 0.)
        phi += 2. * numbers::PI;
    else if (phi > 2. * numbers::PI)
        phi -= 2. * numbers::PI;
    return phi;
}



template <int dim>
double ChristensenBenchmark<dim>::compute_azimuthal_gradient_of_radial_velocity(
  const double radius,
  const double azimuthal_angle,
  const double polar_angle) const
{
  AssertThrow(radius > 0.,
              ExcLowerRangeType<double>(radius, 0));
  AssertThrow((polar_angle >= 0. && polar_angle <= numbers::PI),
               ExcMessage("The polar angle is not in the range [0,pi]."));
  AssertThrow((azimuthal_angle >= 0. && azimuthal_angle < 2. * numbers::PI),
               ExcMessage("The azimuthal angle is not in the range [0,2 pi)."));

  // Define the local basis vectors and the spherical coordinates
  Tensor<1,dim> local_radial_basis_vector;
  Tensor<1,dim> local_azimuthal_basis_vector;

  local_radial_basis_vector[0]    = sin(polar_angle) * cos(azimuthal_angle);
  local_radial_basis_vector[1]    = sin(polar_angle) * sin(azimuthal_angle);

  local_azimuthal_basis_vector[0] = -sin(azimuthal_angle);
  local_azimuthal_basis_vector[1] = cos(azimuthal_angle);

  std::array<double, dim> spherical_coordinates;

  if constexpr(dim == 2)
    spherical_coordinates = {radius, azimuthal_angle};
  else if constexpr(dim == 3)
  {
    spherical_coordinates = {radius, azimuthal_angle, polar_angle};

    local_radial_basis_vector[2] = cos(polar_angle);
  }

  // Compute the derivative of the radial velocity w.r.t. the
  // longitude at the given spherical coordinates.
  const Point<dim> point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);

  const Tensor<1,dim> local_velocity          = velocity->point_value(point, mapping);
  const Tensor<2,dim> local_velocity_gradient = velocity->point_gradient(point, mapping);

  return (sin(polar_angle) *
          (radius * local_radial_basis_vector * local_velocity_gradient * local_azimuthal_basis_vector
           +
           local_velocity * local_azimuthal_basis_vector));
}



template <int dim>
void ChristensenBenchmark<dim>::compute_point_data()
{
  TimerOutput::Scope  t(*computing_timer, "Christensen Benchmark: Point data");

  Tensor<1,dim> local_azimuthal_basis_vector;
  local_azimuthal_basis_vector[0] = -sin(sample_point_longitude);
  local_azimuthal_basis_vector[1] = cos(sample_point_longitude);

  temperature_at_sample_point           = temperature->point_value(sample_point, mapping);
  azimuthal_velocity_at_sample_point    = velocity->point_value(sample_point, mapping) *
                                          local_azimuthal_basis_vector;

  /*
  Tensor<1,dim> local_polar_basis_vector;
  local_polar_basis_vector[0]   = cos(sample_point_colatitude) *
                                  cos(sample_point_longitude);
  local_polar_basis_vector[1]   = cos(sample_point_colatitude) *
                                  sin(sample_point_longitude);
  if constexpr(dim == 3)
    local_polar_basis_vector[2] = -sin(sample_point_colatitude);

  polar_magnetic_field_at_sample_point  = magnetic_field->point_value(sample_point, mapping) *
                                          local_polar_basis_vector;
  */
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

template class RMHD::BenchmarkData::ChristensenBenchmark<2>;
template class RMHD::BenchmarkData::ChristensenBenchmark<3>;

template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::ChristensenBenchmark<2> &);
template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::ChristensenBenchmark<3> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::ChristensenBenchmark<2> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::ChristensenBenchmark<3> &);
