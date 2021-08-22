
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/exceptions.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <boost/math/tools/roots.hpp>

#include <fstream>
#include <ostream>

namespace RMHD
{

using namespace dealii;

namespace BenchmarkData
{

DeclException0(ExcBoostNoConvergence);

template <>
DFGBechmarkRequests<2>::DFGBechmarkRequests
(const double             reynolds_number,
 const types::boundary_id cylinder_id)
:
Re(reynolds_number),
cylinder_boundary_id(cylinder_id),
front_evaluation_point(1.5, 2.0),
rear_evaluation_point(2.5, 2.0),
pressure_difference(0),
drag_coefficient(0),
lift_coefficient(0)
{
  data_table.declare_column("step");
  data_table.declare_column("time");
  data_table.declare_column("pressure difference");
  data_table.declare_column("drag coeff.");
  data_table.declare_column("lift coeff.");

  data_table.set_scientific("time", true);
  data_table.set_scientific("pressure difference", true);
  data_table.set_scientific("drag coeff.", true);
  data_table.set_scientific("lift coeff.", true);

  data_table.set_precision("time", 6);
  data_table.set_precision("pressure difference", 6);
  data_table.set_precision("drag coeff.", 6);
  data_table.set_precision("lift coeff.", 6);
}



template <int dim>
void DFGBechmarkRequests<dim>::compute_pressure_difference
(const Entities::FE_ScalarField<dim> &pressure)
{
  const double front_value = pressure.point_value(front_evaluation_point);

  const double rear_value = pressure.point_value(rear_evaluation_point);

  pressure_difference = front_value - rear_value;
}



template <>
void DFGBechmarkRequests<2>::compute_drag_and_lift_coefficients
(const Entities::FE_VectorField<2>  &velocity,
 const Entities::FE_ScalarField<2>  &pressure)
{
  constexpr unsigned int dim{2};

  const MappingQ<dim> mapping(3);
  const QGauss<dim-1>   face_quadrature_formula(velocity.fe_degree() + 1);

  FEFaceValues<dim> velocity_face_fe_values(mapping,
                                            velocity.get_finite_element(),
                                            face_quadrature_formula,
                                            update_gradients|
                                            update_JxW_values|
                                            update_normal_vectors);

  FEFaceValues<dim> pressure_face_fe_values(mapping,
                                            pressure.get_finite_element(),
                                            face_quadrature_formula,
                                            update_values);

  const unsigned int n_face_q_points = face_quadrature_formula.size();

  const FEValuesExtractors::Vector  velocities(0);

  std::vector<double>         pressure_values(n_face_q_points);
  std::vector<Tensor<1, dim>> normal_vectors(n_face_q_points);
  std::vector<Tensor<2, dim>> velocity_gradients(n_face_q_points);

  Tensor<1, dim>              forces;

  for (const auto &cell : velocity.get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
        {
          velocity_face_fe_values.reinit(cell, face);

          typename DoFHandler<dim>::active_cell_iterator
          pressure_cell(&velocity.get_triangulation(),
                        cell->level(),
                        cell->index(),
                        // pointer to the pressure's DoFHandler
                        &(pressure.get_dof_handler()));

          typename DoFHandler<dim>::active_face_iterator
          pressure_face(&velocity.get_triangulation(),
                        face->level(),
                        face->index(),
                        // pointer to the pressure's DoFHandler
                        &(pressure.get_dof_handler()));

          pressure_face_fe_values.reinit(pressure_cell, pressure_face);

          velocity_face_fe_values[velocities].get_function_gradients(velocity.solution, velocity_gradients);

          pressure_face_fe_values.get_function_values(pressure.solution, pressure_values);

          normal_vectors = velocity_face_fe_values.get_normal_vectors();

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            // The reversed signs here are due to the way how the normal
            // vector is defined in the DFG benchmark.
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

  // Gather the values of each processor
  const Triangulation<dim>  &tria = velocity.get_triangulation();
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&tria);
  if (tria_ptr != nullptr)
  {
    forces = Utilities::MPI::sum(forces, tria_ptr->get_communicator());
  }



  drag_coefficient = 2.0 * forces[0];
  lift_coefficient = 2.0 * forces[1];
}



template <int dim>
void DFGBechmarkRequests<dim>::write_text(std::ostream &file) const
{
  data_table.write_text(file, TableHandler::TextOutputFormat::org_mode_table);
}



template <int dim>
void DFGBechmarkRequests<dim>::update
(const double                         time,
 const unsigned int                   step_number,
 const Entities::FE_VectorField<dim> &velocity,
 const Entities::FE_ScalarField<dim> &pressure)
{
  compute_drag_and_lift_coefficients(velocity, pressure);
  compute_pressure_difference(pressure);

  data_table.add_value("step", step_number);
  data_table.add_value("time", time);
  data_table.add_value("pressure difference",  pressure_difference);
  data_table.add_value("drag coeff.", drag_coefficient);
  data_table.add_value("lift coeff.", lift_coefficient);
}



template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const DFGBechmarkRequests<dim> &dfg)
{
  stream << "    \u0394p = "
         << std::setprecision(6)
         << std::scientific
         << dfg.pressure_difference
         << ", drag coeff. = "
         << dfg.drag_coefficient
         << ", lift coeff. = "
         << dfg.lift_coefficient
         << std::defaultfloat;

  return (stream);
}


template <>
MIT<2>::MIT
(const types::boundary_id left_wall_boundary_id,
 const types::boundary_id right_wall_boundary_id)
:
pressure_differences(3),
width(1.0),
height(8.0),
area(8.0),
left_wall_boundary_id(left_wall_boundary_id),
right_wall_boundary_id(right_wall_boundary_id)
{
  // Initiating the sample points.
  sample_points.emplace_back(0.1810, 7.3700);
  sample_points.emplace_back(0.8190, 0.6300);
  sample_points.emplace_back(0.1810, 0.6300);
  sample_points.emplace_back(0.8190, 7.3700);
  sample_points.emplace_back(0.1810, 4.0000);

  // Setting up columns
  data.declare_column("time");
  data.declare_column("step");
  data.declare_column("velocity_x_1");
  data.declare_column("velocity_y_1");
  data.declare_column("temperature_1");
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
void MIT<dim>::update
(const double                          time,
 const unsigned int                    step_number,
 const Entities::FE_VectorField<dim>  &velocity,
 const Entities::FE_ScalarField<dim>  &pressure,
 const Entities::FE_ScalarField<dim>  &temperature)
{
  // Compute benchmark data
  compute_point_data(velocity, pressure, temperature);
  compute_wall_data(temperature);
  compute_global_data(velocity);

  // Update column's values
  data.add_value("time", time);
  data.add_value("step", step_number);
  data.add_value("velocity_x_1", velocity_at_p1[0]);
  data.add_value("velocity_y_1", velocity_at_p1[1]);
  data.add_value("temperature_1", temperature_at_p1);
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
void MIT<dim>::write_text(std::ostream &file) const
{
  data.write_text(file, TableHandler::TextOutputFormat::org_mode_table);
}


template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const MIT<dim> &mit)
{
  stream << std::scientific
         << std::setprecision(6)
         << "    "
         << "u_x(x_1) = "
         << mit.velocity_at_p1[0]
         << ", T(x_1) = "
         << mit.temperature_at_p1
         << std::endl
         << "    "
         << "\u0394p_14 = "
         << mit.pressure_differences[0]
         << ", Nusselt number = "
         << mit.nusselt_numbers.first
         << std::endl
         << "    "
         << "velocity metric = "
         << mit.average_velocity_metric
         << ", vorticity metric = "
         << mit.average_vorticity_metric
         << std::defaultfloat;

  return stream;
}

template <int dim>
void MIT<dim>::compute_point_data
(const Entities::FE_VectorField<dim>  &velocity,
 const Entities::FE_ScalarField<dim>  &pressure,
 const Entities::FE_ScalarField<dim>  &temperature)
{
  // Obtaining data at sample point 1
  velocity_at_p1        = velocity.point_value(sample_points[0]);
  temperature_at_p1     = temperature.point_value(sample_points[0]);

  // Computing skewness metric
  const double temperature_at_p2 = temperature.point_value(sample_points[1]);
  skewness_metric = temperature_at_p1 + temperature_at_p2;

  // Computing pressure differences
  const double pressure_at_p1 = pressure.point_value(sample_points[0]);
  const double pressure_at_p3 = pressure.point_value(sample_points[2]);
  const double pressure_at_p4 = pressure.point_value(sample_points[3]);
  const double pressure_at_p5 = pressure.point_value(sample_points[4]);

  pressure_differences[0] = pressure_at_p1 - pressure_at_p4;
  pressure_differences[1] = pressure_at_p5 - pressure_at_p1;
  pressure_differences[2] = pressure_at_p3 - pressure_at_p5;
}

template <int dim>
void MIT<dim>::compute_wall_data(const Entities::FE_ScalarField<dim>  &temperature)
{
  // Quadrature formula
  const QGauss<dim-1>   face_quadrature_formula(temperature.fe_degree() + 1);

  // Finite element values
  FEFaceValues<dim> face_fe_values(temperature.get_finite_element(),
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

  for (const auto &cell : temperature.get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          if (face->boundary_id() == left_wall_boundary_id ||
              face->boundary_id() == right_wall_boundary_id)
          {
            // Initialize the finite element values
            face_fe_values.reinit(cell, face);

            // Get the temperature gradients at the quadrature points
            face_fe_values.get_function_gradients(temperature.solution,
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
  const Triangulation<dim>  &tria = temperature.get_triangulation();
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&tria);
  if (tria_ptr != nullptr)
  {
    left_boundary_integral   = Utilities::MPI::sum(left_boundary_integral,
                                                   tria_ptr->get_communicator());
    right_boundary_integral  = Utilities::MPI::sum(right_boundary_integral,
                                                   tria_ptr->get_communicator());
  }

  // Compute and store the Nusselt numbers of the walls
  nusselt_numbers = std::make_pair(left_boundary_integral/height,
                                   right_boundary_integral/height);
}

template <int dim>
void MIT<dim>::compute_global_data(const Entities::FE_VectorField<dim>  &velocity)
{
   // Defining the type to contain the vorticity during assembly
  using CurlType = typename FEValuesViews::Vector< dim >::curl_type;

  // Initiate the average velocity and vorticity metrics
  average_velocity_metric   = 0.0;
  average_vorticity_metric  = 0.0;

  // Quadrature formula
  const QGauss<dim>   quadrature_formula(velocity.fe_degree() + 1);

  // Finite element values
  FEValues<dim> fe_values(velocity.get_finite_element(),
                          quadrature_formula,
                          update_values|
                          update_gradients|
                          update_JxW_values);

  // Number of quadrature points
  const unsigned int n_q_points = quadrature_formula.size();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  std::vector<CurlType> vorticity_values(n_q_points);

  for (const auto &cell : velocity.get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
    {
      // Initialize the finite element values
      fe_values.reinit(cell);

      // Define the vector extractor
      const FEValuesExtractors::Vector  velocities(0);

      // Get the velocity and vorticity values at each quadrature
      // point
      fe_values[velocities].get_function_values(velocity.solution,
                                                velocity_values);
      fe_values[velocities].get_function_curls(velocity.solution,
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
  const Triangulation<dim>  &tria = velocity.get_triangulation();
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&tria);
  if (tria_ptr != nullptr)
  {
    average_velocity_metric = Utilities::MPI::sum(average_velocity_metric,
                                                  tria_ptr->get_communicator());
    average_vorticity_metric = Utilities::MPI::sum(average_vorticity_metric,
                                                   tria_ptr->get_communicator());
  }

  // Compute the global averages
  average_velocity_metric   = std::sqrt(average_velocity_metric/(2.0 * area));
  average_vorticity_metric  = std::sqrt(average_vorticity_metric/(2.0 * area));
}



template <int dim>
ChristensenBenchmark<dim>::ChristensenBenchmark
(const double       inner_radius,
 const double       outer_radius,
 const unsigned int case_number)
:
case_number(case_number),
sampling_radius(0.5*(outer_radius + inner_radius)),
sampling_colatitude(numbers::PI_2),
sampling_longitude(0.)
{
  Assert(case_number < 3,
         ExcMessage("Only case 0, 1 and 2  are defined."));

  // Temporary asserts
  AssertThrow(case_number != 0, ExcNotImplemented());

  // Setting up columns
  data.declare_column("time");
  data.declare_column("mean kinetic energy");
  data.declare_column("longitude");
  data.declare_column("temperature");
  data.declare_column("azimuthal velocity");

  // Setting all columns to scientific notation
  data.set_scientific("time", true);
  data.set_scientific("mean kinetic energy", true);
  data.set_scientific("longitude", true);
  data.set_scientific("temperature", true);
  data.set_scientific("azimuthal velocity", true);

  // Setting columns' precision
  data.set_precision("time", 6);
  data.set_precision("mean kinetic energy", 6);
  data.set_precision("longitude", 6);
  data.set_precision("temperature", 6);
  data.set_precision("azimuthal velocity", 6);
}



template <int dim>
void ChristensenBenchmark<dim>::update
(const double time,
 const unsigned int step_number,
 const Entities::FE_VectorField<dim> &velocity,
 const Entities::FE_ScalarField<dim> &temperature,
 const Mapping<dim>                  &mapping)
{
  // Compute benchmark data
  compute_global_data(velocity, mapping);
  find_sampling_point(velocity, mapping);
  compute_point_data(velocity, temperature, mapping);

  // Update column's values
  data.add_value("time", time);
  data.add_value("step", step_number);
  data.add_value("mean kinetic energy", mean_kinetic_energy_density);
  data.add_value("longitude", sampling_longitude);
  data.add_value("temperature", temperature_at_sampling_point);
  data.add_value("azimuthal velocity", azimuthal_velocity_at_sampling_point);
}



template <int dim>
void ChristensenBenchmark<dim>::write_text(std::ostream &file) const
{
  data.write_text(file, TableHandler::TextOutputFormat::org_mode_table);
}



template<typename Stream, int dim>
Stream& operator<<(Stream &stream, const ChristensenBenchmark<dim> &christensen)
{
  switch (christensen.case_number)
  {
    case 0:
      stream << std::scientific
             << std::setprecision(6)
             << "E_kin = "
             << christensen.mean_kinetic_energy_density
             << ", phi = "
             << std::fixed
             << std::setprecision(2)
             << (christensen.sampling_longitude / 2. / numbers::PI * 360.)
             << " deg "
             << std::scientific
             << std::setprecision(6)
             << ", T = "
             << christensen.temperature_at_sampling_point
             << ", v_phi = "
             << christensen.azimuthal_velocity_at_sampling_point
             << std::defaultfloat;
      break;
    case 1:
      AssertThrow(false, ExcNotImplemented());
      break;
    case 2:
      AssertThrow(false, ExcNotImplemented());
      break;
    default:
      AssertThrow(false, ExcNotImplemented());
      break;
  }

  return (stream);
}



template <int dim>
void ChristensenBenchmark<dim>::compute_global_data
(const Entities::FE_VectorField<dim> &velocity,
 const Mapping<dim>                  &mapping)
{
  // Reset the mean energy densities and the volume
  mean_kinetic_energy_density   = 0.0;
  discrete_volume               = 0.0;

  // Quadrature formula
  const QGauss<dim>   quadrature_formula(velocity.fe_degree() + 1);

  // Finite element values
  FEValues<dim> fe_values(mapping,
                          velocity.get_finite_element(),
                          quadrature_formula,
                          update_values|update_JxW_values);

  // Number of quadrature points
  const unsigned int n_q_points = quadrature_formula.size();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);

  // Define the vector extractor
  const FEValuesExtractors::Vector  vector_extractor(0);

  // Integral value
  double velocity_integral{0.0};

  for (const auto &cell : velocity.get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
    {
      // Initialize the finite element values
      fe_values.reinit(cell);
      // Get the velocity values at each quadrature point
      fe_values[vector_extractor].get_function_values(velocity.solution,
                                                      velocity_values);
      // Numerical integration (Loop over all quadrature points)
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        velocity_integral += velocity_values[q] * velocity_values[q] *
                             fe_values.JxW(q);
        discrete_volume += fe_values.JxW(q);
      }
    }

  // Gather the values of each processor
  const Triangulation<dim>  &tria = velocity.get_triangulation();
  const parallel::TriangulationBase<dim> *tria_ptr =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&tria);
  if (tria_ptr != nullptr)
  {
    velocity_integral = Utilities::MPI::sum(velocity_integral,
                                            tria_ptr->get_communicator());
    discrete_volume = Utilities::MPI::sum(discrete_volume,
                                          tria_ptr->get_communicator());
  }

  // Compute the mean values
  mean_kinetic_energy_density = 0.5 * velocity_integral / discrete_volume;
}






template <int dim>
void ChristensenBenchmark<dim>::find_sampling_point
(const Entities::FE_VectorField<dim> &velocity,
 const Mapping<dim>                  &mapping)
{
  // Set the last known value of the sample point's longitude as initial guess
  if (sampling_longitude > 0.)
  {
    // Compute the derivative of the radial velocity w.r.t. the
    // longitude. This value is needed for the boost's
    // bracket_and_solve_root method.
    const double gradient_at_trial_point =
        compute_azimuthal_gradient_of_radial_velocity(sampling_radius,
                                                      sampling_longitude,
                                                      sampling_colatitude,
                                                      velocity,
                                                      mapping);

    // Compute a root of the radial velocity w.r.t. the longitude
    const double trial_longitude =
        compute_zero_of_radial_velocity(sampling_longitude,
                                        gradient_at_trial_point > 0.,
                                        1e-6,
                                        50,
                                        velocity,
                                        mapping);

    // Compute the derivative of the radial velocity w.r.t. the longitude at
    // the found root.
    const double gradient_at_zero =
        compute_azimuthal_gradient_of_radial_velocity(sampling_radius,
                                                      trial_longitude,
                                                      sampling_colatitude,
                                                      velocity,
                                                      mapping);

    if(gradient_at_zero > 0. &&
       trial_longitude >= 0. &&
       trial_longitude <= 2. * numbers::PI)
    {
        sampling_longitude = trial_longitude;

        // Compute the position vector of the sample point in cartesian
        // coordinates.
        std::array<double, dim> spherical_coordinates;

        if constexpr(dim == 2)
          spherical_coordinates = {sampling_radius,
                                  sampling_longitude};
        else if constexpr(dim == 3)
          spherical_coordinates = {sampling_radius,
                                  sampling_longitude,
                                  sampling_colatitude};

        sampling_point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);

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
      const double gradient_at_trial_point =
          compute_azimuthal_gradient_of_radial_velocity(sampling_radius,
                                                        trial_longitudes[cnt],
                                                        sampling_colatitude,
                                                        velocity,
                                                        mapping);

      try
      {
          const double tentative_longitude =
              compute_zero_of_radial_velocity(trial_longitudes[cnt],
                                              gradient_at_trial_point > 0.,
                                              1e-6,
                                              50,
                                              velocity,
                                              mapping);

          const double gradient_at_zero =
              compute_azimuthal_gradient_of_radial_velocity(sampling_radius,
                                                            tentative_longitude,
                                                            sampling_colatitude,
                                                            velocity,
                                                            mapping);

          if (gradient_at_zero > 0.)
          {
              point_found             = true;
              sampling_longitude  = tentative_longitude;
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
  if (sampling_longitude < 0. || sampling_longitude > 2. * numbers::PI)
  {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on find_sample_point!" << std::endl
                << "The algorithm did not find a point in [0,2*pi)."
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
    spherical_coordinates = {sampling_radius,
                             sampling_longitude};
  else if constexpr(dim == 3)
    spherical_coordinates = {sampling_radius,
                             sampling_longitude,
                             sampling_colatitude};

  sampling_point = GeometricUtilities::Coordinates::from_spherical(spherical_coordinates);
}



template <int dim>
double ChristensenBenchmark<dim>::compute_radial_velocity
(const double radius,
 const double azimuthal_angle,
 const double polar_angle,
 const Entities::FE_VectorField<dim> &velocity,
 const Mapping<dim>                  &mapping) const
{
  using namespace  GeometryExceptions;

  AssertThrow(radius > 0.,
              ExcNegativeRadius(radius));
  AssertThrow((polar_angle >= 0. && polar_angle <= numbers::PI),
              ExcPolarAngleRange(polar_angle));

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
  const Tensor<1,dim> local_velocity = velocity.point_value(point, mapping);

  return local_velocity * point / radius;
}



template<int dim>
double ChristensenBenchmark<dim>::compute_zero_of_radial_velocity
(const double         phi_guess,
 const bool           local_slope,
 const double         tol,
 const unsigned int   max_iter,
 const Entities::FE_VectorField<dim> &velocity,
 const Mapping<dim>                  &mapping) const
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

  const double radius = sampling_radius;
  const double theta  = sampling_colatitude;

  // Declare lambda functions for the boost's
  // root bracket_and_solve_root method
  auto function = [this, radius, theta, &velocity, &mapping](const double &x)
  {
      return this->compute_radial_velocity(radius, x, theta, velocity, mapping);
  };
  auto tolerance_criterion = [tol, &function](const double &a, const double &b)
  {
      return std::abs(function(a)) <= tol && std::abs(function(b)) <= tol;
  };

  // @attention Why initialize phi to this value?
  double phi = -2. * numbers::PI;

  // Find a root of the radial velocity w.r.t. the longitude
  try
  {
      auto phi_interval
      = bracket_and_solve_root(function,
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

  return (phi);
}



template <int dim>
double ChristensenBenchmark<dim>::compute_azimuthal_gradient_of_radial_velocity
(const double radius,
 const double azimuthal_angle,
 const double polar_angle,
 const Entities::FE_VectorField<dim> &velocity,
 const Mapping<dim>                  &mapping) const
{
  AssertThrow(radius > 0.,
              GeometryExceptions::ExcNegativeRadius(radius));
  AssertThrow((polar_angle >= 0. && polar_angle <= numbers::PI),
              GeometryExceptions::ExcPolarAngleRange(polar_angle));
  AssertThrow((azimuthal_angle >= 0. && azimuthal_angle < 2. * numbers::PI),
              GeometryExceptions::ExcAzimuthalAngleRange(azimuthal_angle));

  // Define the local basis vectors and the spherical coordinates
  Tensor<1,dim> local_radial_basis_vector;
  local_radial_basis_vector[0] = sin(polar_angle) * cos(azimuthal_angle);
  local_radial_basis_vector[1] = sin(polar_angle) * sin(azimuthal_angle);

  Tensor<1,dim> local_azimuthal_basis_vector;
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

  const Tensor<1,dim> local_velocity          = velocity.point_value(point, mapping);
  const Tensor<2,dim> local_velocity_gradient = velocity.point_gradient(point, mapping);

  return (sin(polar_angle) *
          (radius * local_radial_basis_vector * local_velocity_gradient * local_azimuthal_basis_vector
           +
           local_velocity * local_azimuthal_basis_vector));
}



template <int dim>
void ChristensenBenchmark<dim>::compute_point_data
(const Entities::FE_VectorField<dim> &velocity,
 const Entities::FE_ScalarField<dim> &temperature,
 const Mapping<dim>                  &mapping)
{
  Tensor<1,dim> local_azimuthal_basis_vector;
  local_azimuthal_basis_vector[0] = -sin(sampling_longitude);
  local_azimuthal_basis_vector[1] = cos(sampling_longitude);

  temperature_at_sampling_point        = temperature.point_value(sampling_point, mapping);
  azimuthal_velocity_at_sampling_point = velocity.point_value(sampling_point, mapping) *
                                       local_azimuthal_basis_vector;
}



} // namespace BenchmarkData

} // namespace RMHD

// explicit instantiations
template struct RMHD::BenchmarkData::DFGBechmarkRequests<2>;

template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::DFGBechmarkRequests<2> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::DFGBechmarkRequests<2> &);

template class RMHD::BenchmarkData::MIT<2>;

template std::ostream & RMHD::BenchmarkData::operator<<
(std::ostream &, const RMHD::BenchmarkData::MIT<2> &);
template dealii::ConditionalOStream & RMHD::BenchmarkData::operator<<
(dealii::ConditionalOStream &, const RMHD::BenchmarkData::MIT<2> &);

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
