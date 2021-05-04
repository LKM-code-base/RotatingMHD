#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/assembly_data.h>

#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <vector>

namespace RMHD
{

using namespace dealii;

namespace BenchmarkData
{

/*!
 * @struct DFGBechmarkRequest
 *
 * @brief A structure containing the data requested by the DFG benchmark
 * and methods to compute them. The structure computes all request data in
 * dimensionless form.
 *
 * @details The following requested data are computed by evaluating the solution:
 *  - drag coefficient,
 *  - lift coefficient,
 *  - pressure difference.
 *
 *  For a definition of these please refer to the discussion in @ref DFG and the
 *  documentation of @ref compute_drag_and_lift_coefficients .
 *
 */
template <int dim>
struct DFGBechmarkRequest
{

  /*!
   *
   * @brief The default constructor of the structure.
   *
   */
  DFGBechmarkRequest(const double reynolds_number = 100.0,
                     const double reference_length = 1.0);

  /*!
   * @brief The Reynolds number of the problem.
   *
   * @details Its value is defined as
   * \f[ \Reynolds = \frac{v_\mathrm{ref}D}{\nu} \f],
   * where \f$ D \f$ denotes the diameter of the cylinder, \f$ v_\mathrm{ref}\f$
   * the reference velocity which is equal to the average value of the velocity
   * at the inlet. \f$ \nu \f$ is the kinematic viscosity.
   *
   */
  const double  Re;

  /*!
   * @brief The point \f$ \mathrm{P}_\text{front} \f$ at the front side of the
   * cylinder, at which the pressure will be evaluated.
   *
   * @details The point \f$ \mathrm{P}_\text{front} \f$ is located at the
   * dimensionless position
   * \f$ \tilde{\bs{x}}_\text{front} = \frac{1}{D}\big(0.15\,\mathrm{m}\, \ex
   * + 0.20\,\mathrm{m}\, \ey \big)\f$.
   *
   */
  const Point<dim>  front_evaluation_point;

  /*!
   * @brief The point \f$ \mathrm{P}_\text{rear} \f$ at the rear side of the
   * cylinder, at which the pressure will be evaluated.
   *
   * @details The point \f$ \mathrm{P}_\text{rear} \f$ is located at the
   * dimensionless position
   * \f$ \tilde{\bs{x}}_\text{rear} = \frac{1}{D}\big(0.25\,\mathrm{m}\, \ex
   * + 0.20\,\mathrm{m}\, \ey \big)\f$.
   *
   */
  const Point<dim>  rear_evaluation_point;

  /*!
   * @brief The dimensionless pressure difference between the front an the rear
   * evaluation points.
   *
   * @details The dimensionless pressure difference is defined as
   *
   * \f[
   *    \Delta \tilde{p} = \tilde{p}|_{\tilde{\bs{x}}_\text{front}} -
   *    \tilde{p}|_{\tilde{\bs{x}}_\text{rear}}\,.
   * \f]
   *
   */
  double        pressure_difference;

  /*!
   * @brief The drag coefficient.
   *
   * @details It is defined as
   *
   * \f[
   *    c_\text{drag} = 2 \tilde{\bs{F}} \cdot \ex\,,
   * \f]
   *
   * where \f$ \tilde{\bs{F}} \f$ is the dimensionless resulting force acting on
   * the cylinder as described in the documentation of
   * @ref compute_drag_and_lift_coefficients and @ref DFG.
   *
   */
  double        drag_coefficient;

  /*!
   * @brief The lift coefficient.
   *
   * @details It is defined as
   *
   * \f[
   *    c_\text{lift} = 2 \tilde{\bs{F}} \cdot \ey\,,
   * \f]
   *
   * where \f$ \tilde{\bs{F}} \f$ is the dimensionless resulting force acting on
   * the cylinder as described in the documentation of
   * @ref compute_drag_and_lift_coefficients and @ref DFG.
   *
   */
  double        lift_coefficient;

  /*!
   * @brief A table containing the step number, current dimensionless
   * time, the @ref pressure_difference, the @ref drag_coefficient and
   * the @ref lift_coefficient.
   */
  TableHandler  data_table;

  /*!
   * @brief This method computes the @ref pressure_difference.
   */
  void compute_pressure_difference
  (const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure);

  /*!
   * @brief This method computes the @ref drag_coefficient and the
   * @ref lift_coefficient .
   *
   * @details It computes the dimensionless force acting on the cylinder using
   * the following formula
   *
   * \f[
   *    \tilde{\bs{F}} = \int_{\tilde{\Gamma}_3} \Big(-\tilde{p}\bs{1} +
   *    \tfrac{1}{\Reynolds} \big( \tilde{\nabla} \otimes \tilde{\bs{v}}
   *    + \tilde{\bs{v}} \otimes \tilde{\nabla}\big)
   *    \Big) \cdot \bs{n} \dint{\tilde{A}}\,.
   * \f]
   *
   * Then the @ref drag_coefficient and @ref lift_coefficient are obtained as
   *
   * \f[
   *    c_\text{drag} = 2 \tilde{\bs{F}} \cdot \ex\,, \qquad
   *    c_\text{lift} = 2 \tilde{\bs{F}} \cdot \ey\,.
   * \f]
   *
   */
  void compute_drag_and_lift_coefficients
  (const std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
   const std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
   const types::boundary_id                             cylinder_boundary_id = 2);

  /*!
   * @brief A method that updates @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and the @ref lift_coefficient.
   */
  void update_table(TimeDiscretization::DiscreteTime  &time);

  /*!
   * @brief A method that prints @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and the @ref lift_coefficient.
   */
  void print_step_data(TimeDiscretization::DiscreteTime &time);

  /*!
   * @brief A method that outputs the @ref data_table into a file.
   */
  void write_table_to_file(const std::string &file);
};

template<int dim> class MIT;

template<typename Stream, int dim>
Stream& operator<<(Stream &, const MIT<dim> &);

/*!
 * @class MIT
 * @brief A class which computes and contains all the MIT benchmark's data.
 * @details Furthermore the probed data at the first sample point can
 * be printed to the terminal through the overloaded stream operator
 * and the entire data can be printed to a text file.
 * @todo Compute the stream function and the vorticity at the
 * first sample point.
 */
template <int dim>
class MIT
{
public:
  /*!
   * @brief Constructor.
   */
  MIT(const std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
      const std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
      const std::shared_ptr<Entities::ScalarEntity<dim>>  &temperature,
      TimeDiscretization::VSIMEXMethod                    &time_stepping,
      const unsigned int                                  left_wall_boundary_id,
      const unsigned int                                  right_wall_boundary_id,
      const std::shared_ptr<Mapping<dim>>                 external_mapping
                                = std::shared_ptr<Mapping<dim>>(),
      const std::shared_ptr<ConditionalOStream>           external_pcout
                                = std::shared_ptr<ConditionalOStream>(),
      const std::shared_ptr<TimerOutput>                  external_timer
                                = std::shared_ptr<TimerOutput>());

  /*!
   * @brief A method that computes all the benchmark data with the
   * last computed field variables.
   */
  void compute_benchmark_data();

  /*!
   * @brief Outputs the computed benchmark data to a text file in
   * org mode format.
   */
  void print_data_to_file(std::string file_name);


  /*!
   * @brief Output of the benchmark data to the terminal.
   */
  template<typename Stream, int dim_>
  friend Stream& operator<<(Stream &stream, const MIT<dim_> &mit);

private:
  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                                &mpi_communicator;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod        &time_stepping;

  /*!
   * @brief A shared pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>           pcout;

  /*!
   * @brief A shared pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>                  computing_timer;

  /*!
   * @brief A shared pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>                 mapping;

  /*!
   * @brief A shared pointer to the velocity field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::VectorEntity<dim>>  velocity;

  /*!
   * @brief A shared pointer to the pressure field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::ScalarEntity<dim>>  pressure;

  /*!
   * @brief A shared pointer to the temperature field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::ScalarEntity<dim>>  temperature;

  /*!
   * @brief A vector containing all the points at which data will be
   * sampled.
   * @details The points are
   * \f[
   * P_1 = (0.1810,\, 7.3700), \quad
   * P_2 = (0.8190,\, 0.6300), \quad
   * P_3 = (0.1810,\, 0.6300), \quad
   * P_4 = (0.8190,\, 7.3700), \quad \textrm{and} \quad
   * P_5 = (0.1810,\, 4.0000).
   * \f]
   */
  std::vector<Point<dim>>                       sample_points;

  /*!
   * @brief A vector containing the pertinent pressure differences.
   * @details Defined as
   * \f[
   * \Delta p_{ij} = p_i - p_j
   * \f]
   * where the subindices indicate the sample point. The computed
   * differences are \f$ \Delta p_{14} \f$, \f$ \Delta p_{51} \f$ and
   * \f$ \Delta p_{35} \f$.
   */
  std::vector<double>                           pressure_differences;

  /*!
   * @brief The velocity vector at the sample point \f$ P_1 \f$.
   */
  Tensor<1,dim>                                 velocity_at_p1;

  /*!
   * @brief The temperature at the sample point \f$ P_1 \f$.
   */
  double                                        temperature_at_p1;

  /*!
   * @brief The stream function at the sample point \f$ P_1 \f$.
   */
  double                                        stream_function_at_p1;

  /*!
   * @brief The vorticity norm at the sample point \f$ P_1 \f$.
   */
  double                                        vorticity_at_p1;

  /*!
   * @brief The Nusselt number at the left and right walls.
   * @details They are given by
   * \f[
   * \mathit{Nu}_{1,2} = \dfrac{1}{H} \int_{\Gamma_{1,2}} \nabla \vartheta
   * \cdot \bs{n} \dint a
   * \f]
   * where the subindices 0 and 1 indicate the left and right walls
   * respectively.
   */
  std::pair<double, double>                     nusselt_numbers;

  /*!
   * @brief The skewness metric.
   * @details Given by
   * \f[
   * \varepsilon_{12} = \vartheta_1 + \vartheta_2
   * \f]
   * where the subindices indicate the sample point.
   */
  double                                        skewness_metric;

  /*!
   * @brief The average velocity metric.
   * @details Given by
   * \f[
   * \hat{u} = \sqrt{ \dfrac{1}{2HW} \int_\Omega \bs{u} \cdot \bs{u} \dint v}
   * \f]   */
  double                                        average_velocity_metric;

  /*!
   * @brief The average vorticity metric.
   * @details Given by
   * \f[
   * \hat{\omega} = \sqrt{ \dfrac{1}{2HW} \int_\Omega
   * (\nabla \times \bs{u}) \cdot (\nabla \times \bs{u}) \dint v}   * \f]   */
  double                                        average_vorticity_metric;

  /*!
   * @brief The table which stores all the benchmark data.
   */
  TableHandler                                  data;

  /*!
   * @brief The width of the cavity.
   * @details Given by  \f$ W = 1.0 \f$.
   */
  const double                                  width;

  /*!
   * @brief The height of the cavity.
   * @details Given by  \f$ H = 8.0 \f$.
   */
  const double                                  height;

  /*!
   * @brief The areaa of the cavity.
   * @details Given by  \f$ A = WH \f$
   */
  const double                                  area;

  /*!
   * @brief The boundary id of the cavity's left wall.
   */
  const unsigned int                            left_wall_boundary_id;

  /*!
   * @brief The boundary id of the cavity's right wall.
   */
  const unsigned int                            right_wall_boundary_id;

  /*!
   * @brief A method that samples all the point data and computes the
   * pressure differences and skew-symmetrie of the temperature field.
   * @brief More specifically, the method updates the values of
   * \f$ \bs{u} \f$, \f$ \vartheta \f$, \f$ \psi \f$ and \f$ \omega \f$
   * at the sample point \f$ P_1 \f$; the skewness metric
   * \f$ \varepsilon_{12} \f$; the pressure differences
   * \f$ \Delta p_{14} \f$, \f$ \Delta p_{51} \f$ and
   * \f$ \Delta p_{35} \f$.
   */
  void compute_point_data();

  /*!
   * @brief A method that computes the Nusselt number of the
   * walls with Dirichlet boundary conditions on the temperature field.
   * @details They are given by
   * \f[
   * \mathit{Nu}_{1,2} = \dfrac{1}{H} \int_{\Gamma_{1,2}} \nabla \vartheta
   * \cdot \bs{n} \dint a
   * \f]
   * where the subindices 0 and 1 indicate the left and right walls
   * respectively.
   * */
  void compute_wall_data();

  /*!
   * @brief A method that computes the average velocity and vorticity
   * metrics.
   * @details They are given by
   * \f[
   * \hat{u} = \sqrt{ \dfrac{1}{2HW} \int_\Omega \bs{u} \cdot \bs{u} \dint v}
   * \quad \textrm{and} \quad
   * \hat{\omega} = \sqrt{ \dfrac{1}{2HW} \int_\Omega
   * (\nabla \times \bs{u}) \cdot (\nabla \times \bs{u}) \dint v}
   * \f]
   */
  void compute_global_data();
};



template<int dim> class ChristensenBenchmark;

template<typename Stream, int dim>
Stream& operator<<(Stream &, const ChristensenBenchmark<dim> &);

/*!
 * @class ChristensenBenchmark
 * @brief A class which computes and contains all the Christensen
 * benchmark's data.
 * @details Furthermore the data can be printed to the terminal through
 * the overloaded stream operator or to a text file.
 */
template <int dim>
class ChristensenBenchmark
{
public:
  /*!
   * @brief Constructor.
   */
  ChristensenBenchmark(
    const std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
    const std::shared_ptr<Entities::ScalarEntity<dim>>  &temperature,
    const std::shared_ptr<Entities::VectorEntity<dim>>  &magnetic_field,
    const TimeDiscretization::VSIMEXMethod              &time_stepping,
    const RunTimeParameters::DimensionlessNumbers       &dimensionless_numbers,
    const double                                        outer_radius,
    const double                                        inner_radius,
    const unsigned int                                  case_number,
    const std::shared_ptr<Mapping<dim>>                 external_mapping
                              = std::shared_ptr<Mapping<dim>>(),
    const std::shared_ptr<ConditionalOStream>           external_pcout
                              = std::shared_ptr<ConditionalOStream>(),
    const std::shared_ptr<TimerOutput>                  external_timer
                              = std::shared_ptr<TimerOutput>());

  /*!
   * @brief A method that computes all the benchmark data with the
   * last computed field variables.
   */
  void compute_benchmark_data();

  /*!
   * @brief Outputs the computed benchmark data to a text file in
   * org mode format.
   */
  void print_data_to_file(std::string file_name);


  /*!
   * @brief Output of the benchmark data to the terminal.
   */
  template<typename Stream, int dim_>
  friend Stream& operator<<(Stream &stream, const ChristensenBenchmark<dim_> &mit);


private:
  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                                &mpi_communicator;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod        &time_stepping;

  /*!
   * @brief A shared pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>           pcout;

  /*!
   * @brief A shared pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>                  computing_timer;

  /*!
   * @brief A shared pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>                 mapping;

  /*!
   * @brief A shared pointer to the velocity field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::VectorEntity<dim>>  velocity;

  /*!
   * @brief A shared pointer to the temperature field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::ScalarEntity<dim>>  temperature;

  /*!
   * @brief A shared pointer to the magnetic flux field's numerical
   * representation.
   */
  const std::shared_ptr<const Entities::VectorEntity<dim>>  magnetic_field;

  /*!
   * @brief A reference to the struct containing all the relevant
   * dimensionless numbers.
   */
  const RunTimeParameters::DimensionlessNumbers             &dimensionless_numbers;

  /*!
   * @brief The number of the case to be performed.
   */
  const unsigned int  case_number;

  /*!
   * @brief The radius of the sample point. Set as the mid-depth radius
   * \f$ r = \tfrac{1}{2}(r_i + r_o) \f$.
   */
  const double  sample_point_radius;

  /*!
   * @brief The colatitude of the sampling point. Set as the equatiorial
   * plane, *i. e.*, \f$ \theta = 90 \f$.
   */
  const double  sample_point_colatitude;

  /*!
   * @brief The longitude of the sampling point. Set by the conditions
   * \f$ u_{\textrm{r}} = 0 \f$ and \f$ \pd{ u_{\textrm{r}}}{\phi} > 0\f$.
   */
  double        sample_point_longitude;

  /*!
   * @brief The sample point.
   */
  Point<dim>    sample_point;

  /*!
   * @brief The drift frequency.
   */
  double        drift_frequency;

  /*!
   * @brief The mean kinetic energy density
   * @details Given by
   * \f[
   * E_{\textrm{kin}} = \frac{1}{2V} \int_\Omega \bs{u} \cdot \bs{u} \dint{v}
   * \f]
   * where \f$ V \f$ is the volume of the shell,
   * \f$ \Omega \f$ the shell domain and
   * \f$ \bs{u} \f$ the velocity field.
   */
  double        mean_kinetic_energy_density;

  /*!
   * @brief The mean magnetic energy density
   * @details Given by
   * \f[
   * E_{\textrm{mag}} = \frac{1}{2V \Ekman \magPrandtl} \int_\Omega \bs{B} \cdot \bs{B} \dint{v}
   * \f]
   * where \f$ V \f$ is the volume of the shell,
   * \f$ \Ekman \f$ the Ekman number,
   * \f$ \magPrandtl \f$ the magnetic Prandtl number,
   * \f$ \Omega \f$ the shell domain and
   * \f$ \bs{B} \f$ the magnetic field.
   */
  double        mean_magnetic_energy_density;

  /*!
   * @brief The volume of the discrete domain.
   */
  double        discrete_volume;

  /*!
   * @brief The temperature evaluated at the @ref sample_point.
   */
  double        temperature_at_sample_point;

  /*!
   * @brief The velocity vector evaluated at the @ref sample_point.
   */
  double        azimuthal_velocity_at_sample_point;

  /*!
   * @brief The magnetic flux vector evaluated at the @ref sample_point.
   */
  double        polar_magnetic_field_at_sample_point;

  /*!
   * @brief The table which stores all the benchmark data.
   */
  TableHandler  data;

  /*!
   * @brief A method that computes the @ref drift_frequency, the
   * @ref mean_kinetic_energy_density and the
   * @ref mean_magnetic_energy_density.
   * @todo Compute drift frequency
   */
  void compute_global_data();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void compute_local_global_squared_norms(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyData::Benchmarks::Christensen::Scratch<dim>  &scratch,
    AssemblyData::Benchmarks::Christensen::Copy          &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the velocity field on a single cell into the global matrices.
   */
  void copy_local_to_global_squared_norms(
    const AssemblyData::Benchmarks::Christensen::Copy  &data);

  /*!
   * @brief A method that locates the sample point.
   * @details The radius and the colatitude are set as the mid-depth
   * radius and the equatorial plane respectively. The longitude is
   * determine by the conditions
   * \f[
   * u_{\textrm{r}} = 0 \qquad
   * \textrm{and} \qquad
   * \pd{ u_{\textrm{r}}}{\phi} > 0.
   * \f]
   */
  void find_sample_point();

  /*!
   * @brief A method that computes the value of the radial velocity
   * at the given spherical coordinates
   */
  double compute_radial_velocity(
    const double radius,
    const double azimuthal_angle,
    const double polar_angle = numbers::PI_2) const;

  /*!
   * @brief A method that computes a root of the radial velocity w.r.t.
   * the longitude.
   * @details The method is employs around the boost's
   * bracket_and_solve_root method.
   */
  double compute_zero_of_radial_velocity(
    const double       phi_guess,
    const bool         local_slope,
    const double       tol,
    const unsigned int &max_iter) const;

  /*!
   * @brief A method that computes the value of the derivative of the
   * radial velocity w.r.t. to the longitude at the given spherical
   * coordinates
   */
  double compute_azimuthal_gradient_of_radial_velocity(
    const double radius,
    const double azimuthal_angle,
    const double polar_angle = numbers::PI_2) const;

  /*!
   * @brief A method that computes the velocity vector and the temperature
   * at the @ref sample_point.
   */
  void compute_point_data();
};

} // namespace BenchmarkData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */
