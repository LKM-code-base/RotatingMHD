
#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/discrete_time.h>
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
 * @struct DFG
 * 
 * @brief A struct containing variables of the DFG benchmark and methods
 * to compute them.
 * 
 * @details 
 * 
 *
 */
template <int dim>
struct DFG
{
  /*!
   * @brief The density of the fluid, \f$ \rho = 1.0 \f$.
   */
  double        density;
  /*!
   * @brief The characteristic length of the benchmark is set as the 
   * diameter of the cylinder, \f$ d =0.1 \f$.
   */
  double        characteristic_length;
  /*!
   * @brief The mean velocity of the fluid.
   * @details Defined as the  
   * \f[
   * \bar{v} = \frac{2}{3} \mathbf{v}_\textrm{inflow}\big|_{(0,0.5H)} 
   * \mathbf{e}_\textrm{x} = \frac{2}{3} \left( 
   * \frac{4v_\textrm{max}y(H-y)}{H^2} \right)\bigg|_{(0,0.5H)}
   * \f]
   * which with \f$ v_\textrm{max} = 1.5 \f$ equals to \f$ 1.0 \f$.
   */
  double        mean_velocity;
  /*!
   * @brief The kinematic viscosity of the fluid, \f$ \nu = 0.001 \f$.
   */
  double        kinematic_viscosity;
  /*!
   * @brief The Reynolds number of the problem. Its value is 
   * \f[
   *   \textrm{Re} = \frac{\bar{v}d}{\nu} = 100.
   * \f]
   */
  const double  Re;

  /*!
   * @brief The point at the front side of the cylinder, at which the
   * pressure will be evaluated.
   * @details The point is located at 
   * \f$ P_\textrm{front} = (0.15,\, 0.20) \f$.
   * @attention The struct defines the coordinates in their
   * dimensionless form in the constructor using the characteristic
   * length.
   */
  const Point<dim>  front_evaluation_point;

  /*!
   * @brief The point at the rear side of the cylinder, at which the
   * pressure will be evaluated.
   * @details The point is located at 
   * \f$ P_\textrm{front} = (0.25,\, 0.20) \f$.
   * @attention The struct defines the coordinates in their
   * dimensionless form in the constructor using the characteristic
   * length.
   */
  const Point<dim>  rear_evaluation_point;

  /*!
   * @brief The pressure difference between the front an the rear 
   * evaluation points.
   * @details Defined as
   * \f[
   * \Delta p = p|_{P_\textrm{front}} - p|_{P_\textrm{rear}}
   * \f]
   * @attention The computed pressure difference is dimensionless, but
   * it is interchangeable with the actual pressure.
   */
  double        pressure_difference;

  /*!
   * @brief The drag force around the cylinder. 
   * @details Defined as the \f$ x \f$ component of the total force
   * exorted on the cylinder. The total force is given by
   * \f[
	 * \boldsymbol{\mathfrak{f}} = \mathfrak{f}_\textrm{drag}  \mathbf{e}_\textrm{x} 
   * +  \mathfrak{f}_\textrm{lift} \mathbf{e}_\textrm{y} = 
   * \int_{\Gamma_3} [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] 
   * \cdot \mathbf{n} \mathrm{d} \ell
   * \f]
   * Its dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method and 
   * considering the formulation of the NavierStokesProjection class,
   * is given then by
   * \f[
   * \boldsymbol{\tilde{\mathfrak{f}}} = 
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} (\tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} + \mathbf{\tilde{v}} \otimes \tilde{\nabla}) 
   *  ] \cdot \mathbf{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   */
  double        drag_force;

  /*!
   * @brief The drag coefficient.
   * @details Defined as 
   * \f[
   * c_\textrm{drag} = \dfrac{2}{\rho\bar{v}^2D} \boldsymbol{\mathfrak{f}} \cdot 
   * \mathbf{e}_\textrm{x}
   * \f]
   * which is equivalent to \f$ c_\textrm{drag} = 2
   * \boldsymbol{\tilde{\mathfrak{f}}} \cdot \mathbf{e}_\textrm{x} \f$, 
   * what is actually computed by the by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method.
   */
  double        drag_coefficient;

  /*!
   * @brief The lift force around the cylinder. 
   * @details Defined as the \f$ y \f$ component of the total force
   * exorted on the cylinder. The total force is given by
   * \f[
	 * \boldsymbol{\mathfrak{f}} = \mathfrak{f}_\textrm{drag}  \mathbf{e}_\textrm{x} 
   * +  \mathfrak{f}_\textrm{lift} \mathbf{e}_\textrm{y} = 
   * \int_{\Gamma_3} [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] 
   * \cdot \mathbf{n} \mathrm{d} \ell
   * \f]
   * In dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method and 
   * considering the formulation of the NavierStokesProjection class,
   * is given then by
   * \f[
   * \boldsymbol{\tilde{\mathfrak{f}}} = 
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} (\tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} + \mathbf{\tilde{v}} \otimes \tilde{\nabla}) 
   *  ] \cdot \mathbf{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   */
  double        lift_force;

  /*!
   * @brief The lift coefficient.
   * @details Defined as 
   * \f[
   * c_\textrm{lift} = \dfrac{2}{\rho\bar{v}^2D} \boldsymbol{\mathfrak{f}} \cdot 
   * \mathbf{e}_\textrm{y}
   * \f]
   * which is equivalent to \f$ c_\textrm{lift} = 2
   * \boldsymbol{\tilde{\mathfrak{f}}} \cdot \mathbf{e}_\textrm{y} \f$, 
   * what is actually computed by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method.
   */
  double        lift_coefficient;

  /*!
   * @brief A table containing the step number, current dimensionless
   * time, the @ref pressure_difference, the @ref drag_coefficient and 
   * the @ref lift_coefficient.
   */
  TableHandler  data_table;

  /*!
   * @brief The default constructor of the struct
   */
  DFG();

  /*!
   * @brief The method computes the @ref pressure_difference.
   */
  void compute_pressure_difference(
    const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure);

  /*!
   * @brief This method computes the @ref drag_force and @ref lift_force
   * and their respective coefficients.
   * @details It computes the dimensionless force around the cylinder
   * given by
   * \f[
   * \boldsymbol{\tilde{\mathfrak{f}}} = \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} (\tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} + \mathbf{\tilde{v}} \otimes \tilde{\nabla}) 
   *  ] \cdot \mathbf{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   * and the @ref drag_coefficient and @ref lift_coefficient as
   * \f[
   * c_\textrm{drag} = 2 \boldsymbol{\tilde{\mathfrak{f}}} \cdot 
   * \mathbf{e}_\textrm{x},
   * \qquad
   * c_\textrm{lift} = 2 \boldsymbol{\tilde{\mathfrak{f}}} \cdot 
   * \mathbf{e}_\textrm{y}
   * \f]
   */
  void compute_drag_and_lift_forces_and_coefficients
  (const std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
   const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure);

  /*!
   * @brief A method that updates @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and 
   * the @ref lift_coefficient.
   */
  void update_table(DiscreteTime  &time);
  /*!
   * @brief A method that prints @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and 
   * the @ref lift_coefficient.
   */
  void print_step_data(DiscreteTime &time);
  /*!
   * @brief A method that outputs the @ref data_table into a file.
   */
  void write_table_to_file(const std::string &file);
};

/*! @attention I kept getting a "trying to access private member" error
    while outputing to the terminal with the Stream object. After
    reading posts in StackOverflow this worked. I do not know if this is
    the best method as there is a bug in the source file */

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
  MIT(std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
      std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
      std::shared_ptr<Entities::ScalarEntity<dim>>  &temperature,
      TimeDiscretization::VSIMEXMethod              &time_stepping,
      const std::shared_ptr<Mapping<dim>>           external_mapping
                                = std::shared_ptr<Mapping<dim>>(),
      const std::shared_ptr<ConditionalOStream>     external_pcout
                                = std::shared_ptr<ConditionalOStream>(),
      const std::shared_ptr<TimerOutput>            external_timer 
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
   * @attention I had to change the name of the second template variable
   * or else I would get a compilation error.
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
  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  /*!
   * @brief A shared pointer to the pressure field's numerical
   * representation.
   */
  std::shared_ptr<Entities::ScalarEntity<dim>>  pressure;

  /*!
   * @brief A shared pointer to the temperature field's numerical
   * representation.
   */
  std::shared_ptr<Entities::ScalarEntity<dim>>  temperature;

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
   * @brief The velocity vector at the sample point \f$ P_1 \f$;.
   */
  Tensor<1,dim>                                 velocity_at_p1;

  /*!
   * @brief The temperature at the sample point \f$ P_1 \f$;.
   */
  double                                        temperature_at_p1;

  /*!
   * @brief The stream function at the sample point \f$ P_1 \f$;.
   */
  double                                        stream_function_at_p1;

  /*!
   * @brief The vorticity norm at the sample point \f$ P_1 \f$;.
   */
  double                                        vorticity_at_p1;

  /*!
   * @brief The Nusselt number at the left and right walls.
   * @details They are given by
   * \f[
   * \mathit{Nu}_{0,1} = \dfrac{1}{H} \int_{\Gamma_{0,1}} \nabla \vartheta 
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
   * \mathit{Nu}_{0,1} = \dfrac{1}{H} \int_{\Gamma_{0,1}} \nabla \vartheta 
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

} // namespace BenchmarkData
} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */
