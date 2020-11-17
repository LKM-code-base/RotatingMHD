#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <rotatingMHD/entities_structs.h>

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

namespace RMHD
{

using namespace dealii;

namespace BenchmarkData
{

/*!
 * @struct DFG
 * 
 * @brief A structure containing the data requested by the DFG benchmark
 * and methods to compute them.
 *
 */
template <int dim>
struct DFG
{

  /*!
   *
   * @brief The default constructor of the structure.
   *
   * @attention Shouldn't there be some input arguments?
   *
   */
  DFG();

  /*!
   * @brief The characteristic length of the benchmark is set as the 
   * diameter of the cylinder, \f$ d =0.1 \f$.
   *
   * @attention SG thinks that this variable is not really because the problem
   * formulated in dimensionless form.
   *
   */
  double        characteristic_length;

  /*!
   * @brief The mean velocity of the fluid.

   * @details Defined as the  
   * \f[
   * \bar{v} = \frac{2}{3} \bm{v}{v}_\text{inflow}\big|_{(0,0.5H)}
   * \ex = \frac{2}{3} \left(
   * \frac{4v_\text{max}y(H-y)}{H^2} \right)\bigg|_{(0,0.5H)}
   * \f]
   * which with \f$ v_\text{max} = 1.5 \f$ equals to \f$ 1.0 \f$.
   *
   */
  double        mean_velocity;

  /*!
   * @brief The kinematic viscosity of the fluid, \f$ \nu = 0.001 \f$.
   *
   * @attention SG thinks that this variable is not really because the problem
   * formulated in dimensionless form.
   *
   */
  double        kinematic_viscosity;

  /*!
   * @brief The Reynolds number of the problem.
   *
   * @details Its value is defined as
   * \f[
   *   \Reynolds = \frac{\bar{v}d}{\nu},
   * \f]
   * where \f$ d \f$ denotes the diameter of the cylinder, \f$ \bar{v} \f$ the
   * mean velocity at the inlet and \f$ \nu \f$ the kinematic viscosity.
   */
  const double  Re;

  /*!
   * @brief The point \f$ \mathrm{P}_\text{front} \f$ at the front side of the
   * cylinder, at which the pressure will be evaluated.
   *
   * @details The point \f$ \mathrm{P}_\text{front} \f$ is located at the position
   * \f$ \bm{x}_\text{rear} = 0.15 \ex + 0.20 \ey \f$.
   *
   *
   * @attention The coordinates are defined in dimensionless form from
   * the characteristic length when the constructor is called.
   */
  const Point<dim>  front_evaluation_point;

  /*!
   * @brief The point \f$ \mathrm{P}_\text{rear} \f$ at the rear side of the
   * cylinder, at which the pressure will be evaluated.
   *
   * @details The point \f$ \mathrm{P}_\text{front} \f$ is located at the
   * position
   * \f$ \bm{x}_\text{rear} = 0.25 \ex + 0.20 \ey \f$.
   *
   * @attention The coordinates are defined in dimensionless form from
   * the characteristic length when the constructor is called.
   *
   */
  const Point<dim>  rear_evaluation_point;

  /*!
   * @brief The pressure difference between the front an the rear 
   * evaluation points.
   *
   * @details Defined as
   * \f[
   * \Delta p = p|_{\bm{x}_\text{front}} - p|_{\bm{x}_\text{rear}}
   * \f]
   *
   * @attention The computed pressure difference is dimensionless, but
   * it is interchangeable with the actual pressure.
   *
   */
  double        pressure_difference;

  /*!
   * @brief The drag force acting on the cylinder.
   *
   * @details Defined as the \f$ x \f$-component of the resulting force
   * acting on the cylinder. The resulting force is given by
   * \f[
  * \bm{F} = F_\text{drag}  \ex
   * +  F_\text{lift} \ey =
   * \int_{\Gamma_3} [-p \bm{1} + \mu \nabla \otimes \bm{v}]
   * \cdot \bm{n} \,\mathrm{d}{A}
   * \f]
   * Its dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method and 
   * considering the formulation of the NavierStokesProjection class,
   * is given then by
   * \f[
   * \tilde{\bm{F}} =
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\bm{1} +
   *  (\rho\text{Re})^{-1} (\tilde{\nabla} \otimes
   *  \tilde{\bm{v}} + \tilde{\bm{v}} \otimes \tilde{\nabla})
   *  ] \cdot \bm{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   */
  double        drag_force;

  /*!
   * @brief The drag coefficient.
   * @details Defined as 
   * \f[
   * c_\text{drag} = \dfrac{2}{\rho\bar{v}^2D} \bm{F} \cdot
   * \ex
   * \f]
   * which is equivalent to \f$ c_\text{drag} = 2
   * \tilde{\bm{F}} \cdot \ex \f$,
   * what is actually computed by the by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method.
   */
  double        drag_coefficient;

  /*!
   * @brief The lift force acting on the cylinder.
   *
   * @details Defined as the \f$ y \f$ component of the total force
   * exorted on the cylinder. The total force is given by
   * \f[
	 * \bm{F} = F_\text{d} \ex + F_\text{l} \ey =
   * \int_{\Gamma_3} [-p\bm{1} + \mu \nabla \otimes \bm{v}]
   * \cdot \bm{n} \mathrm{d} \ell
   * \f]
   *
   * In dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method and 
   * considering the formulation of the NavierStokesProjection class,
   * is given then by
   * \f[
   * \tilde{\bm{F}} =
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\bm{1} +
   *  \frac{1}{\Reynolds} (\tilde{\nabla} \otimes
   *  \tilde{\bm{v}} + \tilde{\bm{v}} \otimes \tilde{\nabla})
   *  ] \cdot \bm{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   */
  double        lift_force;

  /*!
   * @brief The lift coefficient.
   *
   * @details The lift coefficient is defined as
   *
   * \f[
   * c_\text{l} = \dfrac{2}{\rho\bar{v}^2D} \bm{F} \cdot \ey
   * \f]
   *
   * which is equivalent to the following dimensionless value
   *
   * \f$ c_\text{l} = 2 \tilde{\bm{F}} \cdot \ey \f$,
   *
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
   * @brief The method computes the @ref pressure_difference.
   */
  void compute_pressure_difference
  (const std::shared_ptr<Entities::ScalarEntity<dim>> &pressure);

  /*!
   * @brief This method computes the @ref drag_force and @ref lift_force
   * and their respective coefficients.
   *
   * @details It computes the dimensionless force around the cylinder
   * given by
   *
   * \f[
   * \tilde{\bm{F}} = \int_{\tilde{\Gamma}_3} [-\tilde{p}\bm{1} +
   *  ( \frac{1}{\Reynolds} (\tilde{\nabla} \otimes
   *  \tilde{\bm{v}} + \tilde{\bm{v}} \otimes \tilde{\nabla})
   *  ] \cdot \bm{n} \mathrm{d}{\tilde{\ell}}
   * \f]
   *
   * and the @ref drag_coefficient and @ref lift_coefficient as
   *
   * \f[
   * c_\text{d} = 2 \tilde{\bm{F}} \cdot \ex\,, \qquad
   * c_\text{l} = 2 \tilde{\bm{F}} \cdot \ey\,.
   * \f]
   *
   */
  void compute_drag_and_lift_forces_and_coefficients
  (const std::shared_ptr<Entities::VectorEntity<dim>>  &velocity,
   const std::shared_ptr<Entities::ScalarEntity<dim>>  &pressure,
   const types::boundary_id                             cylinder_boundary_id = 2);

  /*!
   * @brief A method that updates @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and the @ref lift_coefficient.
   */
  void update_table(DiscreteTime  &time);

  /*!
   * @brief A method that prints @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and the @ref lift_coefficient.
   */
  void print_step_data(DiscreteTime &time);

  /*!
   * @brief A method that outputs the @ref data_table into a file.
   */
  void write_table_to_file(const std::string &file);
};

} // namespace BenchmarkData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */
