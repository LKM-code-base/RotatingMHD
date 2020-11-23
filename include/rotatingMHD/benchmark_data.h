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
