
#ifndef INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_
#define INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_

#include <rotatingMHD/auxiliary_functions.h>
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
  double        Re;
  /*!
   * @brief The point at the front side of the cylinder, at which the
   * pressure will be evaluated.
   * @details The point is located at 
   * \f$ P_\textrm{front} = (0.15,\, 0.20) \f$.
   * @attention The struct defines the coordinates in their
   * dimensionless form in the constructor using the characteristic
   * length.
   */
  Point<dim>    front_evaluation_point;
  /*!
   * @brief The point at the rear side of the cylinder, at which the
   * pressure will be evaluated.
   * @details The point is located at 
   * \f$ P_\textrm{front} = (0.25,\, 0.20) \f$.
   * @attention The struct defines the coordinates in their
   * dimensionless form in the constructor using the characteristic
   * length.
   */
  Point<dim>    rear_evaluation_point;
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
   * @brief The pressure evaluated at the @ref front_evaluation_point.
   */
  double        front_point_pressure_value;
  /*!
   * @brief The pressure evaluated at the @ref rear_evaluation_point.
   */
  double        rear_point_pressure_value;
  /*!
   * @brief The drag force around the cylinder. 
   * @details Defined as the \f$ x \f$ component of the total force
   * exorted on the cylinder. The total force is given by
   * \f[
	 * \bs{\mathfrak{f}} = \mathfrak{f}_\textrm{drag}  \mathbf{e}_\textrm{x} 
   * +  \mathfrak{f}_\textrm{lift} \mathbf{e}_\textrm{y} = 
   * \int_{\Gamma_3} [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] 
   * \cdot \mathbf{n} \d \ell
   * \f]
   * It dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method,
   * is given then by
   * \f[
   * \bs{\tilde{\mathfrak{f}}} = 
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} \tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} ] \cdot \mathbf{n} \d{\tilde{\ell}}
   * \f]
   */
  double        drag_force;
  /*!
   * @brief The drag coefficient.
   * @details Defined as 
   * \f[
   * c_\textrm{drag} = \dfrac{2}{\rho\bar{v}^2D} \bs{\mathfrak{f}} \cdot 
   * \mathbf{e}_\textrm{x}
   * \f]
   * which is equivalent to \f$ c_\textrm{drag} = 2
   * \bs{\tilde{\mathfrak{f}}} \cdot \mathbf{e}_\textrm{x} \f$, 
   * what is actually computed by the by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method.
   */
  double        drag_coefficient;
  /*!
   * @brief The lift force around the cylinder. 
   * @details Defined as the \f$ y \f$ component of the total force
   * exorted on the cylinder. The total force is given by
   * \f[
	 * \bs{\mathfrak{f}} = \mathfrak{f}_\textrm{drag}  \mathbf{e}_\textrm{x} 
   * +  \mathfrak{f}_\textrm{lift} \mathbf{e}_\textrm{y} = 
   * \int_{\Gamma_3} [-p\mathbf{I} + \mu \nabla \otimes \mathbf{v}] 
   * \cdot \mathbf{n} \d \ell
   * \f]
   * It dimensionless form, which is what it is actually computed and
   * stored in this member by the 
   * @ref compute_drag_and_lift_forces_and_coefficients method,
   * is given then by
   * \f[
   * \bs{\tilde{\mathfrak{f}}} = 
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} \tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} ] \cdot \mathbf{n} \d{\tilde{\ell}}
   * \f]
   */
  double        lift_force;
  /*!
   * @brief The lift coefficient.
   * @details Defined as 
   * \f[
   * c_\textrm{lift} = \dfrac{2}{\rho\bar{v}^2D} \bs{\mathfrak{f}} \cdot 
   * \mathbf{e}_\textrm{y}
   * \f]
   * which is equivalent to \f$ c_\textrm{lift} = 2
   * \bs{\tilde{\mathfrak{f}}} \cdot \mathbf{e}_\textrm{y} \f$, 
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
    const Entities::ScalarEntity<dim> &pressure);
  /*!
   * @brief This method computes the @ref drag_force and @ref lift_force
   * and their respective coefficients.
   * @details It computes the dimensionless force around the cylinder
   * given by
   * \f[
   * \bs{\tilde{\mathfrak{f}}} = \mathfrak{\tilde{f}}_\textrm{drag}
   *  \mathbf{e}_\textrm{x} 
   * +  \mathfrak{\tilde{f}}_\textrm{lift} \mathbf{e}_\textrm{y} =
   *  \int_{\tilde{\Gamma}_3} [-\tilde{p}\mathbf{I} +
   *  (\rho\textrm{Re})^{-1} \tilde{\nabla} \otimes 
   *  \mathbf{\tilde{v}} ] \cdot \mathbf{n} \d{\tilde{\ell}}
   * \f]
   * and the @ref drag_coefficient and @ref lift_coefficient as
   * \f[
   * c_\textrm{drag} = 2 \bs{\tilde{\mathfrak{f}}} \cdot 
   * \mathbf{e}_\textrm{x},
   * \qquad
   * c_\textrm{lift} = 2 \bs{\tilde{\mathfrak{f}}} \cdot 
   * \mathbf{e}_\textrm{y}
   * \f]
   */
  void compute_drag_and_lift_forces_and_coefficients(
                          const Entities::VectorEntity<dim> &velocity,
                          const Entities::ScalarEntity<dim> &pressure);
  /*!
   * @brief A method that updates @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and 
   * the @ref lift_coefficient.
   */
  void update_table(DiscreteTime          &time);
  /*!
   * @brief A method that prints @ref data_table with the step number,
   * the current dimensionless time, @ref pressure_difference,
   * @ref drag_coefficient and 
   * the @ref lift_coefficient.
   */
  void print_step_data(DiscreteTime       &time);
  /*!
   * @brief A method that outputs the @ref data_table into a file.
   */
  void write_table_to_file(const std::string &file);
};

} // namespace BenchmarkData
} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_DFG_BENCHMARK_DATA_H_ */